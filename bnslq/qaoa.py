"""QAOA para el BNSL, siguiendo Soloviev, Bielza y Larrañaga (2022).

El circuito tiene la estructura estándar de Farhi et al. (2014): superposición
uniforme, y luego p capas alternando el operador de coste U(H_C, gamma) y el
mezclador U(H_B, beta).

Dos detalles que conviene tener presentes
-----------------------------------------
* **No se cuadratiza.**  Un término de coste de grado k se sintetiza como una
  escalera de CNOTs alrededor de un Rz, que es exacta.  La cuadratización sólo
  es obligatoria en un *annealer*, cuyo hardware es literalmente cuadrático.
  Con m = 2 la cuestión es discutible; con m >= 3 esto ahorra las variables
  auxiliares que necesitaría un D-Wave.

* **Se optimiza CVaR, no la media.**  El valor esperado de la energía promedia
  sobre todas las soluciones medidas, incluidas las malas.  El CVaR_alpha
  (Barkoutsos et al., 2020) promedia sólo el alpha por ciento mejor, lo que en
  la práctica permite alcanzar los mismos resultados con menos capas.  Con
  alpha = 1 se recupera el valor esperado ordinario.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .qubo import BNSLHamiltonian


# --------------------------------------------------------------------------
# Circuito
# --------------------------------------------------------------------------

def cost_layer(circuit: QuantumCircuit, terms: Dict[Tuple[int, ...], float], gamma) -> None:
    """Aplica exp(-i * gamma * H_C) para un H_C diagonal de grado arbitrario."""
    for qubits, coef in terms.items():
        if len(qubits) == 1:
            circuit.rz(2.0 * coef * gamma, qubits[0])
        else:
            for a, b in zip(qubits[:-1], qubits[1:]):
                circuit.cx(a, b)
            circuit.rz(2.0 * coef * gamma, qubits[-1])
            for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
                circuit.cx(a, b)


def build_qaoa_circuit(
    terms: Dict[Tuple[int, ...], float], num_qubits: int, p: int
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Circuito QAOA parametrizado de p capas, con medición en la base Z."""
    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for layer in range(p):
        cost_layer(qc, terms, gammas[layer])
        qc.barrier()
        qc.rx(2.0 * betas[layer], range(num_qubits))
        qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc, gammas, betas


# --------------------------------------------------------------------------
# Función objetivo
# --------------------------------------------------------------------------

def cvar(costs: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """CVaR_alpha: media del alpha por ciento de menor coste.

    `weights` son las frecuencias de cada solución medida.  Con alpha = 1
    devuelve la media ponderada, es decir el valor esperado de la energía.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha debe estar en (0, 1]")
    order = np.argsort(costs)
    costs, weights = costs[order], weights[order]
    total = weights.sum()
    objetivo = alpha * total
    acumulado = np.cumsum(weights)
    corte = int(np.searchsorted(acumulado, objetivo)) + 1
    corte = min(corte, len(costs))
    w = weights[:corte].astype(float).copy()
    exceso = acumulado[corte - 1] - objetivo
    if exceso > 0:
        w[-1] -= exceso
    return float(np.dot(costs[:corte], w) / w.sum())


@dataclass
class QAOAResult:
    """Resultado de una ejecución completa del bucle híbrido."""

    best_bits: str
    best_cost: float
    best_arcs: List[Tuple[str, str]]
    optimal_params: np.ndarray
    history: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    seconds: float = 0.0
    feasible: bool = False
    final_counts: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Bucle híbrido
# --------------------------------------------------------------------------

class QAOARunner:
    """Bucle cuántico-clásico completo sobre un `BNSLHamiltonian`."""

    def __init__(
        self,
        hamiltonian: BNSLHamiltonian,
        p: int = 3,
        alpha: float = 0.5,
        shots: int = 1024,
        seed: int = 0,
        optimizer: str = "COBYLA",
        maxiter: int = 200,
        simulator: Optional[AerSimulator] = None,
    ):
        self.H = hamiltonian
        self.p = p
        self.alpha = alpha
        self.shots = shots
        self.seed = seed
        self.optimizer = optimizer
        self.maxiter = maxiter

        self.terms, self.offset = hamiltonian.poly.to_ising()
        self.num_qubits = hamiltonian.num_vars
        self.backend = simulator or self.default_simulator(self.num_qubits, seed)
        circuit, self.gammas, self.betas = build_qaoa_circuit(
            self.terms, self.num_qubits, p
        )
        self.circuit = transpile(circuit, self.backend, optimization_level=1)
        self._cost_cache: Dict[str, float] = {}

    @staticmethod
    def default_simulator(num_qubits: int, seed: int = 0) -> AerSimulator:
        """Elige el método de simulación según el tamaño del registro.

        El vector de estado es exacto y muy rápido mientras quepa en memoria:
        2^n amplitudes complejas, es decir ~1 GB con 26 qubits, 4,3 GB con 28 y
        17 GB con 30.  Por encima hay que recurrir a *matrix product states*,
        que representa bien los estados poco entrelazados pero se degrada justo
        cuando el QAOA entrelaza de verdad: medido en este proyecto, con 18
        qubits y p=3 es unas 100 veces más lento que el vector de estado, y con
        28 qubits el sondeo que debía tardar minutos no terminó en veinte.

        El umbral está en 28 porque es lo que cabe con holgura en los 17 GB de
        este equipo.  **Hay que bajarlo en máquinas con menos memoria**: pasarse
        provoca intercambio a disco, que es peor que MPS.
        """
        method = "statevector" if num_qubits <= 28 else "matrix_product_state"
        return AerSimulator(method=method, seed_simulator=seed)

    # ------------------------------------------------------------ evaluación

    def _bits_to_x(self, bitstring: str) -> str:
        """Qiskit devuelve el qubit 0 a la derecha; aquí se usa a la izquierda."""
        return bitstring[::-1]

    def _cost_of(self, bitstring: str) -> float:
        hit = self._cost_cache.get(bitstring)
        if hit is None:
            hit = self.H.classical_cost(self._bits_to_x(bitstring))
            self._cost_cache[bitstring] = hit
        return hit

    def sample(self, params: Sequence[float]) -> Dict[str, int]:
        binding = dict(zip(list(self.gammas) + list(self.betas), list(params)))
        bound = self.circuit.assign_parameters(binding)
        result = self.backend.run(bound, shots=self.shots, seed_simulator=self.seed).result()
        return result.get_counts()

    def objective(self, params: Sequence[float]) -> float:
        counts = self.sample(params)
        keys = list(counts)
        costs = np.array([self._cost_of(k) for k in keys])
        weights = np.array([counts[k] for k in keys], dtype=float)
        return cvar(costs, weights, self.alpha)

    # ----------------------------------------------------------------- bucle

    def run(self, x0: Optional[Sequence[float]] = None) -> QAOAResult:
        rng = np.random.default_rng(self.seed)
        if x0 is None:
            # gamma pequeño: los coeficientes del Hamiltoniano no están
            # normalizados y ángulos grandes hacen el paisaje casi aleatorio.
            escala = max(abs(c) for c in self.terms.values())
            x0 = np.concatenate(
                [rng.uniform(0, np.pi / escala, self.p), rng.uniform(0, np.pi, self.p)]
            )
        history: List[float] = []
        best = {"bits": None, "cost": float("inf")}

        def wrapped(params):
            counts = self.sample(params)
            keys = list(counts)
            costs = np.array([self._cost_of(k) for k in keys])
            weights = np.array([counts[k] for k in keys], dtype=float)
            i = int(np.argmin(costs))
            if costs[i] < best["cost"]:
                best.update(bits=keys[i], cost=float(costs[i]))
            value = cvar(costs, weights, self.alpha)
            history.append(value)
            return value

        inicio = time.time()
        res = minimize(
            wrapped,
            np.asarray(x0, dtype=float),
            method=self.optimizer,
            options={"maxiter": self.maxiter},
        )
        transcurrido = time.time() - inicio

        counts = self.sample(res.x)
        for k in counts:
            c = self._cost_of(k)
            if c < best["cost"]:
                best.update(bits=k, cost=c)

        x = self._bits_to_x(best["bits"])
        return QAOAResult(
            best_bits=x,
            best_cost=best["cost"],
            best_arcs=self.H.decode(x),
            optimal_params=res.x,
            history=history,
            n_evaluations=len(history),
            seconds=transcurrido,
            feasible=self.H.is_feasible(x),
            final_counts=counts,
        )

    def circuit_stats(self) -> Dict[str, int]:
        ops = self.circuit.count_ops()
        return {
            "qubits": self.num_qubits,
            "profundidad": self.circuit.depth(),
            "puertas_2q": int(ops.get("cx", 0)),
            "puertas_totales": int(sum(ops.values())),
            "terminos_ising": len(self.terms),
        }
