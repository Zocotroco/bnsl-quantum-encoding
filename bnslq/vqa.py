"""Algoritmos variacionales sobre el espacio de órdenes.

En la Fase 1 el coste era un polinomio en las variables de qubit y podía
sintetizarse entero dentro del circuito: eso es QAOA canónico.  Aquí no.  La
puntuación de un orden contiene un máximo sobre subconjuntos de padres y sólo se
puede evaluar clásicamente sobre cada solución medida.  El circuito, por tanto,
ya no lleva la función objetivo completa, y eso abre una pregunta de diseño que
es el núcleo de esta fase:

**¿qué debe hacer el circuito, si no puede llevar el coste?**

Se comparan dos respuestas:

1. `qaoa_transitividad`: QAOA cuyo operador de coste es sólo H_trans.  El
   circuito no sabe nada de la puntuación, pero concentra amplitud sobre el
   subespacio **factible** (los n! órdenes válidos de entre 2^(n(n-1)/2)
   asignaciones).  La puntuación entra únicamente por el CVaR clásico.

2. `hardware_efficient`: un ansatz genérico de rotaciones Ry y anillos de CNOT,
   sin ninguna información del problema.  Toda la guía viene del bucle clásico.

La comparación dice cuánto aporta meter la restricción en el circuito frente a
dejársela entera al optimizador.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .orderings import OrderingSpace
from .qaoa import QAOARunner, cost_layer, cvar

Arc = Tuple[str, str]


# --------------------------------------------------------------------------
# Ansätze
# --------------------------------------------------------------------------

def qaoa_transitividad(space: OrderingSpace, p: int) -> Tuple[QuantumCircuit, ParameterVector]:
    """QAOA de p capas cuyo operador de coste es H_trans normalizado.

    Se divide H_trans por delta_trans para que los ángulos gamma queden en un
    rango natural: si no, la escala de la penalización se cuela en el paso del
    optimizador y el paisaje se vuelve casi aleatorio.
    """
    terms, _ = space.h_trans.to_ising()
    terms = {q: c / space.delta_trans for q, c in terms.items()}
    n = space.num_vars
    params = ParameterVector("θ", 2 * p)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for capa in range(p):
        cost_layer(qc, terms, params[capa])
        qc.barrier()
        qc.rx(2.0 * params[p + capa], range(n))
        qc.barrier()
    qc.measure(range(n), range(n))
    return qc, params


def hardware_efficient(space: OrderingSpace, layers: int) -> Tuple[QuantumCircuit, ParameterVector]:
    """Ansatz genérico: capas de Ry seguidas de un anillo de CNOTs."""
    n = space.num_vars
    params = ParameterVector("θ", (layers + 1) * n)
    qc = QuantumCircuit(n, n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(params[k], q)
            k += 1
        for q in range(n):
            qc.cx(q, (q + 1) % n)
        qc.barrier()
    for q in range(n):
        qc.ry(params[k], q)
        k += 1
    qc.measure(range(n), range(n))
    return qc, params


ANSATZE: Dict[str, Callable] = {
    "qaoa": qaoa_transitividad,
    "hardware_efficient": hardware_efficient,
}


# --------------------------------------------------------------------------
# Resultado y bucle
# --------------------------------------------------------------------------

@dataclass
class VQAResult:
    best_bits: str
    best_cost: float
    best_order: Optional[Tuple[str, ...]]
    best_arcs: List[Arc]
    optimal_params: np.ndarray
    history: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    seconds: float = 0.0
    feasible: bool = False
    feasible_fraction_initial: float = 0.0
    feasible_fraction_final: float = 0.0


class OrderingVQA:
    """Bucle híbrido sobre `OrderingSpace` con el ansatz elegido."""

    def __init__(
        self,
        space: OrderingSpace,
        ansatz: str = "qaoa",
        depth: int = 2,
        alpha: float = 0.25,
        shots: int = 1024,
        seed: int = 0,
        optimizer: str = "COBYLA",
        maxiter: int = 200,
        simulator: Optional[AerSimulator] = None,
    ):
        if ansatz not in ANSATZE:
            raise ValueError(f"ansatz desconocido: {ansatz!r}; use {list(ANSATZE)}")
        self.space = space
        self.ansatz_name = ansatz
        self.depth = depth
        self.alpha = alpha
        self.shots = shots
        self.seed = seed
        self.optimizer = optimizer
        self.maxiter = maxiter

        self.num_qubits = space.num_vars
        self.backend = simulator or QAOARunner.default_simulator(self.num_qubits, seed)
        circuit, self.params = ANSATZE[ansatz](space, depth)
        self.circuit = transpile(circuit, self.backend, optimization_level=1)

    # ------------------------------------------------------------ evaluación

    def _to_x(self, bitstring: str) -> str:
        """Qiskit devuelve el qubit 0 a la derecha; aquí se usa a la izquierda."""
        return bitstring[::-1]

    def sample(self, params: Sequence[float]) -> Dict[str, int]:
        bound = self.circuit.assign_parameters(dict(zip(list(self.params), list(params))))
        res = self.backend.run(bound, shots=self.shots, seed_simulator=self.seed).result()
        return res.get_counts()

    def feasible_fraction(self, counts: Dict[str, int]) -> float:
        """Proporción de disparos que caen sobre un orden total válido.

        Es la métrica que separa los dos ansätze: mide si el circuito está
        concentrando amplitud sobre el subespacio factible o si se limita a
        muestrear el hipercubo y deja el trabajo al optimizador clásico.
        """
        total = sum(counts.values())
        buenos = sum(
            n for b, n in counts.items() if self.space.is_total_order(self._to_x(b))
        )
        return buenos / total if total else 0.0

    def initial_params(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0, np.pi, len(self.params))

    # ----------------------------------------------------------------- bucle

    def run(self, x0: Optional[Sequence[float]] = None) -> VQAResult:
        x0 = np.asarray(x0 if x0 is not None else self.initial_params(), dtype=float)
        historial: List[float] = []
        mejor = {"bits": None, "cost": float("inf")}
        fraccion_inicial = self.feasible_fraction(self.sample(x0))

        def objetivo(params):
            counts = self.sample(params)
            claves = list(counts)
            costes = np.array([self.space.cost(self._to_x(b)) for b in claves])
            pesos = np.array([counts[b] for b in claves], dtype=float)
            i = int(np.argmin(costes))
            if costes[i] < mejor["cost"]:
                mejor.update(bits=claves[i], cost=float(costes[i]))
            valor = cvar(costes, pesos, self.alpha)
            historial.append(valor)
            return valor

        inicio = time.time()
        res = minimize(objetivo, x0, method=self.optimizer,
                       options={"maxiter": self.maxiter})
        transcurrido = time.time() - inicio

        counts = self.sample(res.x)
        for b in counts:
            c = self.space.cost(self._to_x(b))
            if c < mejor["cost"]:
                mejor.update(bits=b, cost=c)

        x = self._to_x(mejor["bits"])
        orden = self.space.decode_order(x)
        arcos = self.space.dag_of(x)[0] if orden else []
        return VQAResult(
            best_bits=x,
            best_cost=mejor["cost"],
            best_order=orden,
            best_arcs=arcos,
            optimal_params=res.x,
            history=historial,
            n_evaluations=len(historial),
            seconds=transcurrido,
            feasible=orden is not None,
            feasible_fraction_initial=fraccion_inicial,
            feasible_fraction_final=self.feasible_fraction(counts),
        )

    def circuit_stats(self) -> Dict[str, int]:
        ops = self.circuit.count_ops()
        return {
            "qubits": self.num_qubits,
            "parametros": len(self.params),
            "profundidad": self.circuit.depth(),
            "puertas_2q": int(ops.get("cx", 0)),
            "puertas_totales": int(sum(ops.values())),
        }
