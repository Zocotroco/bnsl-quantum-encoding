"""Inferencia sobre una red bayesiana codificada en un circuito cuántico.

Cierra el ciclo del trabajo. Una vez aprendida la estructura y ajustadas las
CPTs, `representacion.py` construye un circuito A tal que

    A|0...0> = sum_x sqrt(P(x)) |x>.

Sobre ese estado, calcular P(E) para un conjunto de condiciones E se reduce a
estimar la **amplitud** del subespacio que las cumple. Y ahí es donde la
estimación cuántica de amplitud (QAE) ofrece su ventaja: alcanza precisión
epsilon con O(1/epsilon) llamadas al operador, frente a las O(1/epsilon^2) que
necesita el muestreo clásico \\citep{brassard2002}.

La probabilidad condicional se obtiene como cociente,

    P(Q | E) = P(Q and E) / P(E),

estimando numerador y denominador por separado.

Nota sobre el alcance: la ventaja cuadrática es de **estimación**, no de
inferencia en el sentido de la eliminación de variables. El circuito ha de
construirse a partir de las CPTs, que es un paso clásico, y su profundidad crece
con el número de padres.
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XGate

from .representacion import RedCuantica


class InferenciaCuantica:
    """Consultas probabilísticas sobre el circuito de una red bayesiana."""

    def __init__(self, red: RedCuantica):
        self.red = red
        self._exacta = red.distribucion_exacta()

    # ------------------------------------------------------------- clásico

    def exacta(self, consulta: Dict[str, int],
               evidencia: Optional[Dict[str, int]] = None) -> float:
        """P(consulta | evidencia) por enumeración. Referencia de control."""
        evidencia = evidencia or {}
        idx = self.red.indice
        num = den = 0.0
        for x, p in self._exacta.items():
            if all(x[idx[v]] == b for v, b in evidencia.items()):
                den += p
                if all(x[idx[v]] == b for v, b in consulta.items()):
                    num += p
        return num / den if den else float("nan")

    # ----------------------------------------------------- circuito marcado

    def circuito_marcado(self, condiciones: Dict[str, int]) -> QuantumCircuit:
        """Circuito de la red más un qubit bandera que vale 1 si se cumple E.

        La bandera hace falta porque la estimación de amplitud marca los estados
        ``buenos`` mirando un único qubit, mientras que una evidencia es una
        conjunción sobre varios.
        """
        n = self.red.n
        base = self.red.circuito(medir=False)
        qc = QuantumCircuit(n + 1)
        qc.compose(base, qubits=range(n), inplace=True)
        controles = [self.red.indice[v] for v in condiciones]
        estado = "".join(str(condiciones[v]) for v in condiciones)[::-1]
        qc.append(XGate().control(len(controles), ctrl_state=estado),
                  controles + [n])
        return qc

    # ---------------------------------------------------------------- QAE

    def probabilidad_qae(self, condiciones: Dict[str, int],
                         epsilon: float = 0.01, alpha: float = 0.05,
                         disparos: int = 1024, seed: int = 0
                         ) -> Tuple[float, int]:
        """Estima P(condiciones) con QAE iterativa.

        Devuelve la estimación y el **número de aplicaciones del operador de
        preparación A**, que es la magnitud comparable con los disparos del
        muestreo: en la ronda de potencia k el circuito aplica A una vez y el
        operador de Grover k veces, lo que supone 2k+1 aplicaciones de A.

        El atributo `num_oracle_queries` de Qiskit cuenta sólo las aplicaciones
        de Grover, y vale cero si la primera ronda ya converge, de modo que no
        sirve como medida de coste por sí solo.
        """
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

        problema = EstimationProblem(
            state_preparation=self.circuito_marcado(condiciones),
            objective_qubits=[self.red.n],
        )
        iae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha,
            sampler=StatevectorSampler(default_shots=disparos, seed=seed))
        res = iae.estimate(problema)
        coste = disparos * sum(2 * k + 1 for k in res.powers)
        return float(res.estimation), int(coste)

    def condicional_qae(self, consulta: Dict[str, int], evidencia: Dict[str, int],
                        epsilon: float = 0.01) -> Tuple[float, int]:
        """P(consulta | evidencia) como cociente de dos estimaciones."""
        conjunta = dict(evidencia)
        conjunta.update(consulta)
        p_num, c1 = self.probabilidad_qae(conjunta, epsilon)
        p_den, c2 = self.probabilidad_qae(evidencia, epsilon)
        return (p_num / p_den if p_den else float("nan")), c1 + c2

    # ----------------------------------------------------------- muestreo

    def probabilidad_muestreo(self, condiciones: Dict[str, int],
                              disparos: int = 10000, seed: int = 0
                              ) -> Tuple[float, int]:
        """Estima P(condiciones) midiendo el circuito. Coste: `disparos`."""
        from qiskit_aer import AerSimulator

        sim = AerSimulator(seed_simulator=seed)
        qc = transpile(self.red.circuito(), sim, optimization_level=1)
        counts = sim.run(qc, shots=disparos).result().get_counts()
        idx = self.red.indice
        buenos = 0
        for cadena, n in counts.items():
            bits = cadena[::-1]
            if all(int(bits[idx[v]]) == b for v, b in condiciones.items()):
                buenos += n
        return buenos / disparos, disparos
