"""Redes bayesianas de referencia y muestreo de datos.

`Cancer` (n = 5) es el *benchmark* que usan Soloviev, Bielza y Larrañaga (2022)
para evaluar QAOA sobre BNSL, y por eso es nuestro caso de control: si nuestra
formulación es correcta, debe recuperar sus 4 arcos.

Los parámetros son los del repositorio bnlearn
(https://www.bnlearn.com/bnrepository/discrete-small.html#cancer).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

Arc = Tuple[str, str]  # (padre, hijo)


class BenchmarkNetwork:
    """Red bayesiana discreta conocida, lista para muestrear."""

    def __init__(self, name: str, arcs: Sequence[Arc], cpds: Dict, states: Dict):
        self.name = name
        self.arcs: List[Arc] = list(arcs)
        self.cpds = cpds
        self.states = states
        self.variables: Tuple[str, ...] = tuple(states)

    @property
    def n(self) -> int:
        return len(self.variables)

    def to_pgmpy(self):
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.models import DiscreteBayesianNetwork

        model = DiscreteBayesianNetwork(self.arcs)
        for var, spec in self.cpds.items():
            evidence = spec.get("evidence", [])
            model.add_cpds(
                TabularCPD(
                    variable=var,
                    variable_card=len(self.states[var]),
                    values=spec["values"],
                    evidence=evidence or None,
                    evidence_card=[len(self.states[e]) for e in evidence] or None,
                )
            )
        model.check_model()
        return model

    def sample(self, size: int, seed: int = 0) -> pd.DataFrame:
        """Muestreo lógico probabilístico (*forward sampling*)."""
        model = self.to_pgmpy()
        df = model.simulate(n_samples=size, seed=seed, show_progress=False)
        return df[list(self.variables)]


# --------------------------------------------------------------------------
# Cancer (Korb & Nicholson).  5 variables binarias, 4 arcos.
#
#   Pollution ---> Cancer <--- Smoker
#                  /    \
#              Xray      Dyspnoea
#
# En todas las CPDs el estado 0 es el "negativo"/"bajo" y el 1 el
# "positivo"/"alto".  El orden de columnas de pgmpy para evidencia [P, S] es
# (P=0,S=0), (P=0,S=1), (P=1,S=0), (P=1,S=1).
# --------------------------------------------------------------------------
CANCER = BenchmarkNetwork(
    name="cancer",
    arcs=[
        ("Pollution", "Cancer"),
        ("Smoker", "Cancer"),
        ("Cancer", "Xray"),
        ("Cancer", "Dyspnoea"),
    ],
    states={
        "Pollution": ["low", "high"],
        "Smoker": ["no", "yes"],
        "Cancer": ["no", "yes"],
        "Xray": ["negative", "positive"],
        "Dyspnoea": ["no", "yes"],
    },
    cpds={
        "Pollution": {"values": [[0.90], [0.10]]},
        "Smoker": {"values": [[0.70], [0.30]]},
        "Cancer": {
            "evidence": ["Pollution", "Smoker"],
            #                P=0,S=0  P=0,S=1  P=1,S=0  P=1,S=1
            "values": [[0.999, 0.970, 0.980, 0.950],   # Cancer = no
                       [0.001, 0.030, 0.020, 0.050]],  # Cancer = yes
        },
        "Xray": {
            "evidence": ["Cancer"],
            "values": [[0.80, 0.10],    # Xray = negative
                       [0.20, 0.90]],   # Xray = positive
        },
        "Dyspnoea": {
            "evidence": ["Cancer"],
            "values": [[0.70, 0.35],    # Dyspnoea = no
                       [0.30, 0.65]],   # Dyspnoea = yes
        },
    },
)


# --------------------------------------------------------------------------
# Red sintética de 4 nodos, útil para desarrollo: cabe en 18 qubits con la
# formulación de O'Gorman y su óptimo se puede calcular por fuerza bruta.
# --------------------------------------------------------------------------
TOY4 = BenchmarkNetwork(
    name="toy4",
    arcs=[("A", "C"), ("B", "C"), ("C", "D")],
    states={v: ["0", "1"] for v in ("A", "B", "C", "D")},
    cpds={
        "A": {"values": [[0.5], [0.5]]},
        "B": {"values": [[0.5], [0.5]]},
        "C": {
            "evidence": ["A", "B"],
            "values": [[0.95, 0.30, 0.30, 0.05],
                       [0.05, 0.70, 0.70, 0.95]],
        },
        "D": {
            "evidence": ["C"],
            "values": [[0.90, 0.15], [0.10, 0.85]],
        },
    },
)

# --------------------------------------------------------------------------
# Survey (Scutari & Denis).  6 variables, 6 arcos, dos de ellas ternarias.
#
#   Age ---> Education <--- Sex
#             /        \
#      Occupation     Residence
#             \        /
#              Travel
#
# Es el siguiente benchmark estándar por tamaño después de Cancer, y sirve para
# comprobar que nada del código asume variables binarias.  En el espacio de
# órdenes son 15 qubits; con la formulación de O'Gorman serían 45.
# --------------------------------------------------------------------------
SURVEY = BenchmarkNetwork(
    name="survey",
    arcs=[
        ("Age", "Education"),
        ("Sex", "Education"),
        ("Education", "Occupation"),
        ("Education", "Residence"),
        ("Occupation", "Travel"),
        ("Residence", "Travel"),
    ],
    states={
        "Age": ["young", "adult", "old"],
        "Sex": ["M", "F"],
        "Education": ["high", "uni"],
        "Occupation": ["emp", "self"],
        "Residence": ["small", "big"],
        "Travel": ["car", "train", "other"],
    },
    cpds={
        "Age": {"values": [[0.30], [0.50], [0.20]]},
        "Sex": {"values": [[0.60], [0.40]]},
        "Education": {
            "evidence": ["Age", "Sex"],
            #          yM    yF    aM    aF    oM    oF
            "values": [[0.75, 0.64, 0.72, 0.70, 0.88, 0.90],   # high
                       [0.25, 0.36, 0.28, 0.30, 0.12, 0.10]],  # uni
        },
        "Occupation": {
            "evidence": ["Education"],
            "values": [[0.96, 0.92],    # emp
                       [0.04, 0.08]],   # self
        },
        "Residence": {
            "evidence": ["Education"],
            "values": [[0.25, 0.20],    # small
                       [0.75, 0.80]],   # big
        },
        "Travel": {
            "evidence": ["Occupation", "Residence"],
            #          e,s   e,b   s,s   s,b
            "values": [[0.48, 0.58, 0.56, 0.70],   # car
                       [0.42, 0.24, 0.36, 0.21],   # train
                       [0.10, 0.18, 0.08, 0.09]],  # other
        },
    },
)

# --------------------------------------------------------------------------
# Asia (Lauritzen y Spiegelhalter, 1988).  8 variables binarias, 8 arcos.
# El benchmark discreto clásico.  Es genuinamente difícil: `asia` sólo vale "sí"
# un 1 % de las veces, de modo que el arco asia -> tub es casi indetectable a
# tamaños de muestra razonables, y `either` es un OR lógico determinista.
# En el espacio de órdenes son 28 qubits; con la formulación de O'Gorman, 84.
# --------------------------------------------------------------------------
ASIA = BenchmarkNetwork(
    name="asia",
    arcs=[
        ("asia", "tub"),
        ("smoke", "lung"),
        ("smoke", "bronc"),
        ("tub", "either"),
        ("lung", "either"),
        ("either", "xray"),
        ("either", "dysp"),
        ("bronc", "dysp"),
    ],
    states={v: ["no", "yes"] for v in
            ("asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp")},
    cpds={
        "asia": {"values": [[0.99], [0.01]]},
        "smoke": {"values": [[0.50], [0.50]]},
        "tub": {
            "evidence": ["asia"],
            "values": [[0.99, 0.95], [0.01, 0.05]],
        },
        "lung": {
            "evidence": ["smoke"],
            "values": [[0.99, 0.90], [0.01, 0.10]],
        },
        "bronc": {
            "evidence": ["smoke"],
            "values": [[0.70, 0.40], [0.30, 0.60]],
        },
        "either": {   # OR lógico de lung y tub
            "evidence": ["lung", "tub"],
            #          l0t0  l0t1  l1t0  l1t1
            "values": [[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 1.0, 1.0]],
        },
        "xray": {
            "evidence": ["either"],
            "values": [[0.95, 0.02], [0.05, 0.98]],
        },
        "dysp": {
            "evidence": ["bronc", "either"],
            #          b0e0  b0e1  b1e0  b1e1
            "values": [[0.9, 0.3, 0.2, 0.1],
                       [0.1, 0.7, 0.8, 0.9]],
        },
    },
)

# --------------------------------------------------------------------------
# Capas10: red sintética de 10 nodos construida **a propósito** para ser difícil
# en el espacio de órdenes.
#
#   A  B  C        D           <- raíces
#    \/   \/        |
#    E     F        |          E<-{A,B}   F<-{B,C}
#     \   / \      /
#      \ /   \    /
#       G     H            G<-{E,F}   H<-{F,D}
#        \   /
#          I               I<-{G,H}
#          |
#          J               J<-{I}
#
# La dificultad de un problema en el espacio de órdenes la gobierna el número de
# **extensiones lineales** del DAG verdadero: cuantas menos, menor es la
# proporción de órdenes que alcanzan el óptimo. Esta estructura tiene sólo 256
# extensiones de 10! = 3 628 800 (un 0,007 %), frente a las 58 de 40 320 de Asia
# (0,14 %). Medido: muestrear 1024 órdenes válidos al azar la resuelve sólo el
# 7 % de las veces, mientras que resuelve Asia el 100 %.
#
# En la codificación factorial son 25 qubits; con la de torneo serían 45.
# --------------------------------------------------------------------------
_P_DOS = [[0.95, 0.70, 0.70, 0.05],   # P(hijo=0 | padres)
          [0.05, 0.30, 0.30, 0.95]]   # P(hijo=1 | padres)
_P_UNO = [[0.85, 0.15], [0.15, 0.85]]

CAPAS10 = BenchmarkNetwork(
    name="capas10",
    arcs=[
        ("A", "E"), ("B", "E"),
        ("B", "F"), ("C", "F"),
        ("E", "G"), ("F", "G"),
        ("F", "H"), ("D", "H"),
        ("G", "I"), ("H", "I"),
        ("I", "J"),
    ],
    states={v: ["0", "1"] for v in "ABCDEFGHIJ"},
    cpds={
        "A": {"values": [[0.5], [0.5]]},
        "B": {"values": [[0.5], [0.5]]},
        "C": {"values": [[0.5], [0.5]]},
        "D": {"values": [[0.5], [0.5]]},
        "E": {"evidence": ["A", "B"], "values": _P_DOS},
        "F": {"evidence": ["B", "C"], "values": _P_DOS},
        "G": {"evidence": ["E", "F"], "values": _P_DOS},
        "H": {"evidence": ["D", "F"], "values": _P_DOS},
        "I": {"evidence": ["G", "H"], "values": _P_DOS},
        "J": {"evidence": ["I"], "values": _P_UNO},
    },
)

NETWORKS = {net.name: net for net in (CANCER, TOY4, SURVEY, ASIA, CAPAS10)}
