"""Representación de una red bayesiana discreta como circuito cuántico.

Es la metodología de Borujeni et al. (2021), que el trabajo previo del autor
estableció para el caso de dos nodos, generalizada aquí a un DAG arbitrario de
variables binarias.

La idea
-------
Una variable de Bernoulli con P(X=1)=p se prepara con una rotación
    Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>,   theta = 2 arcsin(sqrt(p)),
de modo que la amplitud al cuadrado del estado |1> vale exactamente p.

Para un nodo con k padres hay 2^k probabilidades condicionales, y cada una se
codifica con una rotación Ry **controlada por los k qubits de los padres**, con
los controles negados donde corresponda a esa configuración. Recorriendo los
nodos en orden topológico, el estado final cumple

    |psi> = sum_x sqrt(P(X = x)) |x>,

es decir, las amplitudes al cuadrado reproducen la distribución conjunta
completa. Medir el circuito equivale a muestrear de la red.

Por qué importa aquí
--------------------
Cierra el ciclo del trabajo: el algoritmo variacional aprende la estructura, los
parámetros se ajustan de los datos, y este módulo construye el circuito sobre el
que después se hace inferencia (ver `inferencia.py`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

Arc = Tuple[str, str]


def angulo(p: float) -> float:
    """theta tal que Ry(theta)|0> tiene amplitud sqrt(p) en |1>."""
    p = min(max(float(p), 0.0), 1.0)
    return 2.0 * math.asin(math.sqrt(p))


class RedCuantica:
    """Circuito cuántico que codifica la distribución conjunta de una red binaria.

    Parameters
    ----------
    arcs
        Lista de arcos (padre, hijo) del DAG.
    cpts
        Para cada nodo, un diccionario que asocia la configuración de sus padres
        --- una tupla de 0/1 en el orden de `parents[nodo]` --- con P(nodo = 1).
        Los nodos raíz usan la clave `()`.
    variables
        Orden de las variables; determina qué qubit representa a cada una.
    """

    def __init__(self, arcs: Sequence[Arc], cpts: Dict[str, Dict[Tuple[int, ...], float]],
                 variables: Sequence[str]):
        self.variables: Tuple[str, ...] = tuple(variables)
        self.indice = {v: i for i, v in enumerate(self.variables)}
        self.n = len(self.variables)
        self.arcs = list(arcs)
        self.cpts = cpts

        g = nx.DiGraph()
        g.add_nodes_from(self.variables)
        g.add_edges_from(self.arcs)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("la estructura no es un DAG")
        self.grafo = g
        self.orden = list(nx.topological_sort(g))
        self.padres = {v: sorted(g.predecessors(v), key=self.indice.get)
                       for v in self.variables}

    # ------------------------------------------------------------- circuito

    def circuito(self, medir: bool = True) -> QuantumCircuit:
        """Circuito que prepara sum_x sqrt(P(x)) |x>."""
        qc = QuantumCircuit(self.n, self.n if medir else 0)
        for nodo in self.orden:
            objetivo = self.indice[nodo]
            padres = self.padres[nodo]
            if not padres:
                qc.ry(angulo(self.cpts[nodo][()]), objetivo)
                continue
            controles = [self.indice[p] for p in padres]
            for config in range(1 << len(padres)):
                clave = tuple((config >> i) & 1 for i in range(len(padres)))
                theta = angulo(self.cpts[nodo][clave])
                if abs(theta) < 1e-12:
                    continue
                # controles negados donde la configuración pide un 0
                negados = [c for c, b in zip(controles, clave) if b == 0]
                for c in negados:
                    qc.x(c)
                qc.append(RYGate(theta).control(len(controles)), controles + [objetivo])
                for c in negados:
                    qc.x(c)
        if medir:
            qc.measure(range(self.n), range(self.n))
        return qc

    # ----------------------------------------------------------- referencia

    def distribucion_exacta(self) -> Dict[Tuple[int, ...], float]:
        """Distribución conjunta calculada clásicamente, para validar."""
        dist = {}
        for entero in range(1 << self.n):
            x = tuple((entero >> i) & 1 for i in range(self.n))
            p = 1.0
            for nodo in self.variables:
                i = self.indice[nodo]
                clave = tuple(x[self.indice[q]] for q in self.padres[nodo])
                p_uno = self.cpts[nodo][clave]
                p *= p_uno if x[i] else (1.0 - p_uno)
            dist[x] = p
        return dist

    def circuito_ejecutable(self, backend=None) -> QuantumCircuit:
        """Circuito transpilado a puertas elementales.

        Las rotaciones multicontroladas `C^k-Ry` no son instrucciones nativas de
        ningún simulador ni procesador: hay que descomponerlas antes de
        ejecutar, o Aer falla con `unknown instruction: ccry`.
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        return transpile(self.circuito(), backend or AerSimulator(),
                         optimization_level=1)

    def estadisticas(self) -> Dict[str, int]:
        qc = self.circuito(medir=False).decompose(reps=4)
        ops = qc.count_ops()
        return {
            "qubits": self.n,
            "profundidad": qc.depth(),
            "puertas_2q": int(sum(ops.get(g, 0) for g in ("cx", "cz", "ecr"))),
        }


def ajustar_cpts(datos, arcs: Sequence[Arc], variables: Sequence[str],
                 alpha: float = 1.0) -> Dict[str, Dict[Tuple[int, ...], float]]:
    """Estima las CPTs de una estructura a partir de datos, con suavizado.

    Se usa un prior de Laplace (alpha = 1 por defecto) para que ninguna
    probabilidad sea exactamente 0 o 1: con probabilidades extremas el circuito
    es correcto pero la inferencia condicional se vuelve degenerada.
    """
    import pandas as pd

    d = pd.DataFrame({v: np.asarray(datos[v]).astype(int) for v in variables})
    indice = {v: i for i, v in enumerate(variables)}
    g = nx.DiGraph()
    g.add_nodes_from(variables)
    g.add_edges_from(arcs)
    cpts: Dict[str, Dict[Tuple[int, ...], float]] = {}
    for nodo in variables:
        padres = sorted(g.predecessors(nodo), key=indice.get)
        tabla: Dict[Tuple[int, ...], float] = {}
        for config in range(1 << len(padres)):
            clave = tuple((config >> i) & 1 for i in range(len(padres)))
            mascara = np.ones(len(d), dtype=bool)
            for p, b in zip(padres, clave):
                mascara &= (d[p].values == b)
            total = int(mascara.sum())
            unos = int((d[nodo].values[mascara] == 1).sum())
            tabla[clave] = (unos + alpha) / (total + 2 * alpha)
        cpts[nodo] = tabla
    return cpts
