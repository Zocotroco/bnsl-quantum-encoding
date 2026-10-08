"""BNSL en el espacio de órdenes: la formulación de la Fase 2.

Motivación
----------
La formulación de O'Gorman codifica la **matriz de adyacencia** y necesita
variables auxiliares de orden topológico para poder expresar la aciclicidad,
lo que da 3n(n-1)/2 qubits.  Pero si lo que se codifica es directamente el
orden, las aristas dejan de ser variables libres: fijado un orden, el mejor
conjunto de padres de cada nodo se calcula **exactamente** y en tiempo
polinómico (el paso de K2).  Quedan n(n-1)/2 qubits.

    n        O'Gorman/Soloviev        espacio de órdenes
    4               18                        6
    5               30                       10
    7               63                       21
    10             135                       45

La ventaja no es sólo de tamaño.  Al desaparecer la matriz de adyacencia
desaparece también H_consist y, con él, el hiperparámetro delta_consist que
Zardini et al. (2022) tuvieron que acotar empíricamente.  Queda una única
penalización, la de transitividad.

Es la línea de Romero, Larrañaga y Sierra (2004), que buscaban el orden con
algoritmos de estimación de distribución en vez de buscar el grafo.  Aquí el
motor de búsqueda es cuántico.

Codificación
------------
Una variable binaria r_ij por cada par i<j, con r_ij = 1 si X_i precede a X_j.
Esto define un torneo completo (para cada par hay exactamente una dirección).
Un torneo es un orden total **si y sólo si** es transitivo, es decir, si no
contiene ningún 3-ciclo; de ahí que baste con penalizar los tríos que lo
violan, con el mismo polinomio H_trans de O'Gorman.

De los 2^(n(n-1)/2) estados sólo n! son órdenes válidos: para n=5 son 120 de
1024, un 11.7%.  La penalización tiene que llevar la optimización a ese
subconjunto.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .classical import best_dag_for_order
from .parentsets import ParentSetCache
from .pseudoboolean import PseudoBoolean
from .scores import LocalScore

Arc = Tuple[str, str]


class OrderingSpace:
    """Espacio de búsqueda de órdenes totales sobre las variables de `score`.

    Parameters
    ----------
    score
        Puntuación local descomponible (a maximizar).
    max_parents
        In-degree máximo al reconstruir el DAG a partir del orden.
    delta_trans
        Penalización por cada trío que viola la transitividad.  Si se omite se
        usa una cota suficiente (ver `suggested_penalty`).
    """

    def __init__(
        self,
        score: LocalScore,
        max_parents: int = 2,
        delta_trans: Optional[float] = None,
    ):
        self.score = score
        self.variables: Tuple[str, ...] = tuple(score.variables)
        self.n = len(self.variables)
        if self.n < 2:
            raise ValueError("hacen falta al menos dos variables")
        self.max_parents = min(max_parents, self.n - 1)

        self.r_index: Dict[Tuple[int, int], int] = {}
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.r_index[(i, j)] = k
                k += 1
        self.num_vars = k  # n(n-1)/2

        self.delta_trans = float(
            delta_trans if delta_trans is not None else self.suggested_penalty()
        )
        self.parent_sets = ParentSetCache(score, self.max_parents)
        self.h_trans = self._build_htrans()
        self.infeasible_floor = self.worst_feasible_cost()
        self._cache: Dict[str, float] = {}

    # ------------------------------------------------------------ penalización

    def suggested_penalty(self) -> float:
        """Cota suficiente: mayor que el recorrido total de la puntuación.

        Si ninguna mejora de puntuación puede compensar una violación de
        transitividad, el mínimo global cae necesariamente sobre un orden
        válido.  Es conservadora; los experimentos pueden bajarla para no
        deformar el paisaje más de lo necesario.
        """
        total = 0.0
        for i, node in enumerate(self.variables):
            otros = [v for v in self.variables if v != node]
            valores = [
                self.score(node, J)
                for size in range(self.max_parents + 1)
                for J in itertools.combinations(otros, size)
            ]
            total += max(valores) - min(valores)
        return total + 1.0

    def worst_feasible_cost(self) -> float:
        """Cota superior del coste de cualquier orden válido.

        Como score(orden) = sum_i score_i(mejores padres) >= sum_i min_J score_i(J),
        el coste -score(orden) nunca supera -sum_i min_J score_i(J).  Sirve de
        suelo para las asignaciones infactibles: sumándoles la penalización
        quedan por encima de **todo** orden válido, que es lo que hace falta
        para que el mínimo global caiga sobre un orden.
        """
        total = 0.0
        for node in self.variables:
            otros = [v for v in self.variables if v != node]
            total += min(
                self.score(node, J)
                for size in range(self.max_parents + 1)
                for J in itertools.combinations(otros, size)
            )
        return -total

    def _build_htrans(self) -> PseudoBoolean:
        """H_trans = delta * sum_{i<j<k} (r_ik + r_ij r_jk - r_ij r_ik - r_jk r_ik).

        Vale 0 si el torneo es transitivo y delta por cada trío cíclico.
        """
        poly = PseudoBoolean(self.num_vars)
        d = self.delta_trans
        for i, j, k in itertools.combinations(range(self.n), 3):
            rij, rjk, rik = (
                self.r_index[(i, j)],
                self.r_index[(j, k)],
                self.r_index[(i, k)],
            )
            poly.add([rik], d)
            poly.add([rij, rjk], d)
            poly.add([rij, rik], -d)
            poly.add([rjk, rik], -d)
        return poly.prune()

    # -------------------------------------------------------- orden <-> bits

    def precedes(self, bits: Sequence[int] | str, i: int, j: int) -> bool:
        """¿Precede X_i a X_j según la asignación? (el torneo es completo)"""
        x = [int(b) for b in bits]
        if i < j:
            return bool(x[self.r_index[(i, j)]])
        return not x[self.r_index[(j, i)]]

    def violations(self, bits: Sequence[int] | str) -> int:
        """Número de tríos que violan la transitividad (3-ciclos)."""
        x = [int(b) for b in bits]
        return int(round(self.h_trans(x) / self.delta_trans)) if self.delta_trans else 0

    def is_total_order(self, bits: Sequence[int] | str) -> bool:
        """Un torneo es un orden total si y sólo si no tiene 3-ciclos.

        Se comprueba con los grados de salida: en un torneo transitivo de n
        nodos, los "precede a" valen exactamente n-1, n-2, ..., 1, 0.
        """
        cuenta = self._precedence_counts(bits)
        return sorted(cuenta) == list(range(self.n))

    def _precedence_counts(self, bits: Sequence[int] | str) -> List[int]:
        x = [int(b) for b in bits]
        cuenta = [0] * self.n
        for (i, j), k in self.r_index.items():
            if x[k]:
                cuenta[i] += 1
            else:
                cuenta[j] += 1
        return cuenta

    def decode_order(self, bits: Sequence[int] | str) -> Optional[Tuple[str, ...]]:
        """Orden total codificado, o None si la asignación no es transitiva."""
        cuenta = self._precedence_counts(bits)
        if sorted(cuenta) != list(range(self.n)):
            return None
        # quien precede a más nodos va primero
        posicion = sorted(range(self.n), key=lambda i: -cuenta[i])
        return tuple(self.variables[i] for i in posicion)

    def encode(self, order: Sequence[str]) -> List[int]:
        """Asignación binaria correspondiente a un orden dado."""
        if sorted(order) != sorted(self.variables):
            raise ValueError("el orden debe ser una permutación de las variables")
        posicion = {v: p for p, v in enumerate(order)}
        x = [0] * self.num_vars
        for (i, j), k in self.r_index.items():
            x[k] = int(posicion[self.variables[i]] < posicion[self.variables[j]])
        return x

    # ------------------------------------------------------------------ coste

    def dag_of(self, bits: Sequence[int] | str) -> Tuple[List[Arc], float]:
        """DAG óptimo y su puntuación para el orden codificado."""
        order = self.decode_order(bits)
        if order is None:
            raise ValueError("la asignación no codifica un orden total")
        return self.parent_sets.best_dag_for_order(order)

    def cost(self, bits: Sequence[int] | str) -> float:
        """Coste a minimizar: -puntuación del mejor DAG, o penalización.

        A diferencia de la formulación de O'Gorman, aquí el coste **no** es un
        polinomio en las variables de qubit: contiene un máximo sobre
        subconjuntos de padres.  Se evalúa clásicamente sobre cada solución
        medida, que es lo que hace de esto un VQA tipo muestreador en vez de un
        QAOA canónico.  El precio es que el operador de coste del circuito sólo
        puede llevar H_trans; la ventaja, que el registro es tres veces menor.
        """
        clave = "".join(str(int(b)) for b in bits)
        hit = self._cache.get(clave)
        if hit is not None:
            return hit
        order = self.decode_order(clave)
        if order is None:
            # Anclar sobre el peor coste factible: si se devolviera sólo la
            # penalización, los estados infactibles saldrían más baratos que
            # cualquier orden válido, porque el coste de un orden es menos una
            # log-verosimilitud y vale varios miles.
            valor = self.infeasible_floor + self.delta_trans * self.violations(clave)
        else:
            _, puntuacion = self.parent_sets.best_dag_for_order(order)
            valor = -puntuacion
        self._cache[clave] = valor
        return valor

    # ----------------------------------------------------------- fuerza bruta

    def enumerate_orders(self):
        """Recorre los n! órdenes válidos con su puntuación."""
        for order in itertools.permutations(self.variables):
            arcs, puntuacion = self.parent_sets.best_dag_for_order(order)
            yield order, arcs, puntuacion

    def brute_force_minimum(self) -> Tuple[str, float]:
        """Mínimo global sobre las 2^(n(n-1)/2) asignaciones."""
        if self.num_vars > 24:
            raise ValueError(f"{self.num_vars} variables son demasiadas para enumerar")
        mejor_bits, mejor = None, float("inf")
        for entero in range(1 << self.num_vars):
            bits = "".join(
                str(entero >> b & 1) for b in range(self.num_vars)
            )
            valor = self.cost(bits)
            if valor < mejor:
                mejor_bits, mejor = bits, valor
        return mejor_bits, mejor

    def summary(self) -> str:
        ogorman = 3 * self.n * (self.n - 1) // 2
        return (
            f"espacio de órdenes  n={self.n}  m={self.max_parents}\n"
            f"  qubits: {self.num_vars}  (O'Gorman/Soloviev: {ogorman}, "
            f"reducción {100*(1-self.num_vars/ogorman):.0f}%)\n"
            f"  espacio: 2^{self.num_vars} = {2**self.num_vars} asignaciones, "
            f"de las que {math.factorial(self.n)} son órdenes válidos "
            f"({100*math.factorial(self.n)/2**self.num_vars:.1f}%)\n"
            f"  H_trans: {self.h_trans.num_terms} términos, grado "
            f"{self.h_trans.degree}, delta={self.delta_trans:.3g}"
        )
