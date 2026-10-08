"""Codificación factorial del espacio de órdenes (Romero, Larrañaga y Sierra, 2004).

La codificación por torneo de `orderings.py` gasta n(n-1)/2 qubits binarios de
los que sólo n! configuraciones son válidas: un 0,015 % para n=8. Romero et al.
resolvieron ese mismo problema en el contexto de los algoritmos de estimación de
distribución mediante una **biyección basada en la descomposición factorial de
n!**, y su representación no tiene estados inválidos en absoluto.

La idea
-------
Un orden de n elementos queda determinado por n-1 dígitos

    d_1 in {0..n-1},  d_2 in {0..n-2},  ...,  d_{n-1} in {0,1},

leídos como "de los elementos que aún quedan disponibles, toma el d_k-ésimo".
Es el sistema de numeración factorial, o código de Lehmer. La correspondencia
entre dígitos y órdenes es **biyectiva**: ni dos dígitos distintos dan el mismo
orden, ni hay combinaciones sin sentido.

Qué cambia al llevarlo a qubits
-------------------------------
Cada dígito se representa en binario con ceil(log2(r_k)) qubits. Como las
cardinalidades no son potencias de dos, reaparecen algunos estados inválidos
--- aquellos en los que un dígito excede su cardinalidad --- pero en proporción
incomparablemente menor:

    n     O'Gorman   torneo   factorial      válidos (torneo)   válidos (factorial)
    5        30        10          8             11,7 %              46,9 %
    6        45        15         11              2,2 %              35,2 %
    8        84        28         17              0,015 %            30,8 %
    10      135        45         25              3e-5 %             10,8 %

Y la penalización es además mucho más **local**: la validez de un dígito depende
sólo de sus propios qubits, mientras que H_trans en la codificación por torneo
acopla tríos de variables y tiene 189 términos para n=8.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .parentsets import ParentSetCache
from .pseudoboolean import PseudoBoolean
from .scores import LocalScore

Arc = Tuple[str, str]


class FactorialOrderingSpace:
    """Espacio de órdenes con codificación factorial (código de Lehmer).

    Expone la misma interfaz que `OrderingSpace`, de modo que `OrderingVQA`
    puede usarla sin cambios.
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

        # Dígito k (0-indexado) elige entre n-k elementos restantes.
        self.cardinalities: List[int] = [self.n - k for k in range(self.n - 1)]
        self.bits_per_digit: List[int] = [
            max(1, math.ceil(math.log2(c))) for c in self.cardinalities
        ]
        self.digit_qubits: List[List[int]] = []
        k = 0
        for b in self.bits_per_digit:
            self.digit_qubits.append(list(range(k, k + b)))
            k += b
        self.num_vars = k

        self.parent_sets = ParentSetCache(score, self.max_parents)
        self.delta_trans = float(
            delta_trans if delta_trans is not None else self.suggested_penalty()
        )
        self.h_trans = self._build_penalty()
        self.infeasible_floor = self.worst_feasible_cost()
        self._cache: Dict[str, float] = {}

    # ------------------------------------------------------------ penalización

    def suggested_penalty(self) -> float:
        total = 0.0
        for node in self.variables:
            otros = [v for v in self.variables if v != node]
            valores = [
                self.score(node, J)
                for size in range(self.max_parents + 1)
                for J in itertools.combinations(otros, size)
            ]
            total += max(valores) - min(valores)
        return total + 1.0

    def worst_feasible_cost(self) -> float:
        """Cota superior del coste de cualquier orden válido."""
        total = 0.0
        for node in self.variables:
            otros = [v for v in self.variables if v != node]
            total += min(
                self.score(node, J)
                for size in range(self.max_parents + 1)
                for J in itertools.combinations(otros, size)
            )
        return -total

    def _build_penalty(self) -> PseudoBoolean:
        """Penaliza los dígitos que exceden su cardinalidad.

        Es una suma de términos **independientes por dígito**: la validez de uno
        no depende de los demás.  Contrasta con H_trans de la codificación por
        torneo, que acopla tríos de variables.
        """
        poly = PseudoBoolean(self.num_vars)
        for card, qubits in zip(self.cardinalities, self.digit_qubits):
            if card == 2 ** len(qubits):
                continue  # cardinalidad potencia de dos: no hay estados inválidos
            ind = PseudoBoolean.indicator(
                self.num_vars,
                qubits,
                lambda v, c=card: sum(x << i for i, x in enumerate(v)) >= c,
            )
            for term, coef in ind.terms.items():
                poly.add(term, coef * self.delta_trans)
        return poly.prune()

    # -------------------------------------------------------- orden <-> bits

    def _digits(self, bits: Sequence[int] | str) -> Optional[List[int]]:
        """Valores de los dígitos, o None si alguno excede su cardinalidad."""
        x = [int(b) for b in bits]
        digitos = []
        for card, qubits in zip(self.cardinalities, self.digit_qubits):
            valor = sum(x[q] << i for i, q in enumerate(qubits))
            if valor >= card:
                return None
            digitos.append(valor)
        return digitos

    def violations(self, bits: Sequence[int] | str) -> int:
        """Número de dígitos fuera de rango."""
        x = [int(b) for b in bits]
        fuera = 0
        for card, qubits in zip(self.cardinalities, self.digit_qubits):
            if sum(x[q] << i for i, q in enumerate(qubits)) >= card:
                fuera += 1
        return fuera

    def is_total_order(self, bits: Sequence[int] | str) -> bool:
        return self._digits(bits) is not None

    def decode_order(self, bits: Sequence[int] | str) -> Optional[Tuple[str, ...]]:
        """Orden codificado, o None si la asignación no es válida."""
        digitos = self._digits(bits)
        if digitos is None:
            return None
        disponibles = list(self.variables)
        orden = [disponibles.pop(d) for d in digitos]
        orden.append(disponibles[0])
        return tuple(orden)

    def encode(self, order: Sequence[str]) -> List[int]:
        """Asignación binaria correspondiente a un orden dado."""
        if sorted(order) != sorted(self.variables):
            raise ValueError("el orden debe ser una permutación de las variables")
        disponibles = list(self.variables)
        x = [0] * self.num_vars
        for k, nodo in enumerate(order[:-1]):
            d = disponibles.index(nodo)
            disponibles.pop(d)
            for i, q in enumerate(self.digit_qubits[k]):
                x[q] = (d >> i) & 1
        return x

    # ------------------------------------------------------------------ coste

    def dag_of(self, bits: Sequence[int] | str) -> Tuple[List[Arc], float]:
        orden = self.decode_order(bits)
        if orden is None:
            raise ValueError("la asignación no codifica un orden válido")
        return self.parent_sets.best_dag_for_order(orden)

    def cost(self, bits: Sequence[int] | str) -> float:
        clave = "".join(str(int(b)) for b in bits)
        hit = self._cache.get(clave)
        if hit is not None:
            return hit
        orden = self.decode_order(clave)
        if orden is None:
            valor = self.infeasible_floor + self.delta_trans * self.violations(clave)
        else:
            _, puntuacion = self.parent_sets.best_dag_for_order(orden)
            valor = -puntuacion
        self._cache[clave] = valor
        return valor

    # ----------------------------------------------------------- fuerza bruta

    def enumerate_orders(self):
        for orden in itertools.permutations(self.variables):
            arcs, p = self.parent_sets.best_dag_for_order(orden)
            yield orden, arcs, p

    def brute_force_minimum(self) -> Tuple[str, float]:
        if self.num_vars > 24:
            raise ValueError(f"{self.num_vars} variables son demasiadas para enumerar")
        mejor_bits, mejor = None, float("inf")
        for entero in range(1 << self.num_vars):
            bits = "".join(str(entero >> b & 1) for b in range(self.num_vars))
            valor = self.cost(bits)
            if valor < mejor:
                mejor_bits, mejor = bits, valor
        return mejor_bits, mejor

    def summary(self) -> str:
        torneo = self.n * (self.n - 1) // 2
        validos = math.factorial(self.n)
        estados = 2**self.num_vars
        return (
            f"espacio de órdenes factorial  n={self.n}  m={self.max_parents}\n"
            f"  qubits: {self.num_vars}  (torneo: {torneo}, "
            f"O'Gorman: {3*torneo}; reducción del {100*(1-self.num_vars/torneo):.0f}% "
            f"sobre el torneo)\n"
            f"  dígitos: cardinalidades {self.cardinalities} en "
            f"{self.bits_per_digit} qubits\n"
            f"  espacio: 2^{self.num_vars} = {estados}, de los que {validos} "
            f"son órdenes válidos ({100*validos/estados:.1f}%, "
            f"frente al {100*validos/2**torneo:.4f}% del torneo)\n"
            f"  penalización: {self.h_trans.num_terms} términos, grado "
            f"{self.h_trans.degree}"
        )
