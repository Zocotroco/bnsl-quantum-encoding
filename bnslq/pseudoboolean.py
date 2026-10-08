"""Polinomios pseudo-booleanos y su traducción a Hamiltonianos de Ising.

Toda función de variables binarias admite una representación única como
polinomio multilineal

    f(x) = sum_S  c_S  prod_{i in S} x_i ,      x_i in {0,1}

que es justamente la forma en la que aparece el Hamiltoniano de coste del BNSL.
Se mantiene el **grado arbitrario** a propósito: el término de puntuación de
O'Gorman tiene grado igual al in-degree máximo permitido, y en un ordenador de
puertas no hace falta cuadratizarlo (a diferencia de un *annealer*), porque
exp(-i·gamma·Z_a Z_b Z_c ...) se sintetiza con una escalera de CNOTs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, Sequence, Tuple

import numpy as np

Term = FrozenSet[int]


class PseudoBoolean:
    """Polinomio multilineal sobre variables binarias x_i ∈ {0,1}."""

    __slots__ = ("terms", "num_vars")

    def __init__(self, num_vars: int, terms: Dict[Term, float] | None = None):
        self.num_vars = num_vars
        self.terms: Dict[Term, float] = defaultdict(float)
        if terms:
            for k, v in terms.items():
                self.terms[frozenset(k)] += v

    # ------------------------------------------------------------ construcción

    def add(self, variables: Iterable[int], coefficient: float) -> "PseudoBoolean":
        """Suma `coefficient * prod(x_i for i in variables)`. Devuelve self."""
        if coefficient:
            self.terms[frozenset(variables)] += coefficient
        return self

    def prune(self, tol: float = 1e-12) -> "PseudoBoolean":
        """Elimina términos numéricamente nulos."""
        for k in [k for k, v in self.terms.items() if abs(v) <= tol]:
            del self.terms[k]
        return self

    @staticmethod
    def indicator(num_vars: int, variables: Sequence[int], predicate) -> "PseudoBoolean":
        """Polinomio multilineal exacto de un predicado booleano.

        Dado `predicate(valores)` sobre las variables indicadas, devuelve el
        único polinomio multilineal que vale 1 donde el predicado es cierto y 0
        donde es falso.  Se obtiene por inversión de Möbius sobre la tabla de
        verdad:

            c_S = sum_{T subconjunto de S} (-1)^{|S|-|T|} f(1_T)

        Es la misma herramienta que convierte las puntuaciones locales en el
        polinomio de coste, aplicada aquí a una restricción de validez.
        """
        k = len(variables)
        if k > 20:
            raise ValueError("demasiadas variables para enumerar la tabla de verdad")
        tabla = {}
        for mask in range(1 << k):
            vals = tuple((mask >> b) & 1 for b in range(k))
            tabla[vals] = float(bool(predicate(vals)))

        poly = PseudoBoolean(num_vars)
        for mask_s in range(1 << k):
            S = [b for b in range(k) if mask_s >> b & 1]
            coef = 0.0
            for mask_t in range(1 << k):
                if mask_t & ~mask_s:
                    continue  # T debe ser subconjunto de S
                T = tuple((mask_t >> b) & 1 for b in range(k))
                coef += (-1.0) ** (len(S) - bin(mask_t).count("1")) * tabla[T]
            if abs(coef) > 1e-12:
                poly.add([variables[b] for b in S], coef)
        return poly.prune()

    # -------------------------------------------------------------- evaluación

    def __call__(self, x: Sequence[int]) -> float:
        total = 0.0
        for term, coef in self.terms.items():
            if all(x[i] for i in term):
                total += coef
        return total

    def evaluate_many(self, X: np.ndarray) -> np.ndarray:
        """Evalúa sobre una matriz (n_muestras, num_vars) de 0/1."""
        X = np.asarray(X, dtype=np.int8)
        out = np.zeros(len(X), dtype=np.float64)
        for term, coef in self.terms.items():
            if not term:
                out += coef
            else:
                idx = list(term)
                out += coef * X[:, idx].prod(axis=1)
        return out

    # -------------------------------------------------------------- propiedades

    @property
    def degree(self) -> int:
        return max((len(t) for t in self.terms), default=0)

    @property
    def constant(self) -> float:
        return self.terms.get(frozenset(), 0.0)

    @property
    def num_terms(self) -> int:
        return sum(1 for t in self.terms if t)

    def is_qubo(self) -> bool:
        return self.degree <= 2

    # ------------------------------------------------------------------- Ising

    def to_ising(self) -> Tuple[Dict[Tuple[int, ...], float], float]:
        """Cambia a variables de espín mediante x_i = (1 - z_i) / 2.

        Devuelve ``(terminos, desplazamiento)`` donde `terminos` asocia a cada
        tupla ordenada de qubits el coeficiente del producto de Z's
        correspondiente, y `desplazamiento` es el término identidad.
        """
        out: Dict[Tuple[int, ...], float] = defaultdict(float)
        offset = 0.0
        for term, coef in self.terms.items():
            k = len(term)
            if k == 0:
                offset += coef
                continue
            scale = coef / (2.0**k)
            ordered = sorted(term)
            # prod (1 - z_i) = sum over subsets S de (-1)^|S| prod_{i in S} z_i
            for mask in range(1 << k):
                subset = tuple(ordered[b] for b in range(k) if mask >> b & 1)
                value = scale * (-1.0) ** len(subset)
                if subset:
                    out[subset] += value
                else:
                    offset += value
        return {k: v for k, v in out.items() if abs(v) > 1e-12}, offset

    # ----------------------------------------------------------- fuerza bruta

    def brute_force_minimum(self) -> Tuple[str, float]:
        """Mínimo global por enumeración. Sólo para num_vars pequeño (<= 24)."""
        if self.num_vars > 24:
            raise ValueError(
                f"{self.num_vars} variables son {2**self.num_vars} estados: "
                "demasiados para enumerar"
            )
        n = self.num_vars
        X = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
        values = self.evaluate_many(X)
        best = int(np.argmin(values))
        return "".join(str(b) for b in X[best]), float(values[best])

    def __repr__(self) -> str:
        return (
            f"PseudoBoolean(num_vars={self.num_vars}, grado={self.degree}, "
            f"terminos={self.num_terms})"
        )
