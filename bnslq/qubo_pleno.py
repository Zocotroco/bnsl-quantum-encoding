"""QUBO pleno en el espacio de órdenes: la puntuación dentro del circuito.

Las dos variantes estudiadas hasta aquí dejan la puntuación **fuera** del
circuito: el operador de coste sólo lleva la restricción de validez, y el BIC
entra por el bucle clásico. Es inevitable mientras el estado codifique sólo el
orden, porque la puntuación de un orden contiene un máximo sobre subconjuntos de
padres, que no es un polinomio.

Esta tercera variante elimina ese máximo al estilo de \\citet{shikuri2025}:
además del orden, se codifica **qué conjunto de padres se elige para cada
nodo**. Con esa información la puntuación pasa a ser lineal, y el Hamiltoniano
completo --- puntuación incluida --- es cuadrático y compilable:

    H = H_score + H_unico + H_trans + H_consist

    H_score   = sum_i sum_J  s_i(J) y_{i,J}                      (lineal)
    H_unico   = delta sum_i (1 - sum_J y_{i,J})^2                (un conjunto por nodo)
    H_trans   = transitividad del orden                          (acíclico)
    H_consist = delta sum_i sum_J sum_{j in J} y_{i,J}(1 - [j<i]) (padres antes que el hijo)

El precio es el número de qubits: a los n(n-1)/2 del orden se suman tantos como
conjuntos candidatos de padres haya. La poda por conjuntos candidatos es por
tanto imprescindible, no un lujo.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from .parentsets import ParentSetCache
from .pseudoboolean import PseudoBoolean
from .scores import LocalScore

Arc = Tuple[str, str]


class OrdenQUBOPleno:
    """Orden más selección de padres, con la puntuación dentro del polinomio."""

    def __init__(self, score: LocalScore, max_parents: int = 2,
                 delta: Optional[float] = None):
        self.score = score
        self.variables: Tuple[str, ...] = tuple(score.variables)
        self.n = len(self.variables)
        self.max_parents = min(max_parents, self.n - 1)
        self.parent_sets = ParentSetCache(score, self.max_parents)

        # --- índices: primero el orden, después la selección de padres ---
        self.r_index: Dict[Tuple[int, int], int] = {}
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.r_index[(i, j)] = k
                k += 1
        self.n_r = k

        self.y_index: Dict[Tuple[int, int], int] = {}
        self.candidatos: List[List[Tuple[frozenset, float]]] = []
        for i, nodo in enumerate(self.variables):
            cands = self.parent_sets.candidates[nodo]
            self.candidatos.append(cands)
            for j in range(len(cands)):
                self.y_index[(i, j)] = k
                k += 1
        self.num_vars = k

        self.delta = float(delta if delta is not None else self.penalizacion())
        # desplazamiento que devuelve la energía a la escala de -puntuación
        self.offset = -sum(min(v for _, v in c) for c in self.candidatos)
        self.poly = self._construir()
        # el operador de coste del circuito es el Hamiltoniano COMPLETO
        self.h_trans = self.poly
        self.delta_trans = self.delta
        self._cache: Dict[str, float] = {}

    # ------------------------------------------------------------ construcción

    def penalizacion(self) -> float:
        """Cota: mayor que cualquier ganancia posible de puntuación."""
        total = 0.0
        for nodo in self.variables:
            vals = [v for _, v in self.parent_sets.candidates[nodo]]
            total += max(vals) - min(vals)
        return total + 1.0

    def _precede(self, poly: PseudoBoolean, j: int, i: int, coef: float) -> None:
        """Añade coef * [j NO precede a i] al polinomio."""
        if j < i:
            # j precede a i  <=>  r_ji = 1 ; no precede = 1 - r_ji
            q = self.r_index[(j, i)]
            poly.add([], coef)
            poly.add([q], -coef)
        else:
            # j precede a i  <=>  r_ij = 0 ; no precede = r_ij
            poly.add([self.r_index[(i, j)]], coef)

    def _construir(self) -> PseudoBoolean:
        poly = PseudoBoolean(self.num_vars)
        d = self.delta
        idx = {v: i for i, v in enumerate(self.variables)}

        # H_score: lineal en las variables de selección.
        #
        # OJO al signo y al desplazamiento. Si se usara directamente -score, que
        # es un número positivo grande (~840 por nodo), al optimizador le saldría
        # a cuenta **no elegir ningún conjunto**: pagaría delta por violar la
        # restricción de unicidad y se ahorraría los 840. Por eso cada nodo se
        # desplaza por su peor candidato, de modo que elegir siempre sea mejor
        # que no elegir, y el desplazamiento se devuelve en `offset`.
        for i, cands in enumerate(self.candidatos):
            peor = min(v for _, v in cands)
            for j, (W, valor) in enumerate(cands):
                poly.add([self.y_index[(i, j)]], peor - valor)   # <= 0

        # H_unico: exactamente un conjunto por nodo
        for i, cands in enumerate(self.candidatos):
            ys = [self.y_index[(i, j)] for j in range(len(cands))]
            poly.add([], d)                       # (1 - sum y)^2 = 1 - 2 sum y + (sum y)^2
            for a in ys:
                poly.add([a], -2.0 * d)
                poly.add([a], d)                  # y^2 = y
            for a, b in itertools.combinations(ys, 2):
                poly.add([a, b], 2.0 * d)

        # H_trans: transitividad del orden
        for i, j, k in itertools.combinations(range(self.n), 3):
            rij, rjk, rik = (self.r_index[(i, j)], self.r_index[(j, k)],
                             self.r_index[(i, k)])
            poly.add([rik], d)
            poly.add([rij, rjk], d)
            poly.add([rij, rik], -d)
            poly.add([rjk, rik], -d)

        # H_consist: los padres elegidos han de preceder al hijo
        for i, cands in enumerate(self.candidatos):
            for j, (W, _) in enumerate(cands):
                y = self.y_index[(i, j)]
                for padre in W:
                    p = idx[padre]
                    aux = PseudoBoolean(self.num_vars)
                    self._precede(aux, p, i, d)
                    for term, coef in aux.terms.items():
                        poly.add(set(term) | {y}, coef)
        return poly.prune()

    # -------------------------------------------------------------- decodificar

    def _seleccion(self, bits: Sequence[int] | str) -> Optional[List[int]]:
        """Índice del conjunto elegido por cada nodo, o None si no hay exactamente uno."""
        x = [int(b) for b in bits]
        elegidos = []
        for i, cands in enumerate(self.candidatos):
            activos = [j for j in range(len(cands)) if x[self.y_index[(i, j)]]]
            if len(activos) != 1:
                return None
            elegidos.append(activos[0])
        return elegidos

    def _orden_valido(self, bits: Sequence[int] | str) -> Optional[Tuple[str, ...]]:
        x = [int(b) for b in bits]
        cuenta = [0] * self.n
        for (i, j), q in self.r_index.items():
            if x[q]:
                cuenta[i] += 1
            else:
                cuenta[j] += 1
        if sorted(cuenta) != list(range(self.n)):
            return None
        pos = sorted(range(self.n), key=lambda i: -cuenta[i])
        return tuple(self.variables[i] for i in pos)

    def is_total_order(self, bits: Sequence[int] | str) -> bool:
        """Asignación completamente válida: orden total, un conjunto por nodo y coherencia."""
        orden = self._orden_valido(bits)
        if orden is None:
            return False
        elegidos = self._seleccion(bits)
        if elegidos is None:
            return False
        pos = {v: p for p, v in enumerate(orden)}
        for i, j in enumerate(elegidos):
            hijo = self.variables[i]
            for padre in self.candidatos[i][j][0]:
                if pos[padre] >= pos[hijo]:
                    return False
        return True

    def decode_order(self, bits: Sequence[int] | str) -> Optional[Tuple[str, ...]]:
        return self._orden_valido(bits) if self.is_total_order(bits) else None

    def dag_of(self, bits: Sequence[int] | str) -> Tuple[List[Arc], float]:
        elegidos = self._seleccion(bits)
        if elegidos is None or not self.is_total_order(bits):
            raise ValueError("la asignación no es válida")
        arcs, total = [], 0.0
        for i, j in enumerate(elegidos):
            W, valor = self.candidatos[i][j]
            total += valor
            arcs.extend((p, self.variables[i]) for p in W)
        return sorted(arcs), total

    def encode(self, order: Sequence[str]) -> List[int]:
        """Asignación del mejor DAG compatible con un orden dado."""
        x = [0] * self.num_vars
        pos = {v: p for p, v in enumerate(order)}
        for (i, j), q in self.r_index.items():
            x[q] = int(pos[self.variables[i]] < pos[self.variables[j]])
        for i, nodo in enumerate(self.variables):
            permitidos = frozenset(order[:pos[nodo]])
            for j, (W, _) in enumerate(self.candidatos[i]):
                if W <= permitidos:
                    x[self.y_index[(i, j)]] = 1
                    break
        return x

    # -------------------------------------------------------------------- coste

    def cost(self, bits: Sequence[int] | str) -> float:
        """Valor del polinomio. A diferencia de las otras variantes, aquí el
        coste **sí** es el Hamiltoniano completo: la puntuación está dentro."""
        clave = "".join(str(int(b)) for b in bits)
        hit = self._cache.get(clave)
        if hit is None:
            hit = self.poly([int(b) for b in clave]) + self.offset
            self._cache[clave] = hit
        return hit

    def brute_force_minimum(self) -> Tuple[str, float]:
        import numpy as np
        if self.num_vars > 24:
            raise ValueError(f"{self.num_vars} variables son demasiadas para enumerar")
        X = ((np.arange(1 << self.num_vars)[:, None] >> np.arange(self.num_vars)) & 1)
        vals = self.poly.evaluate_many(X.astype(np.int8)) + self.offset
        i = int(np.argmin(vals))
        return "".join(str(int(b)) for b in X[i]), float(vals[i])

    def summary(self) -> str:
        return (
            f"QUBO pleno  n={self.n}  m={self.max_parents}\n"
            f"  qubits: {self.n_r} orden + {self.num_vars - self.n_r} selección "
            f"= {self.num_vars}\n"
            f"  polinomio: grado {self.poly.degree}, {self.poly.num_terms} términos "
            f"(la puntuación va DENTRO)\n"
            f"  delta={self.delta:.3g}"
        )
