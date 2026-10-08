"""Conjuntos candidatos de padres: poda exacta del espacio de familias.

Idea (de Campos y Ji, 2010; usada por Shikuri, 2025). Un conjunto de padres $W$
es **candidato** para el nodo $X_i$ si todo subconjunto propio suyo puntúa
estrictamente peor:

    W es candidato  <=>  para todo W' subconjunto propio de W,  s_i(W') < s_i(W)

Si algún subconjunto propio puntúa igual o mejor, $W$ nunca hace falta: ese
subconjunto es al menos tan bueno, es más pequeño, y está disponible siempre que
lo esté $W$ (porque cualquier orden que permita $W$ permite también sus
subconjuntos).

La poda es **exacta**: no puede perder el óptimo. Si el mejor subconjunto de los
predecesores no fuera candidato, existiría un subconjunto propio suyo con
puntuación mayor o igual, que también está disponible; iterando se llega a un
candidato con la misma puntuación.

Para qué sirve aquí: el paso de K2 --- elegir los mejores padres dado un orden ---
es el cuello de botella de toda la búsqueda en el espacio de órdenes, porque se
ejecuta una vez por cada solución medida. Precalcular los candidatos ordenados
por puntuación convierte ese paso en recorrer una lista corta hasta el primer
conjunto contenido en los predecesores.
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

from .scores import LocalScore


class ParentSetCache:
    """Candidatos precalculados y ordenados por puntuación descendente.

    Parameters
    ----------
    score
        Puntuación local descomponible.
    max_parents
        In-degree máximo considerado.
    """

    def __init__(self, score: LocalScore, max_parents: int = 2):
        self.score = score
        self.variables: Tuple[str, ...] = tuple(score.variables)
        self.n = len(self.variables)
        self.max_parents = min(max_parents, self.n - 1)
        self.candidates: Dict[str, List[Tuple[FrozenSet[str], float]]] = {}
        self._n_total = 0
        self._build()

    # ------------------------------------------------------------ construcción

    def _build(self) -> None:
        for node in self.variables:
            otros = [v for v in self.variables if v != node]
            todos: Dict[FrozenSet[str], float] = {}
            for tam in range(self.max_parents + 1):
                for W in itertools.combinations(otros, tam):
                    todos[frozenset(W)] = self.score(node, W)
            self._n_total += len(todos)

            candidatos = []
            for W, valor in todos.items():
                if all(
                    todos[frozenset(Wp)] < valor
                    for tam in range(len(W))
                    for Wp in itertools.combinations(sorted(W), tam)
                ):
                    candidatos.append((W, valor))
            # el conjunto vacío no tiene subconjuntos propios: siempre candidato,
            # lo que garantiza que `best_for` termine para cualquier orden.
            candidatos.sort(key=lambda t: -t[1])
            self.candidates[node] = candidatos

    # ------------------------------------------------------------------ uso

    def best_for(
        self, node: str, allowed: Iterable[str]
    ) -> Tuple[Tuple[str, ...], float]:
        """Mejor conjunto de padres de `node` contenido en `allowed`.

        Recorre los candidatos de mayor a menor puntuación y devuelve el primero
        que cabe entre los predecesores.
        """
        permitidos = frozenset(allowed)
        for W, valor in self.candidates[node]:
            if W <= permitidos:
                return tuple(sorted(W)), valor
        raise RuntimeError(f"sin candidatos para {node!r}: no debería ocurrir")

    def score_of_order(self, order: Sequence[str]) -> float:
        """Sólo la puntuación de un orden, sin construir la lista de arcos.

        Camino rápido para barridos masivos: enumerar los 10! = 3,6 millones de
        órdenes de una red de diez nodos exige no pagar la construcción y
        ordenación de los arcos en cada uno.
        """
        total = 0.0
        permitidos: set = set()
        for node in order:
            for W, valor in self.candidates[node]:
                if W <= permitidos:
                    total += valor
                    break
            permitidos.add(node)
        return total

    def best_dag_for_order(
        self, order: Sequence[str]
    ) -> Tuple[List[Tuple[str, str]], float]:
        """Mejor DAG compatible con un orden, usando la poda."""
        arcs: List[Tuple[str, str]] = []
        total = 0.0
        for pos, node in enumerate(order):
            padres, valor = self.best_for(node, order[:pos])
            total += valor
            arcs.extend((p, node) for p in padres)
        return sorted(arcs), total

    # ------------------------------------------------------------- diagnóstico

    @property
    def n_candidates(self) -> int:
        return sum(len(v) for v in self.candidates.values())

    @property
    def n_total(self) -> int:
        return self._n_total

    def summary(self) -> str:
        return (
            f"conjuntos candidatos  n={self.n}  m={self.max_parents}\n"
            f"  familias evaluadas: {self.n_total}\n"
            f"  candidatos:         {self.n_candidates} "
            f"({100*self.n_candidates/self.n_total:.1f}% del total, "
            f"poda {self.n_total/max(1,self.n_candidates):.1f}x)\n"
            f"  por nodo: "
            + ", ".join(f"{v}={len(self.candidates[v])}" for v in self.variables)
        )
