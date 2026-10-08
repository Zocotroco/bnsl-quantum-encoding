"""Referencias clásicas para el BNSL: óptimo exacto y búsquedas heurísticas.

Todas usan el **mismo** objeto `LocalScore` que la formulación cuántica, de modo
que las comparaciones no mezclan convenios distintos de BIC/BDeu.  Esto es
deliberado: comparar contra la implementación de otra librería introduce
diferencias de puntuación que se confunden con diferencias de algoritmo.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .scores import LocalScore

Arc = Tuple[str, str]


def _subsets(pool: Sequence[str], max_size: int):
    for r in range(min(max_size, len(pool)) + 1):
        yield from itertools.combinations(pool, r)


def best_dag_for_order(
    score: LocalScore, order: Sequence[str], max_parents: int = 2
) -> Tuple[List[Arc], float]:
    """Mejor DAG compatible con un orden topológico dado.

    Fijado el orden, la puntuación se maximiza eligiendo para cada nodo, de
    forma independiente, el mejor subconjunto de padres entre sus predecesores:
    la descomponibilidad garantiza que no hay interacción entre nodos.  Es el
    paso de K2 (Cooper y Herskovits, 1992) y se resuelve **exactamente** en
    O(2^(n-1) · n), o en O(n^m) con in-degree acotado.

    Esta es la pieza sobre la que se apoya la búsqueda en el espacio de órdenes:
    reduce el problema de elegir un grafo al de elegir una permutación.
    """
    arcs: List[Arc] = []
    total = 0.0
    for pos, node in enumerate(order):
        local_best, local_pa = float("-inf"), ()
        for pa in _subsets(order[:pos], max_parents):
            s = score(node, pa)
            if s > local_best:
                local_best, local_pa = s, pa
        total += local_best
        arcs.extend((p, node) for p in local_pa)
    return sorted(arcs), total


def exact_best_dag(
    score: LocalScore, max_parents: int = 2, cache=None
) -> Tuple[List[Arc], float]:
    """Óptimo global exacto por enumeración de órdenes topológicos.

    Todo DAG admite un orden topológico, de modo que recorrer las n!
    permutaciones y resolver cada una cubre el espacio completo de DAGs.

    Usa la poda por conjuntos candidatos de padres (`ParentSetCache`), que es
    exacta y evita reevaluar familias dominadas en cada permutación.  Practicable
    hasta n ≈ 9.  Es nuestra referencia de verdad-fundamental para validar que
    los algoritmos variacionales encuentran realmente el óptimo.
    """
    from .parentsets import ParentSetCache

    if cache is None:
        cache = ParentSetCache(score, max_parents)
    best_score = float("-inf")
    best_arcs: List[Arc] = []
    for order in itertools.permutations(score.variables):
        arcs, total = cache.best_dag_for_order(order)
        if total > best_score:
            best_score, best_arcs = total, arcs
    return sorted(best_arcs), best_score


# --------------------------------------------------------------------------
# Búsquedas heurísticas en el espacio de DAGs
# --------------------------------------------------------------------------

def _parents_of(arcs: Sequence[Arc], variables: Sequence[str]) -> Dict[str, List[str]]:
    pa: Dict[str, List[str]] = {v: [] for v in variables}
    for src, dst in arcs:
        pa[dst].append(src)
    return pa


def _creates_cycle(arcs: Sequence[Arc], new: Arc) -> bool:
    """¿Añadir `new` crea un ciclo? (búsqueda en profundidad del hijo al padre)"""
    src, dst = new
    succ: Dict[str, List[str]] = {}
    for a, b in arcs:
        succ.setdefault(a, []).append(b)
    stack, seen = [dst], set()
    while stack:
        cur = stack.pop()
        if cur == src:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(succ.get(cur, ()))
    return False


def _neighbours(arcs: Sequence[Arc], variables: Sequence[str], max_parents: int):
    """Genera (operación, arco, nuevo_conjunto_de_arcos) para añadir/quitar/invertir."""
    current = set(arcs)
    pa = _parents_of(arcs, variables)
    for u, v in itertools.permutations(variables, 2):
        if (u, v) in current:
            yield "quitar", (u, v), sorted(current - {(u, v)})
            rest = current - {(u, v)}
            if len(pa[u]) < max_parents and not _creates_cycle(sorted(rest), (v, u)):
                yield "invertir", (u, v), sorted(rest | {(v, u)})
        elif (v, u) not in current:
            if len(pa[v]) < max_parents and not _creates_cycle(arcs, (u, v)):
                yield "añadir", (u, v), sorted(current | {(u, v)})


def hill_climbing(
    score: LocalScore, max_parents: int = 2, start: Optional[Sequence[Arc]] = None
) -> Tuple[List[Arc], float]:
    """Ascenso de colinas voraz sobre el espacio de DAGs (Gámez et al., 2011)."""
    variables = list(score.variables)
    arcs = sorted(start or [])
    current = score.graph_score(arcs)
    while True:
        best_gain, best_arcs = 0.0, None
        for _op, _arc, candidate in _neighbours(arcs, variables, max_parents):
            gain = score.graph_score(candidate) - current
            if gain > best_gain + 1e-12:
                best_gain, best_arcs = gain, candidate
        if best_arcs is None:
            return arcs, current
        arcs, current = best_arcs, current + best_gain


def tabu_search(
    score: LocalScore,
    max_parents: int = 2,
    tabu_length: Optional[int] = None,
    max_iter: int = 200,
    seed: int = 0,
) -> Tuple[List[Arc], float]:
    """Búsqueda tabú: acepta el mejor vecino no tabú aunque empeore.

    `tabu_length` es sensible y conviene no fijarlo a ciegas.  Medido sobre la
    red Survey (n=6): con longitud 3 o 5 se alcanza el óptimo global, mientras
    que con 10 o 20 la búsqueda queda bloqueada en una solución con SHD=6, y
    aumentar las iteraciones no lo arregla.  La razón es que el vecindario sólo
    tiene n(n-1) arcos posibles, de modo que una lista tabú larga llega a
    prohibir todos los movimientos buenos.  Por eso, si no se indica, se escala
    con el tamaño del problema en vez de usar la constante 10 habitual.
    """
    if tabu_length is None:
        n = len(score.variables)
        tabu_length = max(3, round(0.1 * n * (n - 1)))
    rng = random.Random(seed)
    variables = list(score.variables)
    arcs: List[Arc] = []
    current = score.graph_score(arcs)
    best_arcs, best_score = list(arcs), current
    tabu: List[Arc] = []
    for _ in range(max_iter):
        candidates = []
        for _op, arc, candidate in _neighbours(arcs, variables, max_parents):
            key = tuple(sorted(arc))
            value = score.graph_score(candidate)
            if key in tabu and value <= best_score:
                continue  # tabú, salvo criterio de aspiración
            candidates.append((value, key, candidate))
        if not candidates:
            break
        rng.shuffle(candidates)
        value, key, candidate = max(candidates, key=lambda t: t[0])
        arcs, current = candidate, value
        tabu.append(key)
        if len(tabu) > tabu_length:
            tabu.pop(0)
        if current > best_score + 1e-12:
            best_arcs, best_score = list(arcs), current
    return sorted(best_arcs), best_score
