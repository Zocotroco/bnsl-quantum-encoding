"""Métricas de comparación de estructuras."""

from __future__ import annotations

from typing import Iterable, Set, Tuple

Arc = Tuple[str, str]


def _arc_set(arcs: Iterable[Arc]) -> Set[Arc]:
    return {(a, b) for a, b in arcs}


def structural_hamming_distance(learned: Iterable[Arc], true: Iterable[Arc]) -> int:
    """Distancia de Hamming estructural (SHD) entre dos DAGs.

    Cuenta, sobre cada par no ordenado de nodos, una unidad por arco ausente,
    sobrante o invertido.  SHD = 0 significa recuperación exacta de la
    estructura.  Es la métrica que usan Soloviev et al. (2022) en el benchmark
    Cancer, lo que permite comparar cifras directamente.
    """
    L, T = _arc_set(learned), _arc_set(true)
    pairs = {frozenset(a) for a in L} | {frozenset(a) for a in T}
    shd = 0
    for pair in pairs:
        u, v = tuple(pair) if len(pair) == 2 else (tuple(pair)[0],) * 2
        in_l = (u, v) in L or (v, u) in L
        in_t = (u, v) in T or (v, u) in T
        if in_l and in_t:
            shd += 0 if ((u, v) in L) == ((u, v) in T) else 1  # invertido
        else:
            shd += 1  # sobrante o ausente
    return shd


def confusion(learned: Iterable[Arc], true: Iterable[Arc]) -> dict:
    """Aciertos/fallos de arcos dirigidos, útil para diagnosticar el SHD."""
    L, T = _arc_set(learned), _arc_set(true)
    reversed_ = {(a, b) for a, b in L if (b, a) in T}
    return {
        "correctos": len(L & T),
        "invertidos": len(reversed_),
        "sobrantes": len(L - T - reversed_),
        "ausentes": len(T - L - {(b, a) for a, b in reversed_}),
        "shd": structural_hamming_distance(L, T),
    }
