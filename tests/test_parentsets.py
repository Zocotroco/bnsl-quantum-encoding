"""La poda por conjuntos candidatos de padres tiene que ser EXACTA.

Si perdiera el óptimo, todo lo que se construya encima quedaría invalidado, así
que se compara contra fuerza bruta sobre todos los subconjuntos permitidos.
"""

import itertools

import pytest

from bnslq.classical import best_dag_for_order, exact_best_dag
from bnslq.data import CANCER, SURVEY, TOY4
from bnslq.parentsets import ParentSetCache
from bnslq.scores import LocalScore


@pytest.fixture(scope="module", params=[TOY4, CANCER])
def caso(request):
    red = request.param
    score = LocalScore(red.sample(2000, seed=1), kind="bic")
    return red, score, ParentSetCache(score, max_parents=2)


def test_el_vacio_siempre_es_candidato(caso):
    """Garantiza que `best_for` termina para cualquier conjunto de predecesores."""
    _, _, C = caso
    for node in C.variables:
        assert any(len(W) == 0 for W, _ in C.candidates[node])


def test_los_candidatos_van_ordenados_por_puntuacion(caso):
    _, _, C = caso
    for node in C.variables:
        valores = [v for _, v in C.candidates[node]]
        assert valores == sorted(valores, reverse=True)


def test_best_for_coincide_con_fuerza_bruta(caso):
    """Para TODO subconjunto de predecesores posible."""
    _, score, C = caso
    for node in C.variables:
        otros = [v for v in C.variables if v != node]
        for r in range(len(otros) + 1):
            for allowed in itertools.combinations(otros, r):
                bruto = max(
                    score(node, W)
                    for t in range(C.max_parents + 1)
                    for W in itertools.combinations(allowed, t)
                )
                _, podado = C.best_for(node, allowed)
                assert podado == pytest.approx(bruto, abs=1e-9)


def test_coincide_en_todos_los_ordenes(caso):
    _, score, C = caso
    for orden in itertools.permutations(C.variables):
        _, sin_poda = best_dag_for_order(score, orden, C.max_parents)
        _, con_poda = C.best_dag_for_order(orden)
        assert con_poda == pytest.approx(sin_poda, abs=1e-9)


def test_la_poda_reduce_de_verdad():
    """En una red con estructura real debe descartarse una mayoría de familias."""
    score = LocalScore(SURVEY.sample(20000, seed=1), kind="bic")
    C = ParentSetCache(score, max_parents=2)
    assert C.n_candidates < C.n_total / 2


def test_exact_best_dag_da_el_mismo_optimo_con_y_sin_cache(caso):
    red, score, C = caso
    a1, v1 = exact_best_dag(score, 2, cache=C)
    a2, v2 = exact_best_dag(score, 2)
    assert v1 == pytest.approx(v2, abs=1e-9)
    assert a1 == a2
