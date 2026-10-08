"""El QUBO pleno lleva la puntuación dentro: su mínimo debe ser el DAG óptimo."""

import pytest

from bnslq.classical import exact_best_dag
from bnslq.data import CANCER
from bnslq.qubo_pleno import OrdenQUBOPleno
from bnslq.scores import LocalScore


@pytest.fixture(scope="module")
def caso():
    score = LocalScore(CANCER.sample(2000, seed=1), kind="bic")
    return score, OrdenQUBOPleno(score, max_parents=2)


def test_cuenta_de_variables(caso):
    _, Q = caso
    assert Q.n_r == Q.n * (Q.n - 1) // 2
    assert Q.num_vars == Q.n_r + sum(len(c) for c in Q.candidatos)


def test_el_polinomio_es_cuadratico(caso):
    """Es la razón de ser de esta variante: un QAOA canónico necesita grado 2."""
    _, Q = caso
    assert Q.poly.is_qubo()


def test_el_minimo_global_es_el_DAG_optimo(caso):
    score, Q = caso
    arcos_opt, score_opt = exact_best_dag(score, 2, cache=Q.parent_sets)
    bits, energia = Q.brute_force_minimum()
    assert Q.is_total_order(bits)
    assert energia == pytest.approx(-score_opt, abs=1e-6)
    assert Q.dag_of(bits)[0] == arcos_opt


def test_no_elegir_padres_no_compensa(caso):
    """Sin el desplazamiento de la puntuación, el óptimo sería no elegir nada.

    Es el error que tuvo la primera versión: como -score es un número positivo
    grande, al optimizador le salía a cuenta violar la restricción de unicidad.
    """
    _, Q = caso
    vacio = "0" * Q.num_vars
    bits_opt, energia_opt = Q.brute_force_minimum()
    assert Q.cost(vacio) > energia_opt


def test_encode_reproduce_la_puntuacion_del_orden(caso):
    import itertools
    score, Q = caso
    for orden in itertools.islice(itertools.permutations(Q.variables), 0, 120, 17):
        x = Q.encode(orden)
        assert Q.is_total_order(x)
        _, puntuacion = Q.dag_of(x)
        esperado = Q.parent_sets.score_of_order(orden)
        assert puntuacion == pytest.approx(esperado, abs=1e-9)
        assert Q.cost(x) == pytest.approx(-esperado, abs=1e-6)
