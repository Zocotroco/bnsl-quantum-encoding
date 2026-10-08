"""Validación de la formulación en el espacio de órdenes (Fase 2).

El contrato es el mismo que en la Fase 1: si el mínimo global del coste no es el
DAG óptimo, el VQA que se monte encima no puede dar resultados correctos.
"""

import itertools

import pytest

from bnslq.classical import best_dag_for_order, exact_best_dag
from bnslq.data import CANCER, TOY4
from bnslq.orderings import OrderingSpace
from bnslq.scores import LocalScore


@pytest.fixture(scope="module")
def cancer():
    score = LocalScore(CANCER.sample(2000, seed=1), kind="bic")
    return score, OrderingSpace(score, max_parents=2)


def _todas_las_asignaciones(espacio):
    for entero in range(1 << espacio.num_vars):
        yield "".join(str(entero >> k & 1) for k in range(espacio.num_vars))


def test_cuenta_de_qubits(cancer):
    _, O = cancer
    n = O.n
    assert O.num_vars == n * (n - 1) // 2
    assert O.num_vars * 3 == 3 * n * (n - 1) // 2  # un tercio de O'Gorman


def test_encode_decode_es_biyectiva(cancer):
    _, O = cancer
    ordenes = list(itertools.permutations(O.variables))
    assert len(ordenes) == 120
    for orden in ordenes:
        assert O.decode_order(O.encode(orden)) == orden


def test_solo_hay_n_factorial_ordenes_validos(cancer):
    _, O = cancer
    validas = [b for b in _todas_las_asignaciones(O) if O.is_total_order(b)]
    assert len(validas) == 120
    assert len(list(_todas_las_asignaciones(O))) == 1024


def test_los_tres_criterios_de_validez_coinciden(cancer):
    """is_total_order, decode_order y violations deben decir lo mismo siempre."""
    _, O = cancer
    for bits in _todas_las_asignaciones(O):
        es_orden = O.is_total_order(bits)
        assert es_orden == (O.decode_order(bits) is not None)
        assert es_orden == (O.violations(bits) == 0)


def test_los_infactibles_cuestan_mas_que_cualquier_orden(cancer):
    """Sin este anclaje el mínimo global caería sobre una asignación inválida.

    El coste de un orden es menos una log-verosimilitud y vale varios miles;
    una penalización suelta se queda muy por debajo.  Por eso las asignaciones
    infactibles se anclan sobre una cota del peor coste factible.
    """
    _, O = cancer
    validos = [O.cost(b) for b in _todas_las_asignaciones(O) if O.is_total_order(b)]
    invalidos = [O.cost(b) for b in _todas_las_asignaciones(O) if not O.is_total_order(b)]
    assert max(validos) < min(invalidos)


def test_la_penalizacion_crece_con_las_violaciones(cancer):
    _, O = cancer
    por_violaciones = {}
    for bits in _todas_las_asignaciones(O):
        v = O.violations(bits)
        if v:
            por_violaciones.setdefault(v, []).append(O.cost(bits))
    niveles = sorted(por_violaciones)
    assert len(niveles) > 1
    for a, b in zip(niveles, niveles[1:]):
        assert max(por_violaciones[a]) < min(por_violaciones[b])


@pytest.mark.parametrize("red", [TOY4, CANCER])
def test_el_minimo_global_es_el_DAG_optimo(red):
    """Fuerza bruta sobre las 2^(n(n-1)/2) asignaciones."""
    score = LocalScore(red.sample(2000, seed=1), kind="bic")
    O = OrderingSpace(score, max_parents=2)
    arcos_opt, score_opt = exact_best_dag(score, max_parents=2)
    bits, energia = O.brute_force_minimum()
    assert O.is_total_order(bits)
    assert energia == pytest.approx(-score_opt, abs=1e-9)
    assert O.dag_of(bits)[0] == arcos_opt


def test_coincide_con_la_busqueda_en_el_espacio_de_DAGs(cancer):
    """El mejor orden debe dar la misma puntuación que el mejor DAG.

    Es la propiedad que justifica toda la Fase 2: buscar permutaciones no
    pierde nada frente a buscar grafos, porque todo DAG admite un orden
    topológico y, fijado el orden, los padres óptimos se calculan exactamente.
    """
    score, O = cancer
    mejor_por_orden = max(p for _, _, p in O.enumerate_orders())
    _, mejor_por_dag = exact_best_dag(score, max_parents=2)
    assert mejor_por_orden == pytest.approx(mejor_por_dag, abs=1e-9)


def test_best_dag_for_order_respeta_el_orden(cancer):
    score, O = cancer
    orden = tuple(O.variables)
    arcos, _ = best_dag_for_order(score, orden, max_parents=2)
    posicion = {v: p for p, v in enumerate(orden)}
    assert all(posicion[padre] < posicion[hijo] for padre, hijo in arcos)
