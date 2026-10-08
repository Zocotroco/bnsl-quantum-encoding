"""La codificación factorial debe ser biyectiva y dar el mismo óptimo.

Si la biyección fallara, el espacio de búsqueda dejaría de cubrir todos los
órdenes y el algoritmo podría no poder alcanzar el óptimo.
"""

import itertools
import math

import pytest

from bnslq.classical import exact_best_dag
from bnslq.data import CANCER, SURVEY, TOY4
from bnslq.factorial import FactorialOrderingSpace
from bnslq.orderings import OrderingSpace
from bnslq.pseudoboolean import PseudoBoolean
from bnslq.scores import LocalScore


@pytest.fixture(scope="module", params=[TOY4, CANCER])
def caso(request):
    red = request.param
    score = LocalScore(red.sample(2000, seed=1), kind="bic")
    return red, score, FactorialOrderingSpace(score, max_parents=2)


def _todas(espacio):
    for e in range(1 << espacio.num_vars):
        yield "".join(str(e >> b & 1) for b in range(espacio.num_vars))


def test_indicator_es_exacto():
    """El polinomio indicador debe reproducir el predicado sobre toda asignación."""
    for card, bits in ((5, 3), (7, 3), (3, 2), (6, 3)):
        p = PseudoBoolean.indicator(
            bits, list(range(bits)),
            lambda v, c=card: sum(x << i for i, x in enumerate(v)) >= c,
        )
        for v in itertools.product([0, 1], repeat=bits):
            esperado = float(sum(x << i for i, x in enumerate(v)) >= card)
            assert p(v) == pytest.approx(esperado)


def test_cuenta_de_qubits(caso):
    _, _, F = caso
    esperado = sum(max(1, math.ceil(math.log2(c))) for c in range(F.n, 1, -1))
    assert F.num_vars == esperado
    assert F.num_vars < F.n * (F.n - 1) // 2  # menos que la codificación por torneo


def test_encode_decode_es_biyectiva(caso):
    """Todo orden tiene una codificación y sólo una, y viceversa."""
    _, _, F = caso
    ordenes = list(itertools.permutations(F.variables))
    codigos = set()
    for orden in ordenes:
        x = F.encode(orden)
        assert F.decode_order(x) == orden
        codigos.add(tuple(x))
    assert len(codigos) == len(ordenes)  # sin colisiones


def test_el_numero_de_estados_validos_es_n_factorial(caso):
    _, _, F = caso
    validos = [b for b in _todas(F) if F.is_total_order(b)]
    assert len(validos) == math.factorial(F.n)


def test_es_mas_densa_que_el_torneo(caso):
    """La razón de ser de esta codificación: muchos menos estados inválidos."""
    _, score, F = caso
    O = OrderingSpace(score, max_parents=2)
    densidad_fac = math.factorial(F.n) / 2**F.num_vars
    densidad_tor = math.factorial(O.n) / 2**O.num_vars
    assert densidad_fac > densidad_tor


def test_la_penalizacion_es_local(caso):
    """Cada dígito se valida con sus propios qubits, sin acoplar dígitos."""
    _, _, F = caso
    for term in F.h_trans.terms:
        if not term:
            continue
        digitos = {
            k for k, qs in enumerate(F.digit_qubits) if any(q in term for q in qs)
        }
        assert len(digitos) == 1, "un término acopla dígitos distintos"


def test_los_infactibles_cuestan_mas(caso):
    _, _, F = caso
    validos = [F.cost(b) for b in _todas(F) if F.is_total_order(b)]
    invalidos = [F.cost(b) for b in _todas(F) if not F.is_total_order(b)]
    if invalidos:  # con n=4 todas las cardinalidades son potencia de dos salvo 3
        assert max(validos) < min(invalidos)


@pytest.mark.parametrize("red", [TOY4, CANCER, SURVEY])
def test_el_minimo_global_es_el_DAG_optimo(red):
    score = LocalScore(red.sample(2000, seed=1), kind="bic")
    F = FactorialOrderingSpace(score, max_parents=2)
    arcos_opt, score_opt = exact_best_dag(score, 2, cache=F.parent_sets)
    bits, energia = F.brute_force_minimum()
    assert F.is_total_order(bits)
    assert energia == pytest.approx(-score_opt, abs=1e-9)
    assert F.dag_of(bits)[0] == arcos_opt


def test_da_el_mismo_optimo_que_el_torneo(caso):
    """Ambas codificaciones deben describir exactamente el mismo problema."""
    _, score, F = caso
    O = OrderingSpace(score, max_parents=2)
    _, e_fac = F.brute_force_minimum()
    _, e_tor = O.brute_force_minimum()
    assert e_fac == pytest.approx(e_tor, abs=1e-9)
