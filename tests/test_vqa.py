"""El VQA sobre el espacio de órdenes debe recuperar el óptimo global.

Se usa toy4 (6 qubits, 24 órdenes válidos de 64) para que la prueba sea rápida
y determinista.
"""

import pytest

from bnslq.classical import exact_best_dag
from bnslq.data import TOY4
from bnslq.orderings import OrderingSpace
from bnslq.scores import LocalScore
from bnslq.vqa import ANSATZE, OrderingVQA


@pytest.fixture(scope="module")
def espacio():
    score = LocalScore(TOY4.sample(2000, seed=1), kind="bic")
    return score, OrderingSpace(score, max_parents=2)


@pytest.mark.parametrize("ansatz", sorted(ANSATZE))
def test_encuentra_el_optimo(espacio, ansatz):
    score, O = espacio
    _, mejor = exact_best_dag(score, max_parents=2)
    vqa = OrderingVQA(O, ansatz=ansatz, depth=2, alpha=0.25,
                      shots=512, maxiter=80, seed=0)
    res = vqa.run()
    assert res.feasible
    assert res.best_cost == pytest.approx(-mejor, abs=1e-6)
    assert res.best_arcs == exact_best_dag(score, max_parents=2)[0]


@pytest.mark.parametrize("ansatz", sorted(ANSATZE))
def test_el_circuito_usa_un_qubit_por_par(espacio, ansatz):
    _, O = espacio
    stats = OrderingVQA(O, ansatz=ansatz, depth=1).circuit_stats()
    assert stats["qubits"] == O.n * (O.n - 1) // 2 == 6


def test_el_ansatz_qaoa_favorece_el_subespacio_factible(espacio):
    """Meter H_trans en el circuito debe concentrar amplitud sobre los órdenes.

    Con parámetros iniciales el muestreo es esencialmente uniforme (24/64 =
    37.5% factible); tras optimizar, la fracción factible debe subir.
    """
    _, O = espacio
    res = OrderingVQA(O, ansatz="qaoa", depth=2, alpha=1.0,
                      shots=1024, maxiter=80, seed=0).run()
    assert res.feasible_fraction_final > res.feasible_fraction_initial
    assert res.feasible_fraction_final > 24 / 64


def test_ansatz_desconocido():
    score = LocalScore(TOY4.sample(200, seed=1), kind="bic")
    with pytest.raises(ValueError):
        OrderingVQA(OrderingSpace(score), ansatz="inexistente")
