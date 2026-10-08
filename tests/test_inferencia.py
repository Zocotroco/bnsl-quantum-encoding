"""La inferencia cuántica debe coincidir con la exacta."""

import pytest

from bnslq.data import CANCER
from bnslq.inferencia import InferenciaCuantica
from bnslq.representacion import RedCuantica, ajustar_cpts


@pytest.fixture(scope="module")
def inf():
    datos = CANCER.sample(5000, seed=1).astype(int)
    variables = list(CANCER.variables)
    rc = RedCuantica(CANCER.arcs, ajustar_cpts(datos, CANCER.arcs, variables), variables)
    return InferenciaCuantica(rc)


def test_la_marginal_coincide_con_la_exacta(inf):
    """El muestreo del circuito debe reproducir P(E) dentro del error estadístico."""
    cond = {"Smoker": 1}
    exacta = sum(p for x, p in inf._exacta.items()
                 if x[inf.red.indice["Smoker"]] == 1)
    est, _ = inf.probabilidad_muestreo(cond, disparos=40000)
    assert est == pytest.approx(exacta, abs=0.01)


def test_la_qae_coincide_con_la_exacta(inf):
    cond = {"Smoker": 1, "Pollution": 1}
    exacta = sum(p for x, p in inf._exacta.items()
                 if all(x[inf.red.indice[v]] == b for v, b in cond.items()))
    est, coste = inf.probabilidad_qae(cond, epsilon=0.01)
    assert est == pytest.approx(exacta, abs=0.03)
    assert coste > 0


def test_la_condicional_coincide_con_la_exacta(inf):
    consulta, evidencia = {"Cancer": 1}, {"Smoker": 1}
    exacta = inf.exacta(consulta, evidencia)
    est, _ = inf.condicional_qae(consulta, evidencia, epsilon=0.01)
    assert est == pytest.approx(exacta, abs=0.03)


def test_la_bandera_marca_exactamente_la_evidencia(inf):
    """El qubit auxiliar debe valer 1 si y sólo si se cumple la condición."""
    from qiskit.quantum_info import Statevector

    cond = {"Smoker": 1, "Pollution": 1}
    sv = Statevector(inf.circuito_marcado(cond))
    n = inf.red.n
    for i, amp in enumerate(sv.data):
        if abs(amp) ** 2 < 1e-12:
            continue
        x = tuple((i >> k) & 1 for k in range(n))
        bandera = (i >> n) & 1
        esperado = int(all(x[inf.red.indice[v]] == b for v, b in cond.items()))
        assert bandera == esperado
