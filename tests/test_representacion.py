"""El circuito debe codificar la distribución conjunta, no una parecida.

Si las amplitudes no reprodujeran P(x), la inferencia construida encima estaría
midiendo otra red.
"""

import itertools

import numpy as np
import pytest

from bnslq.data import CANCER, TOY4
from bnslq.representacion import RedCuantica, ajustar_cpts, angulo


def test_el_angulo_codifica_la_probabilidad():
    """sin^2(theta/2) debe ser exactamente p."""
    for p in (0.0, 0.01, 0.25, 0.5, 0.9, 1.0):
        assert np.sin(angulo(p) / 2) ** 2 == pytest.approx(p, abs=1e-12)


@pytest.fixture(scope="module", params=[TOY4, CANCER])
def red(request):
    r = request.param
    datos = r.sample(5000, seed=1).astype(int)
    variables = list(r.variables)
    return r, RedCuantica(r.arcs, ajustar_cpts(datos, r.arcs, variables), variables)


def test_la_distribucion_exacta_suma_uno(red):
    _, rc = red
    assert sum(rc.distribucion_exacta().values()) == pytest.approx(1.0, abs=1e-12)


def test_el_vector_de_estado_reproduce_la_distribucion(red):
    """Comprobación sin muestreo: |amplitud|^2 debe ser P(x) exactamente."""
    from qiskit.quantum_info import Statevector

    _, rc = red
    sv = Statevector(rc.circuito(medir=False))
    exacta = rc.distribucion_exacta()
    for x, p in exacta.items():
        # Qiskit indexa el estado con el qubit 0 como bit menos significativo
        i = sum(b << k for k, b in enumerate(x))
        assert abs(sv.data[i]) ** 2 == pytest.approx(p, abs=1e-10)


def test_respeta_el_orden_topologico(red):
    _, rc = red
    pos = {v: i for i, v in enumerate(rc.orden)}
    for padre, hijo in rc.arcs:
        assert pos[padre] < pos[hijo]


def test_rechaza_estructuras_ciclicas():
    with pytest.raises(ValueError):
        RedCuantica([("A", "B"), ("B", "A")], {}, ["A", "B"])


def test_el_suavizado_evita_probabilidades_degeneradas():
    """Sin suavizado, una configuración no observada daría 0/0."""
    import pandas as pd

    datos = pd.DataFrame({"A": [0] * 50, "B": [0] * 50})
    cpts = ajustar_cpts(datos, [("A", "B")], ["A", "B"])
    for tabla in cpts.values():
        for p in tabla.values():
            assert 0.0 < p < 1.0
