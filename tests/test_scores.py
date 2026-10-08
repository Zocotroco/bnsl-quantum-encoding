"""Las puntuaciones locales deben coincidir con las de pgmpy, que es la
referencia de facto en Python.  Si esto falla, todo lo demás es ruido."""

import itertools

import numpy as np
import pandas as pd
import pytest

from bnslq.scores import LocalScore


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    d = pd.DataFrame(
        {
            "A": rng.integers(0, 2, 400),
            "B": rng.integers(0, 3, 400),
            "C": rng.integers(0, 2, 400),
        }
    )
    d["D"] = (d["A"] + d["B"]) % 3
    return d


@pytest.mark.parametrize("kind", ["bic", "bdeu", "k2"])
def test_coincide_con_pgmpy(data, kind):
    from pgmpy.estimators import BIC, BDeu, K2

    ess = 7.0
    mine = LocalScore(data, kind=kind, ess=ess)
    theirs = {
        "bic": lambda: BIC(data),
        "bdeu": lambda: BDeu(data, equivalent_sample_size=ess),
        "k2": lambda: K2(data),
    }[kind]()
    for node in data.columns:
        others = [c for c in data.columns if c != node]
        for r in range(len(others) + 1):
            for pa in itertools.combinations(others, r):
                assert mine(node, pa) == pytest.approx(
                    theirs.local_score(node, list(pa)), rel=1e-9, abs=1e-9
                )


def test_descomponibilidad(data):
    score = LocalScore(data, kind="bic")
    arcs = [("A", "D"), ("B", "D"), ("A", "C")]
    esperado = score("A", []) + score("B", []) + score("C", ["A"]) + score("D", ["A", "B"])
    assert score.graph_score(arcs) == pytest.approx(esperado)


def test_detecta_dependencia_funcional(data):
    """D = (A + B) % 3 es determinista: {A, B} debe ganar a cualquier otro.

    Nótese que A *en solitario* no aporta información sobre D (fijado A, la
    variable D sigue siendo uniforme porque B lo es), de modo que BIC penaliza
    los parámetros extra y puntúa peor que el conjunto vacío.  Es la navaja de
    Ockham funcionando, no un error.
    """
    score = LocalScore(data, kind="bic")
    otros = [[], ["A"], ["B"], ["C"], ["A", "C"], ["B", "C"]]
    assert all(score("D", ["A", "B"]) > score("D", pa) for pa in otros)
    assert score("D", ["A"]) < score("D", [])
    assert score("D", ["B"]) > score("D", [])


def test_nodo_no_puede_ser_su_propio_padre(data):
    with pytest.raises(ValueError):
        LocalScore(data)("A", ["A"])
