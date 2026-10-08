"""El cambio a variables de espín debe preservar el valor de la función."""

import itertools

import numpy as np
import pytest

from bnslq.pseudoboolean import PseudoBoolean


def test_ising_conserva_el_valor():
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(1, 5))
        p = PseudoBoolean(n)
        for _ in range(int(rng.integers(1, 8))):
            k = int(rng.integers(0, n + 1))
            p.add(rng.choice(n, size=k, replace=False).tolist(), float(rng.normal()))
        terms, offset = p.to_ising()
        for x in itertools.product([0, 1], repeat=n):
            z = [1 - 2 * xi for xi in x]
            energia = offset + sum(
                c * np.prod([z[q] for q in qs]) for qs, c in terms.items()
            )
            assert energia == pytest.approx(p(x), abs=1e-9)


def test_evaluate_many_coincide_con_call():
    rng = np.random.default_rng(11)
    p = PseudoBoolean(6)
    for _ in range(12):
        k = int(rng.integers(0, 4))
        p.add(rng.choice(6, size=k, replace=False).tolist(), float(rng.normal()))
    X = rng.integers(0, 2, size=(50, 6))
    assert np.allclose(p.evaluate_many(X), [p(row) for row in X])


def test_grado_y_constante():
    p = PseudoBoolean(3).add([], 2.5).add([0], 1.0).add([0, 1, 2], -3.0)
    assert p.constant == 2.5
    assert p.degree == 3
    assert not p.is_qubo()
