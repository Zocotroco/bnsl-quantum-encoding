"""Validación de la formulación de O'Gorman.

Estas pruebas son el contrato de la Fase 1: si el Hamiltoniano no reproduce la
puntuación sobre los DAGs factibles y no tiene su mínimo en el DAG óptimo, el
QAOA que se monte encima no puede dar resultados correctos por mucho que
converja.
"""

import itertools

import networkx as nx
import pytest

from bnslq.classical import exact_best_dag
from bnslq.data import TOY4
from bnslq.qubo import BNSLHamiltonian
from bnslq.scores import LocalScore


@pytest.fixture(scope="module")
def toy():
    data = TOY4.sample(2000, seed=1)
    score = LocalScore(data, kind="bic")
    return score, BNSLHamiltonian(score, max_parents=2)


def _dags_factibles(variables, max_parents):
    pares = [(u, v) for u in variables for v in variables if u != v]
    for mask in range(1 << len(pares)):
        arcs = [pares[b] for b in range(len(pares)) if mask >> b & 1]
        g = nx.DiGraph()
        g.add_nodes_from(variables)
        g.add_edges_from(arcs)
        if nx.is_directed_acyclic_graph(g) and all(
            g.in_degree(v) <= max_parents for v in variables
        ):
            yield arcs


def test_cuenta_de_variables(toy):
    _, H = toy
    n = H.n
    assert H.n_a == n * (n - 1)
    assert H.n_r == n * (n - 1) // 2
    assert H.num_vars == 3 * n * (n - 1) // 2  # Soloviev et al. (2022), ec. (11)


def test_grado_igual_al_indegree_maximo(toy):
    score, _ = toy
    for m in (1, 2, 3):
        H = BNSLHamiltonian(score, max_parents=m)
        assert H.poly.degree == max(m, 2)  # H_trans y H_consist son cuadráticos
    assert BNSLHamiltonian(score, max_parents=2).poly.is_qubo()


def test_energia_igual_a_menos_puntuacion_en_todo_DAG_factible(toy):
    """Prueba central: H(x_G) = -score(G) para todo DAG factible."""
    score, H = toy
    n_comprobados = 0
    for arcs in _dags_factibles(list(score.variables), H.max_parents):
        x = H.encode(arcs)
        assert H(x) == pytest.approx(-score.graph_score(arcs), abs=1e-6)
        assert H.indegree_violation(x) == 0.0
        n_comprobados += 1
    assert n_comprobados == 443


def test_penalizaciones_castigan_ciclos(toy):
    """Un arco contrario al orden codificado debe costar delta_consist."""
    score, H = toy
    variables = list(score.variables)
    x = H.encode([(variables[0], variables[1])], order=variables)
    limpio = H(x)
    x_malo = list(x)
    x_malo[H.a_index[(1, 0)]] = 1  # arco contra el orden
    assert H(x_malo) - limpio >= H.delta_consist - 1e-6


def test_el_minimo_global_es_el_DAG_optimo(toy):
    """Fuerza bruta sobre las 2^18 asignaciones."""
    score, H = toy
    mejores_arcos, mejor_score = exact_best_dag(score, max_parents=2)
    bits, energia = H.brute_force_minimum(classical=True)
    assert H.is_feasible(bits)
    assert H.decode(bits) == mejores_arcos
    assert energia == pytest.approx(-mejor_score, abs=1e-6)


def test_sin_control_de_indegree_el_fundamental_es_infactible(toy):
    """Documenta por qué hace falta H_max (o su equivalente clásico).

    El polinomio de puntuación está truncado a |J| <= m, de modo que una
    asignación con in-degree mayor que m recibe una energía espuriamente baja.
    Soloviev et al. lo resuelven fuera del circuito; O'Gorman, con 2n variables
    de holgura.  Ignorarlo, como hace la formulación conceptual de partida,
    lleva el mínimo a un grafo no válido.
    """
    _, H = toy
    bits, _ = H.poly.brute_force_minimum()  # sin término de in-degree
    assert max(H.in_degrees(bits)) > H.max_parents


def test_hmax_reproduce_la_penalizacion_clasica():
    """Con variables de holgura, H_max equivale a delta * max(0, d-m)^2."""
    data = TOY4.sample(500, seed=2)
    score = LocalScore(data, kind="bic")
    H_clas = BNSLHamiltonian(score, max_parents=2, include_hmax=False)
    H_slack = BNSLHamiltonian(score, max_parents=2, include_hmax=True)
    assert H_slack.num_vars == H_clas.num_vars + 2 * H_clas.n
    for arcs in itertools.islice(_dags_factibles(list(score.variables), 2), 0, 443, 37):
        x_slack = H_slack.encode(arcs)
        assert H_slack(x_slack) == pytest.approx(-score.graph_score(arcs), abs=1e-6)


def test_las_cotas_ajustadas_son_menores_que_la_conservadora(toy):
    """Las cotas de O'Gorman/Zardini deben apretar respecto al recorrido total."""
    _, H = toy
    cotas = H.suggested_penalties()
    conservadora = H.suggested_penalty()
    assert cotas["trans"] < conservadora
    assert cotas["max"] < conservadora


def test_delta_es_no_negativa_y_exacta_para_m2(toy):
    """Delta_ji = max(0, -w_i({j}) - sum_k min(0, w_i({j,k}))) para m = 2."""
    _, H = toy
    for i in range(H.n):
        w = H.mobius_coefficients(i)
        for j in range(H.n):
            if j == i:
                continue
            esperado = -w.get((j,), 0.0) - sum(
                min(0.0, w.get(tuple(sorted((j, k))), 0.0))
                for k in range(H.n)
                if k not in (i, j)
            )
            assert H.delta(j, i) == pytest.approx(max(0.0, esperado))


def test_delta_consist_domina_a_delta_trans(toy):
    """La cota de O'Gorman exige delta_consist > (n-2) delta_trans."""
    _, H = toy
    assert H.delta_consist > (H.n - 2) * H.delta_trans
