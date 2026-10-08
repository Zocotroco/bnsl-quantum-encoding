"""Los modelos de ruido deben degradar el resultado, y sólo cuando toca."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from bnslq.ruido import CANALES, gate_counts, noise_model, noisy_simulator


def test_sin_ruido_no_hay_modelo():
    """omega = 0 debe devolver None: el simulador ideal no debe pagar el coste
    de la simulación por trayectorias."""
    for canal in CANALES:
        assert noise_model(canal, 0.0) is None


@pytest.mark.parametrize("canal", CANALES)
def test_el_modelo_se_construye(canal):
    modelo = noise_model(canal, 0.01)
    assert modelo is not None
    assert modelo.noise_instructions


def test_canal_desconocido():
    with pytest.raises(ValueError):
        noise_model("inexistente", 0.01)


@pytest.mark.parametrize("omega", [-0.1, 1.5])
def test_omega_fuera_de_rango(omega):
    with pytest.raises(ValueError):
        noise_model("depolarizing", omega)


def test_el_amortiguamiento_de_amplitud_lleva_el_estado_a_cero():
    """Con omega = 1 el canal de amplitud vacía el cúbit: todo debe medir 0."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    sim = noisy_simulator("amplitude_damping", 1.0, 1, seed=0)
    counts = sim.run(qc, shots=512, seed_simulator=0).result().get_counts()
    assert counts.get("0", 0) == 512


def test_el_despolarizante_solo_afecta_a_puertas_de_dos_cubits():
    """Un circuito sin CNOTs debe salir intacto del canal despolarizante."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    sim = noisy_simulator("depolarizing", 0.9, 1, seed=0)
    counts = sim.run(qc, shots=512, seed_simulator=0).result().get_counts()
    assert counts.get("1", 0) == 512


def test_el_despolarizante_fuerte_aleatoriza_una_bell():
    """Con una CNOT y omega alto, la correlación de una pareja de Bell se pierde."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    limpio = noisy_simulator("depolarizing", 0.0, 2, seed=0)
    ruidoso = noisy_simulator("depolarizing", 0.95, 2, seed=0)
    c_limpio = limpio.run(qc, shots=2048, seed_simulator=0).result().get_counts()
    c_ruido = ruidoso.run(qc, shots=2048, seed_simulator=0).result().get_counts()
    correladas = lambda c: (c.get("00", 0) + c.get("11", 0)) / sum(c.values())
    assert correladas(c_limpio) > 0.99
    assert correladas(c_ruido) < 0.75


def test_gate_counts_separa_una_y_dos_cubits():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(0.3, 1)
    qc.cx(0, 1)
    qc.cx(1, 0)
    c = gate_counts(qc)
    assert c["1q"] == 2 and c["2q"] == 2
