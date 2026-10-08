"""Modelos de ruido para el estudio de resiliencia NISQ.

Se replica el montaje de Soloviev, Bielza y Larrañaga (2022, sec. 5.2): la
intensidad se parametriza con un único escalar omega en [0, 1], los errores de
amortiguamiento (amplitud y fase) se aplican a las puertas de un cúbit y el
despolarizante a las de dos.

Los tres canales corresponden a fenómenos físicos distintos:

* **Amortiguamiento de amplitud**: pérdida de energía del sistema, asociada al
  tiempo de relajación T1.  El cúbit decae hacia |0>.
* **Amortiguamiento de fase**: pérdida de coherencia sin pérdida de energía,
  asociada al tiempo de desfase T2.  Destruye la información de fase relativa,
  que es justamente donde vive la interferencia de la que depende un algoritmo
  variacional.
* **Despolarizante**: con cierta probabilidad el estado se sustituye por el
  estado máximamente mixto.  Se aplica a las CNOT porque en el hardware actual
  las puertas de dos cúbits son las que peor fidelidad tienen, con diferencia.

Que el error de dos cúbits sea el dominante es lo que hace relevante contar
puertas CNOT y no sólo qubits: en `toy4`, la formulación de O'Gorman usa 144
CNOTs y la del espacio de órdenes 48.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

PUERTAS_1Q = ("h", "rx", "rz", "x", "sx", "u", "u1", "u2", "u3")
PUERTAS_2Q = ("cx", "cz")

CANALES = ("amplitude_damping", "phase_damping", "depolarizing")


def noise_model(
    kind: str,
    omega: float,
    one_qubit_gates: Iterable[str] = PUERTAS_1Q,
    two_qubit_gates: Iterable[str] = PUERTAS_2Q,
) -> Optional[NoiseModel]:
    """Modelo de ruido de intensidad `omega` para el canal indicado.

    Devuelve None si omega es 0, para que el simulador ideal no pague el coste
    de trayectorias estocásticas.
    """
    if kind not in CANALES:
        raise ValueError(f"canal desconocido: {kind!r}; use {CANALES}")
    if not 0 <= omega <= 1:
        raise ValueError(f"omega debe estar en [0, 1], recibido {omega!r}")
    if omega == 0:
        return None

    modelo = NoiseModel()
    if kind == "amplitude_damping":
        modelo.add_all_qubit_quantum_error(
            amplitude_damping_error(omega), list(one_qubit_gates)
        )
    elif kind == "phase_damping":
        modelo.add_all_qubit_quantum_error(
            phase_damping_error(omega), list(one_qubit_gates)
        )
    else:  # despolarizante sobre las puertas de dos cúbits
        modelo.add_all_qubit_quantum_error(
            depolarizing_error(omega, 2), list(two_qubit_gates)
        )
    return modelo


def noisy_simulator(
    kind: str, omega: float, num_qubits: int, seed: int = 0
) -> AerSimulator:
    """Simulador con el canal de ruido pedido.

    Con ruido hay que simular por trayectorias: cada disparo muestrea sus
    propios errores.  La matriz densidad sería exacta pero necesita 2^(2n)
    elementos, inviable más allá de una decena de qubits.
    """
    modelo = noise_model(kind, omega)
    metodo = "statevector" if num_qubits <= 26 else "matrix_product_state"
    if modelo is None:
        return AerSimulator(method=metodo, seed_simulator=seed)
    return AerSimulator(method=metodo, noise_model=modelo, seed_simulator=seed)


def gate_counts(circuit) -> Dict[str, int]:
    """Recuento de puertas relevante para el ruido: 1q, 2q y profundidad."""
    ops = circuit.count_ops()
    un_qubit = sum(ops.get(g, 0) for g in PUERTAS_1Q)
    dos_qubits = sum(ops.get(g, 0) for g in PUERTAS_2Q)
    return {
        "1q": un_qubit,
        "2q": dos_qubits,
        "profundidad": circuit.depth(),
    }
