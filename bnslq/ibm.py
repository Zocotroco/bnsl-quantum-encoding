"""Conexión con IBM Quantum Platform.

Adaptado de `conectar_ibm.py`, una utilidad que Raúl ya tenía de otro proyecto.
Se conserva su estructura por pasos, que es la correcta cuando la cuota es
escasa:

    1. `guardar_cuenta(...)`   una sola vez, deja la credencial en
                               ~/.qiskit/qiskit-ibm.json, fuera del código.
    2. `probar_conexion()`     NO gasta cuota: lista backends y colas.
    3. `elegir_backend(...)`   NO gasta cuota: selecciona el menos ocupado.
    4. `estimar_coste(...)`    NO gasta cuota: estima los segundos de QPU de un
                               circuito antes de enviarlo.

El canal antiguo `ibm_quantum` está retirado; se usa `ibm_quantum_platform`,
que además requiere el CRN de la instancia.

SEGURIDAD: no escribir nunca el token en el código ni en los resultados.
"""

from __future__ import annotations

from typing import Optional

CANAL = "ibm_quantum_platform"


def guardar_cuenta(token: str, instance: str, por_defecto: bool = True) -> None:
    """Guarda las credenciales en disco. Ejecutar una sola vez."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel=CANAL, token=token, instance=instance,
        set_as_default=por_defecto, overwrite=True,
    )
    print("credenciales guardadas en ~/.qiskit/qiskit-ibm.json")


def servicio():
    """Servicio a partir de la cuenta guardada."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService()


def probar_conexion(verbose: bool = True):
    """Lista los procesadores reales operativos. No consume cuota."""
    s = servicio()
    backends = s.backends(operational=True, simulator=False)
    if verbose:
        print(f"conexión establecida. {len(backends)} procesadores operativos:")
        for b in backends:
            try:
                cola = b.status().pending_jobs
            except Exception:
                cola = "?"
            print(f"  {b.name:22s} {b.num_qubits:4d} qubits   cola: {cola}")
    return s, backends


def elegir_backend(min_qubits: int = 28, verbose: bool = True):
    """Procesador menos ocupado con al menos `min_qubits`. No consume cuota."""
    s = servicio()
    b = s.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
    if verbose:
        try:
            cola = b.status().pending_jobs
        except Exception:
            cola = "?"
        print(f"elegido {b.name} ({b.num_qubits} qubits, cola: {cola})")
    return b


def estimar_coste(circuito_isa, backend, shots: int) -> Optional[float]:
    """Estima los segundos de QPU de un circuito ya transpilado.

    El tiempo de ejecución es aproximadamente la duración del circuito por el
    número de disparos, más la sobrecarga fija del trabajo.  Sirve para no
    enviar a ciegas algo que se coma la cuota.
    """
    try:
        dt = backend.dt  # duración de un ciclo de instrucción, en segundos
        duracion = circuito_isa.duration
        if duracion is None or dt is None:
            return None
        return duracion * dt * shots
    except Exception:
        return None


def resumen_cuota(trabajo) -> Optional[float]:
    """Segundos de QPU que ha consumido un trabajo ya ejecutado."""
    try:
        uso = trabajo.usage_estimation or {}
        return uso.get("quantum_seconds")
    except Exception:
        return None
