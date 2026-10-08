#!/usr/bin/env python
"""Guarda la credencial de IBM Quantum sin que aparezca en pantalla ni en el
historial de la shell.

Uso:
    ./.venv/bin/python guardar_credencial.py

Pide la API key con entrada oculta y el CRN de la instancia, los guarda en
~/.qiskit/qiskit-ibm.json y comprueba la conexión, todo ello sin consumir cuota.
"""

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bnslq.ibm import guardar_cuenta, probar_conexion


def main() -> None:
    print("Credencial de IBM Quantum Platform (canal ibm_quantum_platform)\n")
    token = getpass.getpass("API key (no se mostrará): ").strip()
    if not token:
        print("no se introdujo ninguna API key; nada que hacer")
        return

    crn = input("CRN de la instancia (crn:v1:bluemix:...): ").strip()
    if not crn.startswith("crn:"):
        print(f"aviso: el CRN no empieza por 'crn:' ({crn[:20]}...). Se guarda igualmente.")

    guardar_cuenta(token=token, instance=crn)

    print("\ncomprobando la conexión (no consume cuota)...\n")
    try:
        probar_conexion()
        print("\nlisto: la credencial funciona.")
    except Exception as e:
        print(f"\nla conexión falló: {type(e).__name__}: {e}")
        print("revisa que la API key y el CRN correspondan a la misma instancia.")


if __name__ == "__main__":
    main()
