"""Fase 3 · Experimento 10: las tres variantes de diseño del circuito.

Completa el objetivo O6. La pregunta de diseño era: si el coste no cabe entero
en el circuito, ¿qué debe llevar el circuito? Hay tres respuestas, y hasta ahora
sólo se habían medido dos.

1. **Validez** --- el operador de coste lleva sólo la restricción de orden; la
   puntuación entra por el bucle clásico. Es un QAOA sobre la factibilidad.
2. **Nada** --- ansatz hardware-efficient genérico; toda la guía es clásica.
3. **Todo** --- se añaden variables que seleccionan el conjunto de padres de
   cada nodo, con lo que la puntuación se vuelve lineal y el Hamiltoniano
   completo es cuadrático y compilable. Es un QAOA canónico, al estilo de la
   cuadratización de Shikuri.

La tercera paga el precio en qubits: a los del orden hay que sumar uno por cada
conjunto candidato de padres. Y ese precio **depende de los datos**, no sólo del
número de variables: con más muestras sobreviven más conjuntos a la poda. Para
Cancer son 22 qubits con mil muestras, 24 con dos mil y 31 con veinte mil, de
modo que el experimento se limita al tamaño que cabe en el simulador.

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp10_tres_ansatze.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag
from bnslq.data import CANCER
from bnslq.factorial import FactorialOrderingSpace
from bnslq.metrics import confusion
from bnslq.qubo_pleno import OrdenQUBOPleno
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

SEMILLAS = (0, 1, 2, 3, 4)
MAXITER = 60
DISPAROS = 1024
N_MUESTRAS = 2000   # ver arriba: con más, el QUBO pleno no es simulable
MAX_QUBITS = 26     # límite del vector de estado en este equipo


def main() -> None:
    red = CANCER
    score = LocalScore(red.sample(N_MUESTRAS, seed=1), kind="bic")
    arcos_opt, score_opt = exact_best_dag(score, 2)
    energia = -score_opt
    print(f"red {red.name}: óptimo E0 = {energia:.4f}\n")

    factorial = FactorialOrderingSpace(score, max_parents=2)
    pleno = OrdenQUBOPleno(score, max_parents=2)

    casos = [
        ("validez (QAOA sobre H_trans)", factorial, "qaoa", 1),
        ("nada (hardware-efficient)", factorial, "hardware_efficient", 2),
        ("todo (QAOA pleno)", pleno, "qaoa", 1),
    ]

    print(f"{'variante':32s} {'qubits':>7} {'términos':>9} {'éxitos':>8} "
          f"{'gap medio':>11} {'factible':>9} {'seg':>6}")
    print("-" * 90)

    registros = []
    for nombre, espacio, ansatz, prof in casos:
        if espacio.num_vars > MAX_QUBITS:
            print(f"{nombre:32s} {espacio.num_vars:>7}   OMITIDO: excede "
                  f"{MAX_QUBITS} qubits, la simulación sería inviable")
            continue
        gaps, fracs, exitos = [], [], 0
        t0 = time.time()
        for semilla in SEMILLAS:
            vqa = OrderingVQA(espacio, ansatz=ansatz, depth=prof, alpha=0.25,
                              shots=DISPAROS, maxiter=MAXITER, seed=semilla)
            res = vqa.run()
            g = (res.best_cost - energia) if res.feasible else None
            gaps.append(g if g is not None else float("nan"))
            fracs.append(res.feasible_fraction_final)
            exitos += int(g is not None and abs(g) < 1e-6)
        seg = time.time() - t0
        n_term = espacio.h_trans.num_terms
        print(f"{nombre:32s} {espacio.num_vars:>7} {n_term:>9} "
              f"{exitos:>3}/{len(SEMILLAS):<4} {np.nanmean(gaps):>11.4f} "
              f"{100*np.mean(fracs):>8.1f}% {seg:>6.0f}")
        registros.append({
            "variante": nombre, "qubits": int(espacio.num_vars),
            "terminos_circuito": int(n_term), "exitos": int(exitos),
            "ejecuciones": len(SEMILLAS),
            "gap_medio": float(np.nanmean(gaps)),
            "fraccion_factible": float(np.mean(fracs)),
            "segundos": seg,
        })

    salida = Path(__file__).resolve().parents[1] / "resultados" / "exp10_tres_ansatze.json"
    salida.write_text(json.dumps({"red": red.name, "energia_optima": energia,
                                  "resultados": registros},
                                 indent=2, ensure_ascii=False))
    print(f"\nresultados en {salida}")


if __name__ == "__main__":
    main()
