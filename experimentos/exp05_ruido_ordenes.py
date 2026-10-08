"""Fase 2 · Experimento 5: optimización completa bajo ruido en el espacio de órdenes.

Réplica del análisis de Soloviev et al. (2022, fig. 7) sobre nuestra
codificación reducida: se ejecuta el **bucle híbrido entero** con ruido, de modo
que el optimizador clásico recibe ya una señal de coste corrompida.  Es lo
complementario del exp04, que congelaba los parámetros para aislar la
degradación del estado.

Lo que se busca es el umbral de omega a partir del cual el algoritmo deja de
encontrar el óptimo.  Soloviev lo sitúa en torno a 10^-2 para su codificación de
30 qubits en Cancer.

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp05_ruido_ordenes.py [red] [n_muestras]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag
from bnslq.data import NETWORKS
from bnslq.metrics import confusion
from bnslq.orderings import OrderingSpace
from bnslq.ruido import CANALES, noisy_simulator
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

OMEGAS = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SHOTS = 256
MAXITER = 80
SEMILLAS = (0, 1, 2)
PROFUNDIDAD = 2
ALPHA = 0.25


def main() -> None:
    red = NETWORKS[sys.argv[1] if len(sys.argv) > 1 else "cancer"]
    n_muestras = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    score = LocalScore(red.sample(n_muestras, seed=1), kind="bic")
    espacio = OrderingSpace(score, max_parents=2)
    arcos_opt, score_opt = exact_best_dag(score, max_parents=2)
    energia_opt = -score_opt

    print(f"red {red.name}: {red.n} nodos, {espacio.num_vars} qubits "
          f"(O'Gorman: {3*red.n*(red.n-1)//2})")
    print(f"óptimo exacto E0 = {energia_opt:.4f}\n")
    print(f"{'canal':18s} {'omega':>8} {'gap medio':>11} {'gap mejor':>11} "
          f"{'SHD':>4} {'fact.':>7} {'éxito':>7} {'evals':>6}")
    print("-" * 80)

    registros = []
    for canal in CANALES:
        for omega in OMEGAS:
            gaps, fracs, evals, exitos, shds = [], [], [], 0, []
            for semilla in SEMILLAS:
                sim = noisy_simulator(canal, omega, espacio.num_vars, seed=semilla)
                vqa = OrderingVQA(espacio, ansatz="qaoa", depth=PROFUNDIDAD,
                                  alpha=ALPHA, shots=SHOTS, maxiter=MAXITER,
                                  seed=semilla, simulator=sim)
                res = vqa.run()
                gap = res.best_cost - energia_opt
                gaps.append(gap)
                fracs.append(res.feasible_fraction_final)
                evals.append(res.n_evaluations)
                shds.append(confusion(res.best_arcs, red.arcs)["shd"]
                            if res.feasible else -1)
                exitos += gap < 1e-6
                if omega == 0.0 and len(gaps) == len(SEMILLAS):
                    break
            fila = {
                "canal": canal, "omega": omega,
                "gap_medio": float(np.mean(gaps)), "gap_mejor": float(min(gaps)),
                "shd_mejor": int(min(s for s in shds if s >= 0)) if any(s >= 0 for s in shds) else None,
                "fraccion_factible": float(np.mean(fracs)),
                "exitos": exitos, "ejecuciones": len(gaps),
                "evaluaciones_medias": float(np.mean(evals)),
            }
            registros.append(fila)
            print(f"{canal:18s} {omega:>8g} {fila['gap_medio']:>11.4f} "
                  f"{fila['gap_mejor']:>11.4f} {str(fila['shd_mejor']):>4} "
                  f"{100*fila['fraccion_factible']:>6.1f}% "
                  f"{exitos}/{len(gaps):<5} {fila['evaluaciones_medias']:>6.0f}",
                  flush=True)

    salida = (Path(__file__).resolve().parents[1] / "resultados"
              / f"exp05_ruido_ordenes_{red.name}.json")
    salida.write_text(json.dumps({
        "red": red.name, "n_muestras": n_muestras, "qubits": espacio.num_vars,
        "profundidad": PROFUNDIDAD, "alpha": ALPHA, "shots": SHOTS,
        "maxiter": MAXITER, "semillas": list(SEMILLAS), "omegas": list(OMEGAS),
        "energia_optima": energia_opt, "arcos_optimos": arcos_opt,
        "resultados": registros,
    }, indent=2, ensure_ascii=False))
    print(f"\nresultados en {salida}")


if __name__ == "__main__":
    main()
