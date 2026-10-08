"""Fase 2 · Experimento 6: VQA en el espacio de órdenes sobre Asia (n = 8).

Es el experimento que decide si la formulación **escala**. Asia es el primer
benchmark que hemos encontrado donde el problema es genuinamente difícil:

* el óptimo exacto tiene SHD = 1 (el arco asia -> tub es casi indetectable,
  porque `asia` sólo vale "sí" un 1 % de las veces);
* **hill climbing falla**: SHD = 8, gap 25,8;
* sólo 40 320 de 2^28 asignaciones son órdenes válidos, un **0,015 %**, así que
  el muestreo uniforme no encontrará nada.

Con la formulación de O'Gorman harían falta 84 qubits, fuera de todo alcance.
En el espacio de órdenes son 28, simulables con vector de estado (4,3 GB).

Arranque en caliente
--------------------
El sondeo previo (`resultados/sondeo_asia.log`) mostró que **beta es el
parámetro decisivo**: con beta = 0,4 la fracción factible es 0 % para los cuatro
valores de gamma probados, y toda la concentración aparece en beta = 0,8, donde
se llega al 3,12 % (208x el azar). Inicializar uniformemente en [0, pi] tiraría
la mayoría de los arranques a la zona muerta y desperdiciaría horas de cómputo,
así que se parte del punto que concentra, con dispersión por semilla.

COSTE: 60-110 s por evaluación. Lanzar en segundo plano y con paciencia.

    nohup ./.venv/bin/python -u -W ignore experimentos/exp06_asia.py \
        > resultados/exp06.log 2>&1 &
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag, hill_climbing, tabu_search
from bnslq.data import ASIA
from bnslq.metrics import confusion
from bnslq.orderings import OrderingSpace
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

N_MUESTRAS = 50000
SHOTS = 1024
MAXITER = 30
SEMILLAS = (0, 1, 2)
PUNTO_CALIENTE = np.array([1.0, 0.8])   # (gamma, beta) del sondeo, p = 1


def control_azar(espacio, shots, repeticiones=200, seed=0):
    """¿Encuentra el óptimo el muestreo uniforme con el mismo presupuesto?

    Sin esta línea base no se puede afirmar nada: en Cancer el azar acertaba el
    100 % de las veces y eso invalidó aquel resultado.
    """
    rng = np.random.default_rng(seed)
    _, mejor = exact_best_dag(espacio.score, espacio.max_parents,
                              cache=espacio.parent_sets)
    exitos = validos = 0
    for _ in range(repeticiones):
        X = rng.integers(0, 2, size=(shots, espacio.num_vars))
        costes = []
        for fila in X:
            bits = "".join(map(str, fila))
            if espacio.is_total_order(bits):
                validos += 1
                costes.append(espacio.cost(bits))
        if costes and abs(min(costes) + mejor) < 1e-6:
            exitos += 1
    return exitos / repeticiones, validos / (repeticiones * shots)


def main() -> None:
    datos = ASIA.sample(N_MUESTRAS, seed=1)
    score = LocalScore(datos, kind="bic")
    espacio = OrderingSpace(score, max_parents=2)
    print(espacio.summary())

    arcos_opt, score_opt = exact_best_dag(score, 2, cache=espacio.parent_sets)
    energia_opt = -score_opt
    print(f"\nóptimo exacto: E0 = {energia_opt:.4f}  "
          f"SHD = {confusion(arcos_opt, ASIA.arcs)['shd']}")

    print("\nreferencias clásicas:")
    clasicos = {}
    for nombre, fn in (("hill climbing", hill_climbing), ("tabú", tabu_search)):
        arcos, valor = fn(score, max_parents=2)
        c = confusion(arcos, ASIA.arcs)
        clasicos[nombre] = {"gap": -valor - energia_opt, "shd": c["shd"]}
        print(f"  {nombre:14s} gap={-valor - energia_opt:8.3f}  SHD={c['shd']}")

    print("\ncontrol de muestreo uniforme (mismo presupuesto de disparos):")
    t = time.time()
    tasa, frac = control_azar(espacio, SHOTS)
    print(f"  éxito={100*tasa:.1f}%   fracción factible={100*frac:.4f}%   "
          f"[{time.time()-t:.0f}s]")

    print(f"\n{'ansatz':20s} {'prof':>4} {'alpha':>6} {'sem':>4} {'gap':>10} "
          f"{'SHD':>4} {'fact.fin':>9} {'evals':>6} {'min':>6}")
    print("-" * 78)

    registros = []
    for ansatz, depth, x0_base in (("qaoa", 1, PUNTO_CALIENTE),
                                   ("hardware_efficient", 2, None)):
        for alpha in (0.25, 1.0):
            for semilla in SEMILLAS:
                vqa = OrderingVQA(espacio, ansatz=ansatz, depth=depth,
                                  alpha=alpha, shots=SHOTS, maxiter=MAXITER,
                                  seed=semilla)
                if x0_base is not None:
                    rng = np.random.default_rng(semilla)
                    x0 = x0_base + rng.normal(0, 0.15, size=len(x0_base))
                else:
                    x0 = None
                inicio = time.time()
                res = vqa.run(x0)
                minutos = (time.time() - inicio) / 60
                gap = res.best_cost - energia_opt
                shd = confusion(res.best_arcs, ASIA.arcs)["shd"] if res.feasible else None
                print(f"{ansatz:20s} {depth:>4} {alpha:>6} {semilla:>4} "
                      f"{gap:>10.4f} {str(shd):>4} "
                      f"{100*res.feasible_fraction_final:>8.2f}% "
                      f"{res.n_evaluations:>6} {minutos:>6.1f}", flush=True)
                registros.append({
                    "ansatz": ansatz, "profundidad": depth, "alpha": alpha,
                    "semilla": semilla, "gap": gap, "shd": shd,
                    "factible_inicial": res.feasible_fraction_initial,
                    "factible_final": res.feasible_fraction_final,
                    "evaluaciones": res.n_evaluations, "minutos": minutos,
                    "orden": list(res.best_order) if res.best_order else None,
                    "arcos": res.best_arcs,
                })
                salida = (Path(__file__).resolve().parents[1] / "resultados"
                          / "exp06_asia.json")
                salida.write_text(json.dumps({
                    "red": "asia", "n_muestras": N_MUESTRAS,
                    "qubits": espacio.num_vars,
                    "qubits_ogorman": 3 * espacio.n * (espacio.n - 1) // 2,
                    "shots": SHOTS, "maxiter": MAXITER,
                    "punto_caliente": PUNTO_CALIENTE.tolist(),
                    "energia_optima": energia_opt, "arcos_optimos": arcos_opt,
                    "shd_optimo": confusion(arcos_opt, ASIA.arcs)["shd"],
                    "clasicos": clasicos,
                    "control_azar": {"exito": tasa, "fraccion_factible": frac},
                    "resultados": registros,
                }, indent=2, ensure_ascii=False))

    print(f"\nresultados en {salida}")


if __name__ == "__main__":
    main()
