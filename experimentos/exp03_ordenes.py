"""Fase 2 · Experimento 3: VQA en el espacio de órdenes.

Dos preguntas:

1. ¿Recupera el VQA el óptimo global con 10 qubits, donde la Fase 1 necesitaba
   30 y no era simulable en este equipo?
2. ¿Aporta algo meter la restricción de transitividad dentro del circuito
   (ansatz `qaoa`) frente a dejársela entera al optimizador clásico (ansatz
   `hardware_efficient`)?

La métrica que responde a la segunda es la **fracción factible**: qué
proporción de los disparos cae sobre uno de los n! órdenes válidos.  El
muestreo uniforme da 120/1024 = 11.7%, que es la línea base a batir.

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp03_ordenes.py [red] [n_muestras]
      redes disponibles: cancer (10 qubits), survey (15), toy4 (6)
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag, hill_climbing, tabu_search
from bnslq.data import NETWORKS
from bnslq.metrics import confusion
from bnslq.orderings import OrderingSpace
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

SHOTS = 512
MAXITER = 150
SEMILLAS = (0, 1, 2, 3, 4)


def main() -> None:
    red = NETWORKS[sys.argv[1] if len(sys.argv) > 1 else "cancer"]
    n_muestras = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    print(f"red {red.name}: {red.n} nodos, {len(red.arcs)} arcos, "
          f"{n_muestras} muestras\n")
    datos = red.sample(n_muestras, seed=1)
    score = LocalScore(datos, kind="bic")
    espacio = OrderingSpace(score, max_parents=2)
    print(espacio.summary())

    arcos_opt, score_opt = exact_best_dag(score, max_parents=2)
    energia_opt = -score_opt
    print(f"\nóptimo exacto: E0 = {energia_opt:.4f}  SHD = "
          f"{confusion(arcos_opt, red.arcs)['shd']}  arcos = {arcos_opt}")

    for nombre, fn in (("hill climbing", hill_climbing), ("tabú", tabu_search)):
        arcos, valor = fn(score, max_parents=2)
        print(f"  {nombre:14s} gap={-valor - energia_opt:8.4f}  "
              f"SHD={confusion(arcos, red.arcs)['shd']}")

    n_ordenes = math.factorial(espacio.n)
    base = 100 * n_ordenes / 2**espacio.num_vars
    print(f"\nlínea base de fracción factible (muestreo uniforme): {base:.1f}%\n")
    print(f"{'ansatz':20s} {'prof':>4} {'alpha':>6} {'gap':>9} {'SHD':>4} "
          f"{'fact.ini':>9} {'fact.fin':>9} {'evals':>6} {'seg':>6} {'éxito':>6}")
    print("-" * 92)

    registros = []
    for ansatz in ("qaoa", "hardware_efficient"):
        for depth in (1, 2, 4):
            for alpha in (0.1, 0.25, 1.0):
                corridas = []
                for semilla in SEMILLAS:
                    vqa = OrderingVQA(espacio, ansatz=ansatz, depth=depth,
                                      alpha=alpha, shots=SHOTS,
                                      maxiter=MAXITER, seed=semilla)
                    inicio = time.time()
                    res = vqa.run()
                    corridas.append((res, time.time() - inicio))
                exitos = sum(1 for r, _ in corridas
                             if abs(r.best_cost - energia_opt) < 1e-6)
                mejor, seg = min(corridas, key=lambda t: t[0].best_cost)
                fi = sum(r.feasible_fraction_initial for r, _ in corridas) / len(corridas)
                ff = sum(r.feasible_fraction_final for r, _ in corridas) / len(corridas)
                evals = sum(r.n_evaluations for r, _ in corridas) // len(corridas)
                c = confusion(mejor.best_arcs, red.arcs)
                print(f"{ansatz:20s} {depth:>4} {alpha:>6} "
                      f"{mejor.best_cost - energia_opt:>9.4f} {c['shd']:>4} "
                      f"{100*fi:>8.1f}% {100*ff:>8.1f}% {evals:>6} {seg:>6.1f} "
                      f"{exitos}/{len(SEMILLAS):<4}")
                registros.append({
                    "ansatz": ansatz, "profundidad": depth, "alpha": alpha,
                    "gap": mejor.best_cost - energia_opt, "shd": c["shd"],
                    "fraccion_factible_inicial": fi,
                    "fraccion_factible_final": ff,
                    "evaluaciones_medias": evals, "segundos": seg,
                    "exitos": exitos, "ejecuciones": len(SEMILLAS),
                    "orden": list(mejor.best_order) if mejor.best_order else None,
                    "arcos": mejor.best_arcs,
                    "stats_circuito": OrderingVQA(
                        espacio, ansatz=ansatz, depth=depth).circuit_stats(),
                })

    salida = (Path(__file__).resolve().parents[1] / "resultados"
              / f"exp03_ordenes_{red.name}.json")
    salida.write_text(json.dumps({
        "red": red.name, "n_muestras": n_muestras, "puntuacion": "bic",
        "max_padres": 2, "qubits": espacio.num_vars,
        "qubits_ogorman": 3 * espacio.n * (espacio.n - 1) // 2,
        "shots": SHOTS, "maxiter": MAXITER, "semillas": list(SEMILLAS),
        "energia_optima": energia_opt, "arcos_optimos": arcos_opt,
        "fraccion_factible_uniforme": n_ordenes / 2**espacio.num_vars,
        "resultados": registros,
    }, indent=2, ensure_ascii=False))
    print(f"\nresultados en {salida}")


if __name__ == "__main__":
    main()
