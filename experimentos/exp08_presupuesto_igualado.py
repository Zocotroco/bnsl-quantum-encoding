"""Fase 3 · Experimento 8: ¿aporta el VQA algo sobre el muestreo aleatorio?

La pregunta que decide si este trabajo es un estudio de viabilidad o algo más.
Hasta ahora las comparaciones con el azar eran injustas en un sentido u otro:
o el azar muestreaba cadenas de bits (y perdía por no saber construir órdenes
válidos), o no se igualaba el presupuesto de muestras.

Diseño
------
El VQA consume `disparos x evaluaciones` muestras en total, de las que una
fracción resulta ser un orden válido. Se compara contra dos líneas base con
**exactamente ese mismo presupuesto**:

* **azar sobre cadenas de bits**: lo mismo que hace el circuito, sin aprender.
  Mide cuánto aporta el circuito a la hora de *construir* órdenes válidos.
* **azar sobre órdenes válidos**: la línea base fuerte, y la que emplean
  Romero et al. (2004). Se le dan tantas muestras como órdenes válidos produjo
  el VQA, de modo que ninguno de los dos tiene ventaja de presupuesto.

Para la segunda no hace falta simular: se enumera la distribución completa de
puntuaciones sobre los n! órdenes y se calcula **analíticamente** la
probabilidad de que k muestras al azar igualen o superen un gap dado.

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp08_presupuesto_igualado.py [red]
"""

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag
from bnslq.data import NETWORKS
from bnslq.factorial import FactorialOrderingSpace
from bnslq.metrics import confusion
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

SEMILLAS = (0, 1, 2, 3, 4)
MAXITER = 40
DISPAROS = 1024   # se puede fijar por línea de órdenes

# El presupuesto lo decide todo. Con 25 000 muestras válidas el azar acierta el
# 83 % y la comparación no discrimina; con ~2 500 acierta el 16 %, y ahí sí se
# ve si el algoritmo variacional aporta algo. Por eso se barre el número de
# disparos: es la variable que fija el presupuesto total.


def distribucion_gaps(espacio, optimo):
    """Gap de **todos** los n! órdenes válidos. Es la referencia analítica."""
    n_ordenes = math.factorial(espacio.n)
    vals = np.fromiter(
        (espacio.parent_sets.score_of_order(o)
         for o in itertools.permutations(espacio.variables)),
        float, n_ordenes)
    return np.sort(optimo - vals)  # gaps, ascendente (0 = óptimo)


def prob_azar_alcanza(gaps_ordenados, gap_objetivo, k):
    """P(al menos una de k muestras uniformes de órdenes logre <= gap_objetivo)."""
    f = float((gaps_ordenados <= gap_objetivo + 1e-9).mean())
    return 1.0 - (1.0 - f) ** k if f < 1 else 1.0


def gap_esperado_azar(gaps_ordenados, k):
    """Mejor gap MEDIANO de k muestras uniformes de órdenes válidos.

    No hace falta simular: si los gaps están ordenados de forma ascendente, el
    mejor de k muestras es menor o igual que `gaps[i]` con probabilidad
    1 - (1 - (i+1)/n)^k. La mediana es el menor `gaps[i]` cuya probabilidad
    alcanza 1/2, es decir el cuantil 1 - 2^(-1/k).
    """
    n = len(gaps_ordenados)
    cuantil = 1.0 - 0.5 ** (1.0 / k)          # fracción necesaria para P = 1/2
    i = min(n - 1, int(np.ceil(cuantil * n)) - 1)
    return float(gaps_ordenados[max(i, 0)])


def main() -> None:
    global DISPAROS
    red = NETWORKS[sys.argv[1] if len(sys.argv) > 1 else "capas10"]
    if len(sys.argv) > 2:
        DISPAROS = int(sys.argv[2])
    n_muestras = 20000 if red.n >= 8 else 2000
    score = LocalScore(red.sample(n_muestras, seed=1), kind="bic")
    espacio = FactorialOrderingSpace(score, max_parents=2)
    arcos_opt, optimo = exact_best_dag(score, 2, cache=espacio.parent_sets)

    print(espacio.summary())
    print(f"\nóptimo exacto: SHD = {confusion(arcos_opt, red.arcs)['shd']}")

    t = time.time()
    gaps = distribucion_gaps(espacio, optimo)
    frac_opt = float((gaps <= 1e-9).mean())
    print(f"distribución sobre {len(gaps):,} órdenes calculada en {time.time()-t:.0f}s")
    print(f"  órdenes óptimos: {100*frac_opt:.4f}%   gap mediano: {np.median(gaps):.2f}\n")

    print(f"{'p':>2} {'alpha':>6} {'sem':>4} {'muestras':>9} {'válidas':>8} "
          f"{'gap VQA':>9} {'gap azar':>9} {'P(azar<=)':>10} {'min':>5}")
    print("-" * 78)

    registros = []
    for depth in (1, 2):
        for alpha in (0.25, 1.0):
            for semilla in SEMILLAS:
                vqa = OrderingVQA(espacio, ansatz="qaoa", depth=depth, alpha=alpha,
                                  shots=DISPAROS, maxiter=MAXITER, seed=semilla)
                t0 = time.time()
                res = vqa.run(np.tile([1.0, 0.8], depth)
                              + np.random.default_rng(semilla).normal(0, 0.15, 2 * depth))
                minutos = (time.time() - t0) / 60

                total = DISPAROS * res.n_evaluations
                validas = int(total * res.feasible_fraction_final)
                # `optimo` es la puntuación (negativa) y `best_cost` el coste (= -puntuación),
                # de modo que el gap es su SUMA, no su diferencia.
                gap_vqa = res.best_cost + optimo if res.feasible else None

                # líneas base con el MISMO presupuesto
                p_azar = (prob_azar_alcanza(gaps, gap_vqa, validas)
                          if gap_vqa is not None else None)
                g_azar = gap_esperado_azar(gaps, validas)

                print(f"{depth:>2} {alpha:>6} {semilla:>4} {total:>9,} {validas:>8,} "
                      f"{gap_vqa:>9.4f} {g_azar:>9.4f} {100*p_azar:>9.1f}% {minutos:>5.1f}")
                registros.append({
                    "profundidad": depth, "alpha": alpha, "semilla": semilla,
                    "muestras_totales": total, "muestras_validas": validas,
                    "gap_vqa": gap_vqa, "gap_esperado_azar": g_azar,
                    "prob_azar_iguala": p_azar,
                    "fraccion_factible": res.feasible_fraction_final,
                    "evaluaciones": res.n_evaluations, "minutos": minutos,
                })

    ganadas = sum(1 for r in registros
                  if r["gap_vqa"] is not None and r["gap_vqa"] < r["gap_esperado_azar"])
    print(f"\ntiradas en las que el VQA mejora el gap típico del azar: "
          f"{ganadas} de {len(registros)}")

    salida = (Path(__file__).resolve().parents[1] / "resultados"
              / f"exp08_presupuesto_{red.name}_{DISPAROS}.json")
    salida.write_text(json.dumps({
        "red": red.name, "qubits": espacio.num_vars, "disparos": DISPAROS,
        "maxiter": MAXITER, "fraccion_ordenes_optimos": frac_opt,
        "gap_mediano_azar": float(np.median(gaps)),
        "victorias_vqa": ganadas, "tiradas": len(registros),
        "resultados": registros,
    }, indent=2, ensure_ascii=False))
    print(f"resultados en {salida}")


if __name__ == "__main__":
    main()
