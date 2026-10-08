"""Fase 1 · Experimento 1: barrido de p y alpha del QAOA sobre la red toy4.

Pregunta: ¿cuántas capas y qué CVaR hacen falta para que el QAOA alcance el
óptimo global, que aquí conocemos exactamente por enumeración?

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp01_barrido_toy4.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag, hill_climbing, tabu_search
from bnslq.data import TOY4
from bnslq.metrics import confusion
from bnslq.qaoa import QAOARunner
from bnslq.qubo import BNSLHamiltonian
from bnslq.scores import LocalScore

SHOTS = 1024
MAXITER = 150
SEMILLAS = (0, 1, 2)


def main() -> None:
    datos = TOY4.sample(2000, seed=1)
    score = LocalScore(datos, kind="bic")
    H = BNSLHamiltonian(score, max_parents=2)

    arcos_opt, score_opt = exact_best_dag(score, max_parents=2)
    energia_opt = -score_opt
    print(H.summary())
    print(f"\nóptimo exacto: E0 = {energia_opt:.4f}  arcos = {arcos_opt}")
    print(f"simulador: {QAOARunner(H, p=1).backend.options.method}\n")

    for nombre, fn in (("hill climbing", hill_climbing), ("tabú", tabu_search)):
        arcos, valor = fn(score, max_parents=2)
        c = confusion(arcos, TOY4.arcs)
        print(f"  {nombre:14s} E={-valor:10.4f}  gap={-valor - energia_opt:8.4f}  SHD={c['shd']}")

    print(f"\n{'p':>2} {'alpha':>6} {'E_mejor':>11} {'gap':>9} {'SHD':>4} {'fact':>5} "
          f"{'evals':>6} {'seg':>6}")
    print("-" * 60)

    registros = []
    for p in (1, 2, 3, 4):
        for alpha in (0.1, 0.25, 0.5, 1.0):
            mejores = []
            for semilla in SEMILLAS:
                runner = QAOARunner(H, p=p, alpha=alpha, shots=SHOTS,
                                    maxiter=MAXITER, seed=semilla)
                inicio = time.time()
                res = runner.run()
                mejores.append((res, time.time() - inicio))
            res, seg = min(mejores, key=lambda t: t[0].best_cost)
            c = confusion(res.best_arcs, TOY4.arcs)
            gap = res.best_cost - energia_opt
            print(f"{p:>2} {alpha:>6} {res.best_cost:>11.4f} {gap:>9.4f} "
                  f"{c['shd']:>4} {str(res.feasible):>5} {res.n_evaluations:>6} {seg:>6.1f}")
            registros.append({
                "p": p, "alpha": alpha, "energia": res.best_cost, "gap": gap,
                "shd": c["shd"], "factible": res.feasible,
                "evaluaciones": res.n_evaluations, "segundos": seg,
                "arcos": res.best_arcs,
                "stats_circuito": QAOARunner(H, p=p).circuit_stats(),
            })

    salida = Path(__file__).resolve().parents[1] / "resultados" / "exp01_toy4.json"
    salida.write_text(json.dumps({
        "red": "toy4", "n_muestras": 2000, "puntuacion": "bic", "max_padres": 2,
        "shots": SHOTS, "maxiter": MAXITER, "semillas": list(SEMILLAS),
        "energia_optima": energia_opt, "arcos_optimos": arcos_opt,
        "resultados": registros,
    }, indent=2, ensure_ascii=False))
    print(f"\nresultados en {salida}")


if __name__ == "__main__":
    main()
