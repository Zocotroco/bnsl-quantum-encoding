"""Fase 1 · Experimento 2: QAOA sobre el benchmark Cancer (n = 5, 30 qubits).

Es la réplica de Soloviev, Bielza y Larrañaga (2022, tabla 1): tres tamaños de
muestra, dos puntuaciones, SHD frente a la estructura verdadera y comparación
con hill climbing y búsqueda tabú.

ATENCIÓN AL COSTE.  Con 30 qubits el vector de estado ocupa 17 GB, de modo que
hay que simular con *matrix product states*.  Medido en este equipo: unos 51 s
por evaluación con p=1, es decir ~85 min por ejecución de 100 evaluaciones.  Con
p=2 el coste se dispara.  Lánzalo en segundo plano y con paciencia:

    nohup ./.venv/bin/python -u -W ignore experimentos/exp02_cancer.py \
        > resultados/exp02.log 2>&1 &
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag, hill_climbing, tabu_search
from bnslq.data import CANCER
from bnslq.metrics import confusion
from bnslq.qaoa import QAOARunner
from bnslq.qubo import BNSLHamiltonian
from bnslq.scores import LocalScore

TAMANOS = (500, 1000, 10000)
PUNTUACIONES = ("bic", "bdeu")
P = 1
ALPHA = 0.25
SHOTS = 512
MAXITER = 60


def main() -> None:
    registros = []
    for n_muestras in TAMANOS:
        datos = CANCER.sample(n_muestras, seed=1)
        for tipo in PUNTUACIONES:
            score = LocalScore(datos, kind=tipo, ess=10)
            H = BNSLHamiltonian(score, max_parents=2)
            arcos_opt, score_opt = exact_best_dag(score, max_parents=2)
            fila = {"n_muestras": n_muestras, "puntuacion": tipo,
                    "shd_optimo_exacto": confusion(arcos_opt, CANCER.arcs)["shd"]}

            for nombre, fn in (("hc", hill_climbing), ("tabu", tabu_search)):
                arcos, _ = fn(score, max_parents=2)
                fila[f"shd_{nombre}"] = confusion(arcos, CANCER.arcs)["shd"]

            inicio = time.time()
            res = QAOARunner(H, p=P, alpha=ALPHA, shots=SHOTS,
                             maxiter=MAXITER, seed=0).run()
            fila.update(
                shd_qaoa=confusion(res.best_arcs, CANCER.arcs)["shd"],
                gap_qaoa=res.best_cost + score_opt,
                factible=res.feasible,
                evaluaciones=res.n_evaluations,
                segundos=time.time() - inicio,
                arcos_qaoa=res.best_arcs,
            )
            print(json.dumps(fila, ensure_ascii=False), flush=True)
            registros.append(fila)

    salida = Path(__file__).resolve().parents[1] / "resultados" / "exp02_cancer.json"
    salida.write_text(json.dumps(
        {"p": P, "alpha": ALPHA, "shots": SHOTS, "maxiter": MAXITER,
         "arcos_verdaderos": CANCER.arcs, "resultados": registros},
        indent=2, ensure_ascii=False))
    print(f"resultados en {salida}")


if __name__ == "__main__":
    main()
