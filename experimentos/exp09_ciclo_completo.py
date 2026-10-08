"""Fase 3 · Experimento 9: el ciclo completo, de los datos a la inferencia.

Cierra el objetivo que motivaba el trabajo de partida: aprender la estructura
con el algoritmo variacional, ajustar los parámetros, construir el circuito que
representa la red aprendida y responder consultas sobre él con estimación
cuántica de amplitud.

Dos mediciones:

1. **El ciclo funciona de extremo a extremo.** Se comprueba que la inferencia
   sobre el circuito de la red *aprendida* coincide con la inferencia exacta
   sobre esa misma red.

2. **Cuándo compensa la QAE.** La ventaja cuadrática sobre el muestreo es
   asintótica; la pregunta práctica es a partir de qué precisión se nota. Se
   mide el coste de ambos métodos --- en aplicaciones del operador de
   preparación frente a disparos --- al mismo nivel de confianza.

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp09_ciclo_completo.py
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
from bnslq.inferencia import InferenciaCuantica
from bnslq.metrics import confusion
from bnslq.representacion import RedCuantica, ajustar_cpts
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

Z95 = 1.959964


def main() -> None:
    red = CANCER
    datos = red.sample(20000, seed=1)
    variables = list(red.variables)

    # ---- paso 1: aprender la estructura con el algoritmo variacional -----
    score = LocalScore(datos, kind="bic")
    espacio = FactorialOrderingSpace(score, max_parents=2)
    arcos_opt, score_opt = exact_best_dag(score, 2, cache=espacio.parent_sets)

    vqa = OrderingVQA(espacio, ansatz="qaoa", depth=1, alpha=0.25,
                      shots=1024, maxiter=60, seed=0)
    res = vqa.run(np.array([1.0, 0.8]))
    arcos = res.best_arcs
    print("PASO 1 — aprendizaje de la estructura")
    print(f"  qubits: {espacio.num_vars}   gap: {res.best_cost + score_opt:.4f}   "
          f"SHD: {confusion(arcos, red.arcs)['shd']}")
    print(f"  arcos aprendidos: {arcos}")

    # ---- paso 2: ajustar las CPTs de la estructura aprendida -------------
    enteros = datos.astype(int)
    cpts = ajustar_cpts(enteros, arcos, variables)
    print("\nPASO 2 — ajuste de parámetros a partir de los datos (Laplace)")

    # ---- paso 3: construir el circuito de la red aprendida ---------------
    rc = RedCuantica(arcs=arcos, cpts=cpts, variables=variables)
    print(f"\nPASO 3 — circuito de la red aprendida: {rc.estadisticas()}")

    # ---- paso 4: inferencia sobre ese circuito ---------------------------
    inf = InferenciaCuantica(rc)
    consultas = [
        ({"Cancer": 1}, {"Smoker": 1, "Pollution": 1}),
        ({"Cancer": 1}, {"Xray": 1}),
        ({"Dyspnoea": 1}, {"Smoker": 1}),
        ({"Xray": 1}, {"Smoker": 1, "Dyspnoea": 1}),
    ]
    print("\nPASO 4 — inferencia sobre el circuito")
    print(f"  {'consulta':<42} {'exacta':>9} {'QAE':>9} {'error':>9}")
    filas = []
    for consulta, evidencia in consultas:
        ex = inf.exacta(consulta, evidencia)
        q, coste = inf.condicional_qae(consulta, evidencia, epsilon=0.005)
        etiqueta = (f"P({list(consulta)[0]}=1 | "
                    + ", ".join(f"{v}={b}" for v, b in evidencia.items()) + ")")
        print(f"  {etiqueta:<42} {ex:>9.5f} {q:>9.5f} {abs(q-ex):>9.5f}")
        filas.append({"consulta": etiqueta, "exacta": ex, "qae": q,
                      "error": abs(q - ex), "coste": coste})

    # ---- paso 5: ¿cuándo compensa la QAE? --------------------------------
    print("\nPASO 5 — coste de QAE frente a muestreo, a 95 % de confianza")
    cond = {"Smoker": 1, "Pollution": 1}
    idx = rc.indice
    p = sum(pp for x, pp in rc.distribucion_exacta().items()
            if all(x[idx[v]] == b for v, b in cond.items()))
    print(f"  estimando P{cond} = {p:.5f}")
    print(f"  {'epsilon':>9} {'QAE (aplic. de A)':>19} {'muestreo (disparos)':>21} {'razón':>8}")
    coste_filas = []
    for eps in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005):
        t0 = time.time()
        q, c_qae = inf.probabilidad_qae(cond, epsilon=eps, disparos=100)
        c_mu = int(np.ceil(Z95**2 * p * (1 - p) / eps**2))
        razon = c_mu / c_qae
        marca = "  <- gana QAE" if razon > 1 else ""
        print(f"  {eps:>9} {c_qae:>19,} {c_mu:>21,} {razon:>7.2f}x{marca}"
              f"   [{time.time()-t0:.0f}s]", flush=True)
        coste_filas.append({"epsilon": eps, "coste_qae": c_qae,
                            "coste_muestreo": c_mu, "razon": razon,
                            "error_qae": abs(q - p)})

    salida = Path(__file__).resolve().parents[1] / "resultados" / "exp09_ciclo.json"
    salida.write_text(json.dumps({
        "red": red.name, "qubits_aprendizaje": espacio.num_vars,
        "arcos_aprendidos": arcos, "arcos_verdaderos": red.arcs,
        "shd": confusion(arcos, red.arcs)["shd"],
        "gap": res.best_cost + score_opt,
        "circuito_representacion": rc.estadisticas(),
        "consultas": filas, "coste": coste_filas,
    }, indent=2, ensure_ascii=False))
    print(f"\nresultados en {salida}")


if __name__ == "__main__":
    main()
