"""Fase 2 · Experimento 4: resiliencia al ruido de las dos codificaciones.

Compara, sobre el mismo problema y con el mismo ruido, la formulación de
O'Gorman (matriz de adyacencia, 18 qubits en toy4) y la del espacio de órdenes
(6 qubits).  Es la única comparación cabeza a cabeza posible: en Cancer la
formulación de O'Gorman necesita 30 qubits y no es simulable en este equipo.

Por qué esta es la segunda versión
----------------------------------
La primera tirada quedó inservible (resultados en `exp04_ruido_SESGADO.json`).
Se tomaron los parámetros de una única semilla y, para la codificación de
adyacencia, resultaron ser malos: gap 3,46 y **probabilidad del óptimo 0 % ya
sin ruido**.  Si el estado preparado no contiene la respuesta, el ruido no tiene
nada que degradar y la comparación no mide nada.

Corrección: para cada codificación se buscan parámetros entre varias semillas y
se elige el que **maximiza la probabilidad del óptimo** en la distribución de
salida, no el mejor disparo suelto.  Además se reporta la degradación relativa
al propio omega=0 de cada codificación, que es lo comparable entre las dos.

Diseño
------
Se congelan esos parámetros y sólo después se enciende el ruido, para aislar la
degradación del estado preparado del efecto del ruido sobre el optimizador.

Métricas, ambas comparables entre codificaciones y acompañadas de su línea base
de azar (imprescindible: los subespacios factibles tienen densidades muy
distintas):

* **fracción factible**: disparos que decodifican una solución válida.
* **probabilidad del óptimo**: disparos que dan exactamente la solución óptima.

Uso:  ./.venv/bin/python -u -W ignore experimentos/exp04_ruido_comparativa.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag
from bnslq.data import TOY4
from bnslq.orderings import OrderingSpace
from bnslq.qaoa import QAOARunner
from bnslq.qubo import BNSLHamiltonian
from bnslq.ruido import CANALES, gate_counts, noisy_simulator
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

OMEGAS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SHOTS = 256
SEMILLAS_RUIDO = (0, 1, 2)
SEMILLAS_BUSQUEDA = tuple(range(8))
PROFUNDIDAD = 2


def elegir_parametros(nombre, hacer_runner, es_optimo, n_qubits):
    """Busca en varias semillas los parámetros que mejor preparan el óptimo.

    Criterio: máxima probabilidad del óptimo en la distribución final; se
    desempata por menor coste.  Si ninguna semilla consigue probabilidad no
    nula, se avisa, porque entonces la comparación bajo ruido no es informativa.
    """
    mejor = None
    for semilla in SEMILLAS_BUSQUEDA:
        runner = hacer_runner(semilla)
        res = runner.run()
        counts = runner.sample(res.optimal_params)
        total = sum(counts.values())
        p_opt = sum(n for b, n in counts.items() if es_optimo(b)) / total
        clave = (p_opt, -res.best_cost)
        if mejor is None or clave > mejor[0]:
            mejor = (clave, res.optimal_params, res.best_cost, p_opt)
    _, params, coste, p_opt = mejor
    print(f"  {nombre:12s} coste={coste:.4f}  P(óptimo)={100*p_opt:.1f}%")
    if p_opt == 0.0:
        print(f"    AVISO: {nombre} nunca prepara el óptimo ni sin ruido; "
              f"su curva bajo ruido no será informativa.")
    return params


def main() -> None:
    datos = TOY4.sample(2000, seed=1)
    score = LocalScore(datos, kind="bic")
    arcos_opt, score_opt = exact_best_dag(score, max_parents=2)
    print(f"toy4: óptimo exacto = {arcos_opt}\n")

    H = BNSLHamiltonian(score, max_parents=2)
    O = OrderingSpace(score, max_parents=2)

    # Líneas base de azar: sin ellas las fracciones factibles no son comparables.
    base_ord = 24 / 2**O.num_vars          # 24 órdenes válidos de 2^6
    base_adj = 443 / 2**12                 # 443 DAGs factibles de 2^12 de arcos
    print(f"líneas base de azar --- órdenes: {100*base_ord:.1f}%   "
          f"adyacencia: {100*base_adj:.1f}%\n")

    print(f"buscando parámetros sobre {len(SEMILLAS_BUSQUEDA)} semillas:")
    opt_ord = lambda b: (O.decode_order(b[::-1]) is not None
                         and O.dag_of(b[::-1])[0] == arcos_opt)
    opt_adj = lambda b: H.is_feasible(b[::-1]) and H.decode(b[::-1]) == arcos_opt

    p_ord = elegir_parametros(
        "órdenes",
        lambda s: OrderingVQA(O, ansatz="qaoa", depth=PROFUNDIDAD, alpha=0.25,
                              shots=512, maxiter=150, seed=s),
        opt_ord, O.num_vars)
    p_adj = elegir_parametros(
        "adyacencia",
        lambda s: QAOARunner(H, p=PROFUNDIDAD, alpha=0.25, shots=512,
                             maxiter=150, seed=s),
        opt_adj, H.num_vars)

    print(f"\npuertas --- órdenes: {gate_counts(OrderingVQA(O, ansatz='qaoa', depth=PROFUNDIDAD).circuit)}"
          f"  adyacencia: {gate_counts(QAOARunner(H, p=PROFUNDIDAD).circuit)}\n")

    def medir(nombre, params, hacer_runner, factible, es_optimo, n_qubits, base):
        filas = []
        for canal in CANALES:
            for omega in OMEGAS:
                frac, opt = [], []
                for semilla in SEMILLAS_RUIDO:
                    sim = noisy_simulator(canal, omega, n_qubits, seed=semilla)
                    counts = hacer_runner(sim, semilla).sample(params)
                    total = sum(counts.values())
                    frac.append(sum(n for b, n in counts.items() if factible(b)) / total)
                    opt.append(sum(n for b, n in counts.items() if es_optimo(b)) / total)
                    if omega == 0.0:
                        break
                filas.append({
                    "codificacion": nombre, "canal": canal, "omega": omega,
                    "fraccion_factible": float(np.mean(frac)),
                    "prob_optimo": float(np.mean(opt)),
                    "factible_sobre_azar": float(np.mean(frac)) / base,
                })
                print(f"  {nombre:11s} {canal:18s} ω={omega:<7g} "
                      f"factible={100*np.mean(frac):5.1f}% ({np.mean(frac)/base:5.1f}x azar)  "
                      f"óptimo={100*np.mean(opt):5.2f}%", flush=True)
        return filas

    print("midiendo degradación (los parámetros ya no cambian):\n")
    inicio = time.time()
    registros = medir(
        "órdenes", p_ord,
        lambda sim, s: OrderingVQA(O, ansatz="qaoa", depth=PROFUNDIDAD,
                                   shots=SHOTS, seed=s, simulator=sim),
        lambda b: O.is_total_order(b[::-1]), opt_ord, O.num_vars, base_ord)
    print()
    registros += medir(
        "adyacencia", p_adj,
        lambda sim, s: QAOARunner(H, p=PROFUNDIDAD, shots=SHOTS, seed=s,
                                  simulator=sim),
        lambda b: H.is_feasible(b[::-1]), opt_adj, H.num_vars, base_adj)

    # Degradación relativa al propio omega = 0: es lo comparable entre codificaciones.
    print("\ndegradación relativa al propio ω=0:")
    for nombre in ("órdenes", "adyacencia"):
        for canal in CANALES:
            serie = [r for r in registros
                     if r["codificacion"] == nombre and r["canal"] == canal]
            base0 = serie[0]["fraccion_factible"]
            for r in serie:
                r["factible_relativa"] = (r["fraccion_factible"] / base0) if base0 else None
            resumen = "  ".join(
                f"ω={r['omega']:g}:{100*r['factible_relativa']:.0f}%"
                if r["factible_relativa"] is not None else "n/d" for r in serie)
            print(f"  {nombre:11s} {canal:18s} {resumen}")

    salida = Path(__file__).resolve().parents[1] / "resultados" / "exp04_ruido.json"
    salida.write_text(json.dumps({
        "red": "toy4", "profundidad": PROFUNDIDAD, "shots": SHOTS,
        "semillas_ruido": list(SEMILLAS_RUIDO),
        "semillas_busqueda": list(SEMILLAS_BUSQUEDA),
        "omegas": list(OMEGAS),
        "qubits_ordenes": O.num_vars, "qubits_adyacencia": H.num_vars,
        "base_azar_ordenes": base_ord, "base_azar_adyacencia": base_adj,
        "arcos_optimos": arcos_opt, "resultados": registros,
    }, indent=2, ensure_ascii=False))
    print(f"\ncompletado en {time.time()-inicio:.0f} s; resultados en {salida}")


if __name__ == "__main__":
    main()
