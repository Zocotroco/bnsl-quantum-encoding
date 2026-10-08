"""Fase 3 · Experimento 11: qué hace difícil a una instancia de BNSL.

Regenera, como artefacto reproducible, las dos tablas estructurales del
artículo:

1. **Dificultad.** El número de *extensiones lineales* del grafo generador
   --- los órdenes topológicos compatibles con él --- determina si el muestreo
   aleatorio de órdenes resuelve la instancia. Se mide sobre cuatro
   estructuras de diez nodos y sobre los benchmarks habituales.

2. **Codificaciones.** Qubits y densidad del subespacio válido de las tres
   codificaciones comparadas (O'Gorman, torneo y factorial).

Uso:  python -u -W ignore experimentos/exp11_dificultad.py
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnslq.classical import exact_best_dag
from bnslq.data import NETWORKS
from bnslq.factorial import FactorialOrderingSpace
from bnslq.scores import LocalScore

SEMILLA = 0
MUESTRAS_AZAR = 1024
REPETICIONES = 1000

# Cuatro topologías de diez nodos con el mismo número de nodos y distinto
# número de extensiones lineales. `capas` es la que se adopta como CAPAS10.
ESTRUCTURAS = {
    "colisionadores": [("A", "E"), ("B", "E"), ("C", "F"), ("D", "F"),
                       ("E", "I"), ("F", "I"), ("G", "J"), ("H", "J")],
    "árbol": [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"),
              ("C", "G"), ("D", "H"), ("E", "I"), ("F", "J")],
    "cadena": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"),
               ("F", "G"), ("G", "H"), ("H", "I"), ("I", "J")],
    "capas": NETWORKS["capas10"].arcs,
}


def extensiones_lineales(nodos, arcos):
    """Cuenta los órdenes totales compatibles con el grafo por enumeración."""
    indice = {v: i for i, v in enumerate(nodos)}
    pares = [(indice[u], indice[v]) for u, v in arcos]
    n = 0
    for perm in itertools.permutations(range(len(nodos))):
        pos = [0] * len(nodos)
        for k, v in enumerate(perm):
            pos[v] = k
        if all(pos[u] < pos[v] for u, v in pares):
            n += 1
    return n


def exito_del_azar(total, compatibles, k=MUESTRAS_AZAR):
    """P(al menos una de k muestras uniformes sea un orden compatible)."""
    p = compatibles / total
    return 1.0 - (1.0 - p) ** k


def qubits_codificaciones(n):
    ogorman = n * (n - 1) + n * (n - 1) // 2          # aristas + orden
    torneo = n * (n - 1) // 2
    factorial = sum(max(1, math.ceil(math.log2(k))) for k in range(2, n + 1))
    return ogorman, torneo, factorial


def densidad(n, torneo, factorial):
    total_ordenes = math.factorial(n)
    return total_ordenes / 2 ** torneo, total_ordenes / 2 ** factorial


def main() -> None:
    salida = {}

    print("=== dificultad: extensiones lineales (10 nodos, 10! = 3 628 800) ===")
    print(f"{'estructura':>16} {'ext. lineales':>14} {'% órdenes':>11} {'éxito azar':>11}")
    filas = []
    nodos = list("ABCDEFGHIJ")
    for nombre, arcos in ESTRUCTURAS.items():
        ext = extensiones_lineales(nodos, arcos)
        total = math.factorial(10)
        pct = 100 * ext / total
        exito = exito_del_azar(total, ext)
        print(f"{nombre:>16} {ext:>14,} {pct:>10.4f}% {100*exito:>10.1f}%")
        filas.append({"estructura": nombre, "extensiones": ext,
                      "porcentaje": pct, "exito_azar_1024": exito})
    salida["dificultad_10nodos"] = filas

    print("\n=== benchmarks habituales ===")
    print("la columna 'óptimos' se mide sobre la puntuación: fracción de los n!")
    print("órdenes cuyo mejor DAG alcanza el óptimo exacto.\n")
    print(f"{'red':>10} {'n':>3} {'n!':>10} {'ext.lin.':>9} "
          f"{'óptimos':>10} {'éxito azar':>11}")
    filas = []
    for nombre in ("cancer", "survey", "asia", "capas10"):
        red = NETWORKS[nombre]
        nodos_r = sorted({v for a in red.arcs for v in a} | set(red.states))
        ext = extensiones_lineales(nodos_r, red.arcs)
        total = math.factorial(len(nodos_r))

        n_datos = 20000 if len(nodos_r) >= 8 else 2000
        score = LocalScore(red.sample(n_datos, seed=1), kind="bic")
        espacio = FactorialOrderingSpace(score, max_parents=2)
        _, optimo = exact_best_dag(score, 2, cache=espacio.parent_sets)
        buenos = sum(1 for perm in itertools.permutations(espacio.variables)
                     if espacio.parent_sets.score_of_order(perm) >= optimo - 1e-9)
        frac = buenos / total
        exito = 1.0 - (1.0 - frac) ** MUESTRAS_AZAR

        print(f"{nombre:>10} {len(nodos_r):>3} {total:>10,} {ext:>9,} "
              f"{100*frac:>9.4f}% {100*exito:>10.1f}%")
        filas.append({"red": nombre, "n": len(nodos_r), "ordenes": total,
                      "extensiones": ext, "fraccion_optimos": frac,
                      "exito_azar_1024": exito})
    salida["benchmarks"] = filas

    print("\n=== codificaciones ===")
    cab = "O'Gorman"
    print(f"{'n':>3} {cab:>9} {'torneo':>7} {'factorial':>10} "
          f"{'válidos torneo':>15} {'válidos factorial':>18}")
    filas = []
    for n in (5, 6, 8, 10):
        og, to, fa = qubits_codificaciones(n)
        dt, df = densidad(n, to, fa)
        print(f"{n:>3} {og:>9} {to:>7} {fa:>10} {100*dt:>14.4f}% {100*df:>17.1f}%")
        filas.append({"n": n, "ogorman": og, "torneo": to, "factorial": fa,
                      "validos_torneo": dt, "validos_factorial": df})
    salida["codificaciones"] = filas

    destino = Path(__file__).resolve().parents[1] / "resultados" / "exp11_dificultad.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1))
    print(f"\nresultados en {destino}")


if __name__ == "__main__":
    main()
