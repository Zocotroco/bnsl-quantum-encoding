"""Fase 3 · Experimento 7: ejecución en hardware cuántico real de IBM.

Presupuesto: **10 minutos de QPU** (plan Open). Se gasta con cuidado.

Estrategia
----------
Optimizar los parámetros **en simulación** (gratis) y ejecutar en hardware sólo
el circuito final, con muchos disparos. Es el procedimiento habitual cuando el
tiempo de QPU es escaso, y hay que declararlo: no se está optimizando sobre el
hardware, se está verificando que el estado preparado sobrevive al ruido real.

Qué se mide, y por qué estos tres casos
---------------------------------------
1. `asia_factorial`  --- 17 qubits, 28 puertas de dos cúbits tras transpilar.
2. `capas10_factorial` --- 25 qubits, 69 puertas. Es el benchmark difícil: el
   muestreo aleatorio de órdenes sólo lo resuelve el 7 % de las veces.
3. `asia_torneo` --- 28 qubits, **998 puertas**. Se espera que falle, y esa
   falla es justamente el resultado: mide cuánto importa la codificación cuando
   el circuito tiene que ejecutarse de verdad.

Uso
---
    # 1. Guardar la credencial UNA VEZ (el canal nuevo exige también el CRN):
    ./.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from bnslq.ibm import guardar_cuenta; \
        guardar_cuenta(token='TU_API_KEY', instance='TU_CRN')"

    # 1b. Comprobar que conecta, sin gastar cuota:
    ./.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from bnslq.ibm import probar_conexion; probar_conexion()"

    # 2. Ensayo completo sin gastar QPU (simulador con topología real):
    ./.venv/bin/python -u -W ignore experimentos/exp07_hardware_ibm.py --simulacro

    # 3. Ejecución real:
    ./.venv/bin/python -u -W ignore experimentos/exp07_hardware_ibm.py --real
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit import transpile

from bnslq.classical import exact_best_dag
from bnslq.data import ASIA, CAPAS10
from bnslq.factorial import FactorialOrderingSpace
from bnslq.metrics import confusion
from bnslq.orderings import OrderingSpace
from bnslq.scores import LocalScore
from bnslq.vqa import OrderingVQA

DISPAROS = {"asia_factorial": 4096, "capas10_factorial": 4096, "asia_torneo": 2048}

# Presupuesto de simulación por caso. La codificación de torneo cuesta ~60 s por
# evaluación con 28 qubits, así que optimizarla aquí costaría horas: para ese
# caso se usa directamente el punto (gamma, beta) = (1,0 ; 0,8) que el sondeo
# previo identificó como el que concentra amplitud sobre el subespacio factible.
# No condiciona la conclusión, porque de ese circuito se espera que falle en
# hardware por profundidad, no por una mala elección de parámetros.
PUNTO_CALIENTE = np.array([1.0, 0.8])
PRESUPUESTO = {
    "asia_factorial":    {"semillas": 4, "maxiter": 60},
    "capas10_factorial": {"semillas": 3, "maxiter": 40},
    "asia_torneo":       {"semillas": 0, "maxiter": 0},   # sin optimizar
}


CACHE = Path(__file__).resolve().parents[1] / "resultados" / "exp07_parametros.json"


def preparar():
    """Construye los tres casos y fija sus parámetros en simulación.

    Los parámetros se cachean en disco: la optimización tarda varios minutos y
    no debe repetirse si una ejecución contra hardware falla a mitad.
    """
    guardados = {}
    if CACHE.exists():
        guardados = json.loads(CACHE.read_text())
        print(f"  (reutilizando parámetros de {CACHE.name})")
    casos = {}
    for nombre, red, n_muestras, clase in (
        ("asia_factorial", ASIA, 50000, FactorialOrderingSpace),
        ("capas10_factorial", CAPAS10, 20000, FactorialOrderingSpace),
        ("asia_torneo", ASIA, 50000, OrderingSpace),
    ):
        score = LocalScore(red.sample(n_muestras, seed=1), kind="bic")
        espacio = clase(score, max_parents=2)
        arcos_opt, score_opt = exact_best_dag(score, 2, cache=espacio.parent_sets)
        plan = PRESUPUESTO[nombre]

        if nombre in guardados:
            g = guardados[nombre]
            params = np.array(g["params"])
            factible, gap = g["sim_factible"], g["sim_gap"]
            print(f"  {nombre:20s} en caché: factible="
                  f"{'n/d' if factible is None else f'{100*factible:.1f}%'}", flush=True)
        elif plan["semillas"] == 0:
            params, factible, gap = PUNTO_CALIENTE, None, None
            print(f"  {nombre:20s} sin optimizar (punto del sondeo)", flush=True)
        else:
            mejor = None
            for semilla in range(plan["semillas"]):
                vqa = OrderingVQA(espacio, ansatz="qaoa", depth=1, alpha=0.25,
                                  shots=1024, maxiter=plan["maxiter"], seed=semilla)
                x0 = PUNTO_CALIENTE + np.random.default_rng(semilla).normal(0, 0.15, 2)
                res = vqa.run(x0)
                clave = (res.feasible_fraction_final, -res.best_cost)
                if mejor is None or clave > mejor[0]:
                    mejor = (clave, res)
            res = mejor[1]
            params = res.optimal_params
            factible = res.feasible_fraction_final
            gap = res.best_cost + score_opt
            print(f"  {nombre:20s} sim: factible={100*factible:5.1f}%  gap={gap:.4f}",
                  flush=True)

        casos[nombre] = {
            "red": red, "espacio": espacio, "params": params,
            "energia_optima": -score_opt, "arcos_optimos": arcos_opt,
            "sim_factible": factible, "sim_gap": gap,
        }
        guardados[nombre] = {"params": list(map(float, params)),
                             "sim_factible": factible, "sim_gap": gap}
        CACHE.write_text(json.dumps(guardados, indent=2))
    return casos


def analizar(caso, counts, nombre):
    """Métricas comparables entre casos y con la simulación."""
    espacio, red = caso["espacio"], caso["red"]
    total = sum(counts.values())
    validos = coste_mejor = None
    n_validos = 0
    mejor = float("inf")
    mejores_arcos = []
    for cadena, n in counts.items():
        bits = cadena[::-1]
        if espacio.is_total_order(bits):
            n_validos += n
            c = espacio.cost(bits)
            if c < mejor:
                mejor, mejores_arcos = c, espacio.dag_of(bits)[0]
    gap = mejor - caso["energia_optima"] if n_validos else None
    return {
        "disparos": total,
        "fraccion_factible": n_validos / total,
        "gap": gap,
        "shd": confusion(mejores_arcos, red.arcs)["shd"] if n_validos else None,
        "arcos": mejores_arcos,
    }


def ejecutar(isa, backend, disparos, real):
    """Envía un circuito y devuelve (cuentas, segundos de QPU).

    Usa `SamplerV2` **en ambos modos**. El simulacro sólo cambia el backend por
    uno falso, de modo que el código que se ejercita ensayando es exactamente el
    que se ejecutará contra el hardware. Tenerlos separados costó un fallo por
    un import ausente que el simulacro no podía detectar.
    """
    from qiskit_ibm_runtime import SamplerV2

    trabajo = SamplerV2(mode=backend).run([isa], shots=disparos)
    if real:
        print(f"    trabajo {trabajo.job_id()} enviado...", flush=True)
    resultado = trabajo.result()
    datos = resultado[0].data
    counts = getattr(datos, list(datos.keys())[0]).get_counts()
    if not real:
        return counts, None
    from bnslq.ibm import resumen_cuota
    return counts, resumen_cuota(trabajo)


def replicar(caso, backend, copias, disparos, real):
    """Ejecuta varias copias del mismo circuito en regiones disjuntas del chip.

    Con un solo circuito de 17 qubits el transpilador escoge los **mejores** 17
    del procesador. Al replicarlo se ocupan también zonas peores, de modo que la
    dispersión entre copias mide cuánto depende el resultado de qué qubits
    físicos toquen --- un efecto NISQ real que rara vez se reporta.

    Como todas las copias son idénticas y se ejecutan a la vez, la duración del
    trabajo es la de una sola: se multiplican los disparos efectivos sin coste
    adicional de QPU. Lo que sí cambia es la fidelidad, por diafonía entre
    regiones vecinas; comparar con la tirada aislada la cuantifica.
    """
    from qiskit import QuantumCircuit

    vqa = OrderingVQA(caso["espacio"], ansatz="qaoa", depth=1, shots=1)
    base = vqa.circuit.assign_parameters(
        dict(zip(list(vqa.params), list(caso["params"]))))
    ancho = base.num_qubits

    if ancho * copias > backend.num_qubits:
        copias = backend.num_qubits // ancho
        print(f"    (reducido a {copias} copias por tamaño del chip)")

    combinado = QuantumCircuit(ancho * copias, ancho * copias)
    for c in range(copias):
        combinado.compose(base, qubits=range(c * ancho, (c + 1) * ancho),
                          clbits=range(c * ancho, (c + 1) * ancho), inplace=True)
    isa = transpile(combinado, backend, optimization_level=3, seed_transpiler=0)
    print(f"    {copias} copias: {ancho*copias} qubits lógicos, "
          f"profundidad {isa.depth()}")

    counts, uso = ejecutar(isa, backend, disparos, real)

    # Desagregar por copia: cada una ocupa su propio tramo de bits clásicos.
    espacio = caso["espacio"]
    por_copia = []
    for c in range(copias):
        parcial = {}
        for cadena, n in counts.items():
            bits = cadena.replace(" ", "")[::-1][c * ancho:(c + 1) * ancho]
            parcial[bits] = parcial.get(bits, 0) + n
        total = sum(parcial.values())
        n_val = sum(n for b, n in parcial.items() if espacio.is_total_order(b))
        mejor = min((espacio.cost(b) for b in parcial if espacio.is_total_order(b)),
                    default=None)
        por_copia.append({
            "copia": c,
            "fraccion_factible": n_val / total,
            "gap": None if mejor is None else mejor - caso["energia_optima"],
        })
        print(f"      copia {c}: factible {100*n_val/total:5.2f}%   "
              f"gap {'n/d' if mejor is None else round(mejor - caso['energia_optima'], 4)}")
    return {"copias": copias, "qpu_segundos": uso, "por_copia": por_copia}


def main() -> None:
    real = "--real" in sys.argv
    print("=== preparación en simulación (sin coste de QPU) ===")
    casos = preparar()

    if real:
        from bnslq.ibm import elegir_backend
        backend = elegir_backend(min_qubits=28)
        print(f"\n=== HARDWARE REAL: {backend.name} ({backend.num_qubits} qubits) ===")
    else:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
        print(f"\n=== SIMULACRO sobre topología de {backend.name} (sin coste) ===")

    registros = {}
    qpu_total = 0.0
    for nombre, caso in casos.items():
        vqa = OrderingVQA(caso["espacio"], ansatz="qaoa", depth=1, shots=1)
        circuito = vqa.circuit.assign_parameters(
            dict(zip(list(vqa.params), list(caso["params"]))))
        isa = transpile(circuito, backend, optimization_level=3, seed_transpiler=0)
        ops = isa.count_ops()
        cx = sum(ops.get(g, 0) for g in ("cz", "ecr", "cx"))
        print(f"\n  {nombre}: {isa.num_qubits} qubits usados, "
              f"profundidad {isa.depth()}, {cx} puertas de dos cúbits")

        t0 = time.time()
        counts, uso = ejecutar(isa, backend, DISPAROS[nombre], real)
        if uso is not None:
            qpu_total += uso
            print(f"    QPU consumido: {uso:.1f} s  (acumulado {qpu_total:.1f} s de 600)")

        m = analizar(caso, counts, nombre)
        m.update(qubits=isa.num_qubits, profundidad=isa.depth(), puertas_2q=cx,
                 segundos=time.time() - t0,
                 sim_factible=caso["sim_factible"], sim_gap=caso["sim_gap"])
        registros[nombre] = m
        ideal = ("n/d" if caso["sim_factible"] is None
                 else f"{100*caso['sim_factible']:.1f}%")
        print(f"    factible {100*m['fraccion_factible']:.2f}% (ideal: {ideal})   "
              f"gap {m['gap'] if m['gap'] is None else round(m['gap'],4)}   SHD {m['shd']}")

    sufijo = "real" if real else "simulacro"
    salida = Path(__file__).resolve().parents[1] / "resultados" / f"exp07_hardware_{sufijo}.json"

    def guardar(extra=None):
        salida.write_text(json.dumps(
            {"backend": backend.name, "disparos": DISPAROS,
             "qpu_segundos": qpu_total, "resultados": registros,
             "replicacion": extra},
            indent=2, ensure_ascii=False, default=str))

    # El resultado principal se guarda ANTES de intentar nada más: la réplica es
    # un añadido y no debe poder comprometerlo.
    guardar()
    print(f"\nresultado principal guardado en {salida}")

    print("\n=== réplica en regiones disjuntas (asia_factorial) ===")
    try:
        extra = replicar(casos["asia_factorial"], backend, copias=4,
                         disparos=DISPAROS["asia_factorial"], real=real)
        qpu_total += extra.get("qpu_segundos") or 0.0
        guardar(extra)
        print(f"resultados completos en {salida}")
    except Exception as e:
        print(f"la réplica falló ({type(e).__name__}: {e}); "
              f"el resultado principal queda intacto en {salida}")


if __name__ == "__main__":
    main()
