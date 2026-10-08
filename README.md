# Aprendizaje estructural cuántico de redes bayesianas

Código, experimentos y resultados que respaldan el artículo *«Aprendizaje
estructural cuántico de redes bayesianas: reducción de la codificación y
ausencia de ventaja de búsqueda»*.

El trabajo tiene dos partes. La primera reduce la codificación del problema de
`3n(n−1)/2` qubits a `O(n log n)` —de 84 a 17 para ocho variables—, lo que
permite ejecutarlo en un procesador cuántico de puertas real. La segunda usa esa
reducción para medir, en condiciones favorables al método cuántico y con el
presupuesto de muestras igualado, si el algoritmo variacional busca mejor que el
azar. **No lo hace.**

## Resultado principal

Sobre `capas10` (10 nodos), comparando contra muestreo uniforme de *órdenes
válidos* con el mismo número de muestras:

| Disparos | Muestras válidas | Gap mediano VQA | Gap mediano azar | VQA mejor | p |
|---|---|---|---|---|---|
| 32 | 250–1 240 | 9,09 | 9,09 | 12 de 20 | 0,252 |
| 64 | 780–2 360 | 9,13 | 4,84 | 7 de 20 | 0,942 |
| 1 024 | 14 100–34 900 | 0,00 | 0,00 | 0 de 20 | — |

La última fila no es un fracaso: el VQA alcanza el óptimo exacto en 17 de 20
tiradas. Es que el azar también lo alcanza (mediana 87 %). A ese presupuesto la
instancia está **saturada** y la comparación deja de discriminar.

## Por qué los benchmarks al uso no sirven

Fracción de los `n!` órdenes que alcanzan la puntuación óptima:

| Red | n | % órdenes óptimos | Éxito del azar (1 024 muestras) |
|---|---|---|---|
| Cancer | 5 | 50,0 % / 3,3 % | **100 %** |
| Survey | 6 | 66,7 % / 0,6 % | **100 %** |
| Asia | 8 | 1,15 % | **100 %** |
| capas10 | 10 | **0,007 %** | **7 %** |

La fracción de órdenes óptimos depende del tamaño muestral (segundo valor: con
5 000 muestras o más). Lo que no depende de él es la última columna: en las tres
primeras redes el muestreo de 1 024 órdenes válidos al azar encuentra el óptimo
**siempre**, con cualquier tamaño entre 1 000 y 50 000. No hay búsqueda que
hacer. El criterio que gobierna la dificultad es el número de **extensiones
lineales** del grafo generador.

## Coste en qubits

| n | O'Gorman | Torneo | Factorial | Subespacio válido (torneo → factorial) |
|---|---|---|---|---|
| 5 | 30 | 10 | **8** | 11,7 % → 46,9 % |
| 6 | 45 | 15 | **11** | 2,2 % → 35,2 % |
| 8 | 84 | 28 | **17** | 0,015 % → **30,8 %** |
| 10 | 135 | 45 | **25** | 3·10⁻⁵ % → 10,8 % |

En hardware real (`ibm_marrakesh`, 11,4 s de cuota): la codificación factorial
da un 75,4 % de soluciones válidas y encuentra el óptimo; la de torneo, con
1 001 puertas de dos cúbits, **no produce ni una sola solución válida**.

## Módulos

| Módulo | Papel |
|---|---|
| `scores.py` | puntuaciones locales BIC / BDeu / K2 (verificadas contra pgmpy) |
| `pseudoboolean.py` | polinomios pseudo-booleanos, paso a Ising, indicadores por Möbius |
| `qubo.py` | Hamiltoniano de O'Gorman: H_score, H_max, H_trans, H_consist y cotas de penalización |
| `qubo_pleno.py` | variante que codifica también la selección de padres |
| `orderings.py` | espacio de órdenes por torneo, `n(n−1)/2` qubits |
| `factorial.py` | **codificación factorial**, `O(n log n)` qubits |
| `parentsets.py` | poda exacta de conjuntos candidatos de padres (de Campos & Ji) |
| `classical.py` | óptimo exacto, ascenso de colinas, búsqueda tabú |
| `qaoa.py` | circuito QAOA sobre la codificación de adyacencia (fase 1) |
| `vqa.py` | bucle variacional, ansätze y CVaR (fase 2) |
| `representacion.py` | circuito de la red bayesiana aprendida |
| `inferencia.py` | inferencia por estimación cuántica de amplitud |
| `ruido.py` | modelos de ruido NISQ |
| `ibm.py` | conexión a IBM Quantum (**nunca almacena credenciales**) |
| `metrics.py` | SHD y matriz de confusión de arcos |
| `data.py` | redes `toy4`, `cancer`, `survey`, `asia`, `capas10` |

## Experimentos

Cada tabla del artículo procede de un guión de `experimentos/`, y su salida
queda en `resultados/` como JSON junto al `.log` de la ejecución.

| Guión | Qué mide |
|---|---|
| `exp01_barrido_toy4.py` | verificación del Hamiltoniano sobre 443 grafos factibles |
| `exp03_ordenes.py` | espacio de órdenes frente a espacio de grafos |
| `exp04_ruido_comparativa.py` | degradación bajo ruido NISQ |
| `exp06_asia.py` | comparación de ansätze a 8 nodos |
| `exp07_hardware_ibm.py` | ejecución en `ibm_marrakesh` |
| `exp08_presupuesto_igualado.py` | **resultado principal**: presupuesto igualado |
| `exp09_ciclo_completo.py` | ciclo completo e inferencia por QAE |
| `exp10_tres_ansatze.py` | las tres formas de repartir el problema |
| `exp11_dificultad.py` | extensiones lineales y tablas de codificación |

> `resultados/exp08_presupuesto_capas10_OBSOLETO_signo.json` se conserva a
> propósito: procede de una versión del guión con un error de signo que
> inflaba los gaps. Se publica para que la corrección quede documentada, y no
> debe usarse.

## Instalación y uso

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest tests -q          # 93 pruebas
./.venv/bin/python -u -W ignore experimentos/exp11_dificultad.py
```

Para ejecutar en hardware de IBM hace falta una cuenta en IBM Quantum. Las
credenciales se guardan **fuera del repositorio**, en `~/.qiskit/qiskit-ibm.json`:

```bash
./.venv/bin/python guardar_credencial.py      # pide la clave sin mostrarla
```

El token no aparece nunca en el código, en los resultados ni en los registros.

## Citación

```bibtex
@misc{medrano2026bnslq,
  author = {Medrano Millán, Raúl},
  title  = {Aprendizaje estructural cuántico de redes bayesianas:
            reducción de la codificación y ausencia de ventaja de búsqueda},
  year   = {2026}
}
```

## Licencia

MIT. Véase [LICENSE](LICENSE).
