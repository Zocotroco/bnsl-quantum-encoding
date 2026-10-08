"""Formulación del BNSL como polinomio pseudo-booleano (O'Gorman et al., 2015).

Este módulo es la pieza que faltaba en la Parte II del trabajo.  Allí el
Hamiltoniano se escribía como

    H_coste = sum_G -Score(G) |G><G|,

que es una definición correcta pero **no compilable**: para sintetizar
exp(-i·gamma·H_C) hace falta H_C escrito como polinomio en las variables de
qubit.  Aquí se construye ese polinomio.

Tres ideas, y por qué cada una es necesaria
-------------------------------------------
1. *Inversión de Möbius.*  Como la puntuación es descomponible, cada nodo aporta
   s_i(Pa_i).  Para convertir "el valor que toma una función sobre conjuntos" en
   "un polinomio sobre las aristas" se usan los coeficientes

       w_i(J) = sum_{K subconjunto de J} (-1)^{|J|-|K|} s_i(K),

   que cumplen  sum_{J subconjunto de Pa}  w_i(J) = s_i(Pa).  El grado del
   polinomio resultante es el in-degree máximo permitido m.

2. *Aciclicidad.*  "G no tiene ciclos" no es una función local de las aristas, y
   por tanto no se puede escribir en Paulis.  La solución de O'Gorman es
   introducir variables auxiliares de **orden topológico** r_ij (i<j) y exigir
   (a) que r sea transitivo — H_trans — y (b) que las aristas respeten ese orden
   — H_consist.  Un grafo cuyas aristas son consistentes con un orden total es
   acíclico por construcción.

3. *Signo.*  Los Hamiltonianos se minimizan y las puntuaciones se maximizan, de
   modo que se usa s_i(J) = -score_i(J).  La energía del estado fundamental es,
   por tanto, menos la puntuación del DAG óptimo.

Coste en qubits (variables binarias)
------------------------------------
    n(n-1)      aristas (matriz de adyacencia sin diagonal)
    n(n-1)/2    orden topológico
    2n          holgura del in-degree máximo (opcional; ver `include_hmax`)

es decir 3n(n-1)/2 siguiendo a Soloviev et al. (2022), que impone el in-degree
de forma clásica, o n(n-1)+n(n-1)/2+2n con la formulación completa de O'Gorman.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from .pseudoboolean import PseudoBoolean
from .scores import LocalScore

Arc = Tuple[str, str]


class BNSLHamiltonian:
    """Construye el polinomio pseudo-booleano del BNSL para unos datos dados.

    Parameters
    ----------
    score
        Puntuación local descomponible (a maximizar).
    max_parents
        In-degree máximo m.  Fija el grado del polinomio de puntuación.
    delta_trans, delta_consist
        Pesos de las penalizaciones.  Si se omiten se usa una cota suficiente
        (ver `suggested_penalty`).
    include_hmax
        Si es ``True`` se añade el término H_max de O'Gorman con 2n variables de
        holgura, que impone el in-degree dentro del propio Hamiltoniano.  Si es
        ``False`` (por defecto, como en Soloviev et al.) el in-degree se
        comprueba clásicamente al evaluar cada solución medida.
    """

    def __init__(
        self,
        score: LocalScore,
        max_parents: int = 2,
        delta_trans: Optional[float] = None,
        delta_consist: Optional[float] = None,
        include_hmax: bool = False,
        delta_max: Optional[float] = None,
    ):
        self.score = score
        self.variables: Tuple[str, ...] = tuple(score.variables)
        self.n = len(self.variables)
        if self.n < 2:
            raise ValueError("hacen falta al menos dos variables")
        self.max_parents = min(max_parents, self.n - 1)
        self.include_hmax = include_hmax

        self._build_index()
        cotas = self.suggested_penalties()
        self.delta_trans = float(delta_trans if delta_trans is not None else cotas["trans"])
        self.delta_consist = float(
            delta_consist if delta_consist is not None else cotas["consist"]
        )
        self.delta_max = float(delta_max if delta_max is not None else cotas["max"])

        self.poly = PseudoBoolean(self.num_vars)
        self._add_hscore()
        self._add_htrans()
        self._add_hconsist()
        if include_hmax:
            self._add_hmax()
        self.poly.prune()

    # ------------------------------------------------------------- índices

    def _build_index(self) -> None:
        self.a_index: Dict[Tuple[int, int], int] = {}
        self.r_index: Dict[Tuple[int, int], int] = {}
        self.y_index: Dict[Tuple[int, int], int] = {}
        k = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.a_index[(i, j)] = k  # arco X_i -> X_j
                    k += 1
        self.n_a = k
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.r_index[(i, j)] = k  # X_i precede a X_j
                k += 1
        self.n_r = k - self.n_a
        if self.include_hmax:
            for i in range(self.n):
                for b in range(2):
                    self.y_index[(i, b)] = k
                    k += 1
        self.n_y = k - self.n_a - self.n_r
        self.num_vars = k

    # ------------------------------------------------------------ términos

    def _s(self, node_idx: int, parents: Sequence[int]) -> float:
        """s_i(J) = -score_i(J): puntuación con el signo invertido."""
        node = self.variables[node_idx]
        return -self.score(node, [self.variables[p] for p in parents])

    def mobius_coefficients(self, node_idx: int) -> Dict[Tuple[int, ...], float]:
        """w_i(J) para todo J con |J| <= m.  Expuesto para poder inspeccionarlo."""
        others = [j for j in range(self.n) if j != node_idx]
        out: Dict[Tuple[int, ...], float] = {}
        for size in range(self.max_parents + 1):
            for J in itertools.combinations(others, size):
                w = 0.0
                for r in range(len(J) + 1):
                    for K in itertools.combinations(J, r):
                        w += (-1.0) ** (len(J) - len(K)) * self._s(node_idx, K)
                out[J] = w
        return out

    def _add_hscore(self) -> None:
        for i in range(self.n):
            for J, w in self.mobius_coefficients(i).items():
                # el arco va de cada padre j hacia i
                self.poly.add([self.a_index[(j, i)] for j in J], w)

    def _add_htrans(self) -> None:
        d = self.delta_trans
        for i, j, k in itertools.combinations(range(self.n), 3):
            rij, rjk, rik = (
                self.r_index[(i, j)],
                self.r_index[(j, k)],
                self.r_index[(i, k)],
            )
            # d * (r_ik + r_ij r_jk - r_ij r_ik - r_jk r_ik)
            self.poly.add([rik], d)
            self.poly.add([rij, rjk], d)
            self.poly.add([rij, rik], -d)
            self.poly.add([rjk, rik], -d)

    def _add_hconsist(self) -> None:
        d = self.delta_consist
        for i in range(self.n):
            for j in range(i + 1, self.n):
                aij, aji, rij = (
                    self.a_index[(i, j)],
                    self.a_index[(j, i)],
                    self.r_index[(i, j)],
                )
                # d * (a_ji r_ij + a_ij (1 - r_ij))
                self.poly.add([aji, rij], d)
                self.poly.add([aij], d)
                self.poly.add([aij, rij], -d)

    def _add_hmax(self) -> None:
        """H_max = delta * sum_i (m - in_degree_i - y_i)^2 con y_i de 2 bits."""
        d, m = self.delta_max, self.max_parents
        for i in range(self.n):
            # términos lineales de la expresión interior, como (coef, variable)
            inner: List[Tuple[float, Optional[int]]] = [(float(m), None)]
            inner += [(-1.0, self.a_index[(j, i)]) for j in range(self.n) if j != i]
            inner += [(-(2.0**b), self.y_index[(i, b)]) for b in range(2)]
            for (c1, v1), (c2, v2) in itertools.product(inner, repeat=2):
                vs = [v for v in (v1, v2) if v is not None]
                self.poly.add(set(vs), d * c1 * c2)

    # ----------------------------------------------------------- utilidades

    def delta(self, j: int, i: int) -> float:
        """Delta_ji: mayor aumento de puntuación que puede aportar el arco j -> i.

        Es la cantidad auxiliar sobre la que O'Gorman et al. (2015) construyen
        todas sus cotas de penalización.  Para m <= 2 se calcula **exactamente**
        (su ec. 37); para m >= 3 el máximo interior es a su vez un problema de
        optimización intratable y hay que conformarse con la cota superior de su
        ec. 38, que es lo que se hace aquí.
        """
        w = self.mobius_coefficients(i)
        if self.max_parents <= 1:
            valor = -w.get((j,), 0.0)
        elif self.max_parents == 2:
            valor = -w.get((j,), 0.0) - sum(
                min(0.0, w.get(tuple(sorted((j, k))), 0.0))
                for k in range(self.n)
                if k not in (i, j)
            )
        else:  # cota superior, ec. 38
            valor = -sum(min(0.0, v) for J, v in w.items() if j in J)
        return max(0.0, valor)

    def suggested_penalties(self) -> Dict[str, float]:
        """Cotas de O'Gorman et al. (2015), con la simplificación de Zardini et al.

        Mucho más ajustadas que acotar por el recorrido total de la puntuación.
        Importa: penalizaciones excesivas comprimen el espectro de energía y,
        como señala el propio O'Gorman, **levantan barreras** entre estados
        válidos separados por estados inválidos, creando cuencas de óptimos
        locales.  Todas se fijan a la cota más uno.
        """
        deltas = {
            (j, i): self.delta(j, i)
            for i in range(self.n)
            for j in range(self.n)
            if j != i
        }
        mayor = max(deltas.values(), default=0.0)
        d_trans = mayor + 1.0
        # delta_consist debe superar a la vez (n-2)*delta_trans (para la
        # consistencia del orden) y max Delta (para eliminar los 2-ciclos).
        d_consist = max((self.n - 2) * d_trans, mayor) + 1.0
        return {"max": mayor + 1.0, "trans": d_trans, "consist": d_consist}

    def suggested_penalty(self) -> float:
        """Cota única y conservadora: el recorrido total de H_score.

        Se conserva como referencia y para comparar con `suggested_penalties()`,
        que es la que se usa por defecto.
        """
        total = 0.0
        for i in range(self.n):
            others = [j for j in range(self.n) if j != i]
            values = [
                self._s(i, J)
                for size in range(self.max_parents + 1)
                for J in itertools.combinations(others, size)
            ]
            total += max(values) - min(values)
        return total + 1.0

    def decode(self, bits: Sequence[int] | str) -> List[Arc]:
        """Traduce una asignación binaria a la lista de arcos correspondiente."""
        x = [int(b) for b in bits]
        return sorted(
            (self.variables[i], self.variables[j])
            for (i, j), k in self.a_index.items()
            if x[k]
        )

    def decode_order(self, bits: Sequence[int] | str) -> Dict[Tuple[str, str], int]:
        x = [int(b) for b in bits]
        return {
            (self.variables[i], self.variables[j]): x[k]
            for (i, j), k in self.r_index.items()
        }

    def encode(self, arcs: Sequence[Arc], order: Optional[Sequence[str]] = None) -> List[int]:
        """Construye la asignación binaria de un DAG (y opcionalmente su orden).

        Si no se da `order` se usa un orden topológico cualquiera del grafo.
        """
        import networkx as nx

        idx = {v: i for i, v in enumerate(self.variables)}
        x = [0] * self.num_vars
        for src, dst in arcs:
            x[self.a_index[(idx[src], idx[dst])]] = 1
        if order is None:
            g = nx.DiGraph()
            g.add_nodes_from(self.variables)
            g.add_edges_from(arcs)
            order = list(nx.topological_sort(g))
        position = {v: p for p, v in enumerate(order)}
        for (i, j), k in self.r_index.items():
            x[k] = int(position[self.variables[i]] < position[self.variables[j]])
        if self.include_hmax:
            for i in range(self.n):
                deg = sum(1 for (p, c) in arcs if idx[c] == i)
                slack = self.max_parents - deg
                if slack < 0 or slack > 3:
                    raise ValueError(f"in-degree {deg} fuera de rango para m={self.max_parents}")
                for b in range(2):
                    x[self.y_index[(i, b)]] = (slack >> b) & 1
        return x

    def in_degrees(self, bits: Sequence[int] | str) -> List[int]:
        x = [int(b) for b in bits]
        return [
            sum(x[self.a_index[(j, i)]] for j in range(self.n) if j != i)
            for i in range(self.n)
        ]

    def indegree_violation(self, bits: Sequence[int] | str) -> float:
        """Penalización clásica de in-degree: delta * sum_i max(0, d_i - m)^2.

        Es el valor al que se reduce H_max de O'Gorman una vez minimizado sobre
        las variables de holgura y, por tanto, el término que Soloviev et al.
        aplican "de forma clásica" fuera del circuito.
        """
        m = self.max_parents
        return self.delta_max * sum(
            max(0, d - m) ** 2 for d in self.in_degrees(bits)
        )

    def classical_cost(self, bits: Sequence[int] | str) -> float:
        """Coste que ve el optimizador clásico: H(x) más el in-degree clásico.

        Con `include_hmax=True` el segundo término ya está dentro de `poly` y
        este método coincide con `__call__`.
        """
        if self.include_hmax:
            return self(bits)
        return self(bits) + self.indegree_violation(bits)

    def brute_force_minimum(self, classical: bool = True):
        """Mínimo global del coste sobre todas las asignaciones (n pequeño)."""
        import numpy as np

        if self.num_vars > 24:
            raise ValueError(f"{self.num_vars} variables son demasiadas para enumerar")
        X = ((np.arange(1 << self.num_vars)[:, None] >> np.arange(self.num_vars)) & 1)
        X = X.astype(np.int8)
        values = self.poly.evaluate_many(X)
        if classical and not self.include_hmax:
            m = self.max_parents
            degrees = np.stack([
                X[:, [self.a_index[(j, i)] for j in range(self.n) if j != i]].sum(axis=1)
                for i in range(self.n)
            ], axis=1)
            excess = np.clip(degrees - m, 0, None).astype(np.float64)
            values = values + self.delta_max * (excess**2).sum(axis=1)
        best = int(np.argmin(values))
        return "".join(str(int(b)) for b in X[best]), float(values[best])

    def is_feasible(self, bits: Sequence[int] | str) -> bool:
        """¿Es la asignación un DAG válido con in-degree <= m y orden coherente?"""
        import networkx as nx

        arcs = self.decode(bits)
        g = nx.DiGraph()
        g.add_nodes_from(self.variables)
        g.add_edges_from(arcs)
        if not nx.is_directed_acyclic_graph(g):
            return False
        if max(self.in_degrees(bits), default=0) > self.max_parents:
            return False
        return True

    def __call__(self, bits: Sequence[int] | str) -> float:
        return self.poly([int(b) for b in bits])

    def summary(self) -> str:
        return (
            f"BNSL n={self.n}  m={self.max_parents}\n"
            f"  variables: {self.n_a} arcos + {self.n_r} orden"
            + (f" + {self.n_y} holgura" if self.include_hmax else "")
            + f" = {self.num_vars}\n"
            f"  polinomio: grado {self.poly.degree}, {self.poly.num_terms} términos\n"
            f"  delta_trans={self.delta_trans:.3g}  delta_consist={self.delta_consist:.3g}"
        )
