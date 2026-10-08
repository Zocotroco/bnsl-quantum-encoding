"""Puntuaciones locales descomponibles para BNSL.

Toda puntuación descomponible cumple

    score(G, D) = sum_i score_i(Pa_i, D),

de modo que basta con saber puntuar una *familia* (un nodo y su conjunto de
padres). Esa es la propiedad que permite construir el Hamiltoniano de coste de
O'Gorman: sin descomponibilidad no hay polinomio multilineal sobre las aristas.

Convenio de signo
-----------------
Las puntuaciones aquí definidas son log-verosimilitudes penalizadas y hay que
**maximizarlas**. Los Hamiltonianos se **minimizan**. La conversión se hace en
`qubo.py` mediante s_i(J) = -score_i(J), y se explicita allí.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaln


class LocalScore:
    """Calcula y cachea puntuaciones locales sobre un conjunto de datos discreto.

    Parameters
    ----------
    data
        `DataFrame` con una columna por variable. Los valores pueden ser de
        cualquier tipo hashable; se recodifican internamente a 0..r_i-1.
    kind
        ``"bic"``, ``"bdeu"`` o ``"k2"``.
    ess
        Tamaño muestral equivalente (*equivalent sample size*) de BDeu. Se
        ignora para BIC y K2.
    """

    def __init__(self, data: pd.DataFrame, kind: str = "bic", ess: float = 1.0):
        kind = kind.lower()
        if kind not in ("bic", "bdeu", "k2"):
            raise ValueError(f"puntuación desconocida: {kind!r}")
        self.kind = kind
        self.ess = float(ess)

        self.variables: Tuple[str, ...] = tuple(data.columns)
        self.n = len(self.variables)
        self.m = len(data)
        if self.m == 0:
            raise ValueError("el conjunto de datos está vacío")

        # Recodificación a enteros 0..r_i-1 y cardinalidades.
        codes = {}
        self.cardinalities: Dict[str, int] = {}
        for v in self.variables:
            cat = pd.Categorical(data[v])
            codes[v] = np.asarray(cat.codes, dtype=np.int64)
            self.cardinalities[v] = len(cat.categories)
            if self.cardinalities[v] < 1:
                raise ValueError(f"la variable {v!r} no toma ningún valor")
        self._codes = codes
        self._cache: Dict[Tuple[str, FrozenSet[str]], float] = {}

    # ------------------------------------------------------------------ API

    def __call__(self, node: str, parents: Iterable[str] = ()) -> float:
        """Puntuación local de la familia (`node`, `parents`). Se cachea."""
        key = (node, frozenset(parents))
        hit = self._cache.get(key)
        if hit is None:
            hit = self._compute(node, sorted(key[1]))
            self._cache[key] = hit
        return hit

    def graph_score(self, arcs: Iterable[Tuple[str, str]]) -> float:
        """Puntuación total de un grafo dado como lista de arcos (padre, hijo)."""
        parents: Dict[str, list] = {v: [] for v in self.variables}
        for src, dst in arcs:
            parents[dst].append(src)
        return sum(self(v, parents[v]) for v in self.variables)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------- interno

    def _contingency(self, node: str, parents: Sequence[str]):
        """Devuelve (N_ijk, N_ij) sólo para las configuraciones *observadas*.

        Las configuraciones de padres con recuento cero no aportan nada ni a BIC
        ni a las puntuaciones bayesianas (sus términos se cancelan), así que no
        hace falta materializar las q_i columnas de la tabla.
        """
        node_codes = self._codes[node]
        r_i = self.cardinalities[node]
        if not parents:
            counts = np.bincount(node_codes, minlength=r_i).reshape(1, r_i)
            return counts.astype(np.float64), counts.sum(axis=1, dtype=np.float64)

        # Índice plano de la configuración de padres.
        flat = np.zeros(self.m, dtype=np.int64)
        for p in parents:
            flat = flat * self.cardinalities[p] + self._codes[p]
        # Compactamos a configuraciones observadas.
        _, inverse = np.unique(flat, return_inverse=True)
        n_conf = int(inverse.max()) + 1
        counts = np.zeros((n_conf, r_i), dtype=np.float64)
        np.add.at(counts, (inverse, node_codes), 1.0)
        return counts, counts.sum(axis=1)

    def _q(self, parents: Sequence[str]) -> int:
        """Número total de configuraciones de los padres (observadas o no)."""
        q = 1
        for p in parents:
            q *= self.cardinalities[p]
        return q

    def _compute(self, node: str, parents: Sequence[str]) -> float:
        if node in parents:
            raise ValueError(f"{node!r} no puede ser padre de sí mismo")
        n_ijk, n_ij = self._contingency(node, parents)
        r_i = self.cardinalities[node]
        q_i = self._q(parents)

        if self.kind == "bic":
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = n_ijk * (np.log(n_ijk) - np.log(n_ij)[:, None])
            loglik = float(np.nansum(np.where(n_ijk > 0, terms, 0.0)))
            n_params = (r_i - 1) * q_i
            return loglik - 0.5 * np.log(self.m) * n_params

        # Puntuaciones bayesianas: log P(D | G) con prior de Dirichlet.
        if self.kind == "bdeu":
            alpha_ijk = self.ess / (r_i * q_i)
            alpha_ij = self.ess / q_i
        else:  # K2: alpha_ijk = 1
            alpha_ijk = 1.0
            alpha_ij = float(r_i)

        term_j = gammaln(alpha_ij) - gammaln(alpha_ij + n_ij)
        term_k = gammaln(alpha_ijk + n_ijk) - gammaln(alpha_ijk)
        return float(term_j.sum() + term_k.sum())
