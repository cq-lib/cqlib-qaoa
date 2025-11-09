# This code is part of cqlib.
#
# Copyright (C) 2025 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MaxCut problem container."""

from dataclasses import dataclass
import numpy as np

Edge = tuple[int, int]


@dataclass
class MaxCut:
    """Undirected weighted MaxCut instance.

    Attributes:
        n: Number of vertices (0..n-1).
        weights: Edge weights mapping with keys as ordered tuples ``(u, v)``
            where ``u < v``. If omitted, a random graph is generated according
            to the generation parameters below.
        edge_prob: Probability of including each potential edge during random
            generation.
        weight_range: Range ``[low, high]`` for sampled edge weights.
        as_int: Whether to round weights to integers (0 rounded to 1).
        seed: RNG seed for reproducibility (used only for random generation).
        no_isolated: If True, ensures no isolated vertices by adding edges.
    """

    n: int
    weights: dict[Edge, float] | None = None

    # ---- random-instance parameters (used when weights is None) ----
    edge_prob: float = 0.5
    weight_range: tuple[float, float] = (1.0, 1.0)
    as_int: bool = True
    seed: int | None = None
    no_isolated: bool = True

    def __post_init__(self):
        """Normalize/initialize the edge weights after dataclass construction.

        - If ``weights`` is None, generate a random instance using the configured
          parameters.
        - If ``weights`` is provided, coerce keys to ordered pairs and values to float.
        """
        if not self.weights:
            self.weights = self._random_weights(
                self.n,
                edge_prob=self.edge_prob,
                weight_range=self.weight_range,
                as_int=self.as_int,
                seed=self.seed,
                no_isolated=self.no_isolated,
            )
        else:
            self.weights = {
                (min(u, v), max(u, v)): float(w) for (u, v), w in self.weights.items()
            }

    @staticmethod
    def _random_weights(
        n: int,
        *,
        edge_prob: float = 0.5,
        weight_range: tuple[float, float] = (1.0, 1.0),
        as_int: bool = True,
        seed: int | None = None,
        no_isolated: bool = True,
    ) -> dict[Edge, float]:
        """Generate a random undirected weighted graph for MaxCut.

        Args:
            n: Number of vertices.
            edge_prob: Probability of including an edge between any pair (i < j).
            weight_range: Inclusive range to sample weights from (uniform).
            as_int: If True, round sampled weights to integers (0 becomes 1).
            seed: RNG seed for reproducibility.
            no_isolated: If True, add edges to remove isolated vertices.

        Returns:
            dict[Edge, float]: Mapping of edges ``(u, v)`` with ``u < v`` to weights.

        Raises:
            ValueError: If ``weight_range`` upper bound is not positive.
        """
        if n < 2:
            return {}
        rng = np.random.default_rng(seed)
        low, high = weight_range
        if high <= 0:
            raise ValueError("weight_range upper bound must be > 0")
        weights: dict[Edge, float] = {}

        # 1) Sample edges independently
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_prob:
                    w = rng.uniform(low, high)
                    if as_int:
                        w = int(round(w))
                        if w == 0:
                            w = 1
                    weights[(i, j)] = float(w)

        if no_isolated:
            # 2) Ensure no isolated vertices (degree >= 1)
            deg = [0] * n
            for u, v in weights.keys():
                deg[u] += 1
                deg[v] += 1

            for u in range(n):
                if deg[u] == 0:
                    candidates = [v for v in range(n) if v != u]
                    v = rng.choice(candidates)
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) not in weights:
                        w = rng.uniform(low, high)
                        if as_int:
                            w = int(round(w))
                            if w == 0:
                                w = 1
                        weights[(a, b)] = float(w)
                        deg[u] += 1
                        deg[v] += 1
        return weights
