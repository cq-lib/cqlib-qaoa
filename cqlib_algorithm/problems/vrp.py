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

"""VRP problem container."""

from dataclasses import dataclass
from typing import Literal
import numpy as np

Edge = tuple[int, int]


@dataclass
class VRP:
    """Capacitated Vehicle Routing Problem (VRP) instance (single depot).

    Attributes:
        n: Total number of nodes including the depot (index 0).
        distance: Symmetric (n, n) distance matrix with zeros on the diagonal.
            If ``None`` or empty, it will be generated according to the parameters
            below.
        demand: Length-n vector of customer demands; ``demand[0]`` must be 0.
            If ``None`` or empty, defaults to 1 for customers (1..n-1) and 0 for depot.
        vehicle_count: Number of vehicles K.
        capacity: Capacity per vehicle (Optional).
        gen_method: Distance generation method: ``"euclidean"`` (from random
            coordinates) or ``"random_symmetric"`` (random symmetric weights).
        coord_range: Coordinate range for Euclidean generation.
        dim: Dimensionality for Euclidean coordinates.
        as_int: If True, round distances to integers (off-diagonals clamped to ≥ 1).
        round_digits: When ``as_int=False``, decimals to round to.
        weight_range: Range for random-symmetric distances.
        seed: RNG seed for reproducibility.
        dist_limit: Clip distances into ``[low, high]`` after generation.
    """

    n: int
    distance: np.ndarray | None = None
    demand: np.ndarray | None = None
    vehicle_count: int = 1
    capacity: int = 0

    # ---- random-instance parameters (used when distance is None) ----
    gen_method: Literal["euclidean", "random_symmetric"] = "euclidean"
    coord_range: tuple[float, float] = (0.0, 100.0)
    dim: int = 2
    as_int: bool = True
    round_digits: int | None = None
    weight_range: tuple[float, float] = (1.0, 100.0)
    seed: int | None = None
    dist_limit: tuple[float, float] = (0.0, 100.0)

    def __post_init__(self):
        """Generate/validate distance and demand arrays.

        Raises:
            AssertionError: If shapes/symmetry/zeros-on-diagonal are invalid or
                demand[0] is not zero.
        """
        if self.distance is None or np.size(self.distance) == 0:
            self.distance = self._random_distance(
                self.n,
                method=self.gen_method,
                coord_range=self.coord_range,
                dim=self.dim,
                as_int=self.as_int,
                round_digits=self.round_digits,
                weight_range=self.weight_range,
                seed=self.seed,
                dist_limit=self.dist_limit,
            )
        else:
            self.distance = np.asarray(self.distance, dtype=float)

        if self.demand is None or np.size(self.demand) == 0:
            self.demand = np.zeros(self.n, dtype=float)
            if self.n > 1:
                self.demand[1:] = 1.0
        else:
            self.demand = np.asarray(self.demand, dtype=float)

        assert self.distance.shape == (self.n, self.n)
        assert np.allclose(self.distance, self.distance.T)
        assert np.all(np.diag(self.distance) == 0)
        assert self.demand.shape == (self.n,)
        assert abs(self.demand[0]) < 1e-12

    @staticmethod
    def _random_distance(
        n: int,
        *,
        method: Literal["euclidean", "random_symmetric"] = "euclidean",
        coord_range: tuple[float, float] = (0.0, 100.0),
        dim: int = 2,
        as_int: bool = True,
        round_digits: int | None = None,
        weight_range: tuple[float, float] = (1.0, 100.0),
        seed: int | None = None,
        dist_limit: tuple[float, float] = (0.0, 100.0),
    ) -> np.ndarray:
        """Generate a symmetric distance matrix with zeros on the diagonal.

        Args:
            n: Number of nodes.
            method: ``"euclidean"`` uses random coordinates and Euclidean distances;
                ``"random_symmetric"`` samples a symmetric matrix directly.
            coord_range: Range for coordinates in Euclidean mode.
            dim: Dimensionality for Euclidean coordinates.
            as_int: If True, round distances to integers; off-diagonal zeros are set to 1.
            round_digits: When ``as_int=False``, number of decimals to round to.
            weight_range: Range for random-symmetric weights.
            seed: RNG seed.
            dist_limit: Clip distances into ``[low, high]`` after generation.

        Returns:
            np.ndarray: Symmetric (n, n) distance matrix.

        Raises:
            ValueError: If ``method`` is unsupported.
        """
        rng = np.random.default_rng(seed)

        if method == "euclidean":
            low, high = coord_range
            coords = rng.uniform(low, high, size=(n, dim))
            diff = coords[:, None, :] - coords[None, :, :]
            D = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(D, 0.0)
        elif method == "random_symmetric":
            a, b = weight_range
            U = rng.uniform(a, b, size=(n, n))
            U = np.triu(U, k=1)
            D = U + U.T
            np.fill_diagonal(D, 0.0)
        else:
            raise ValueError('method must be "euclidean" or "random_symmetric"')

        lo, hi = dist_limit
        D = np.clip(D, lo, hi)

        if as_int:
            D = np.rint(D).astype(float)
            i, j = np.where((~np.eye(n, dtype=bool)) & (D == 0.0))
            if i.size > 0:
                D[i, j] = 1.0
                D[j, i] = 1.0
        else:
            if round_digits is not None:
                D = np.round(D, round_digits)

        return D
