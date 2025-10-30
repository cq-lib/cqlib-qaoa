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

"""TSP problem container."""

from dataclasses import dataclass
from typing import Literal
import numpy as np

Edge = tuple[int, int]


@dataclass
class TSP:
    """Symmetric Traveling Salesman Problem (TSP) instance.

    Attributes:
        n: Number of cities.
        distance_matrix: Symmetric n×n distance matrix with zeros on the diagonal.
            If ``None`` or empty, a matrix is generated according to the parameters
            below.
        gen_method: Generation method: ``"euclidean"`` (from random coordinates,
            triangle inequality holds) or ``"random_symmetric"`` (random symmetric
            weights; triangle inequality not guaranteed).
        coord_range: Coordinate range ``[low, high]`` for Euclidean generation.
        dim: Coordinate dimensionality for Euclidean generation.
        as_int: If True, distances are rounded to integers (with off-diagonals
            clamped to at least 1).
        round_digits: When ``as_int=False``, optionally round to this many decimals.
        weight_range: Range for random symmetric weights.
        seed: RNG seed for reproducibility.
    """

    n: int
    distance_matrix: np.ndarray | None = None

    # ---- random-instance parameters (used when weights is None) ----
    gen_method: Literal["euclidean", "random_symmetric"] = "euclidean"
    coord_range: tuple[float, float] = (0.0, 100.0)
    dim: int = 2
    as_int: bool = True
    round_digits: int | None = None
    weight_range: tuple[float, float] = (1.0, 100.0)
    seed: int | None = None

    def __post_init__(self):
        """Generate or validate the distance matrix after initialization.

        Raises:
            AssertionError: If the provided/generated matrix is not n×n, not symmetric,
                or has non-zero diagonal.
        """
        if self.distance_matrix is None or np.size(self.distance_matrix) == 0:
            self.distance_matrix = self._random_tsp_distance(
                n=self.n,
                method=self.gen_method,
                coord_range=self.coord_range,
                dim=self.dim,
                as_int=self.as_int,
                round_digits=self.round_digits,
                weight_range=self.weight_range,
                seed=self.seed,
            )

        # Validate
        assert self.distance_matrix.shape == (self.n, self.n), (
            "distance_matrix must be n x n"
        )
        assert np.allclose(self.distance_matrix, self.distance_matrix.T), (
            "distance_matrix must be symmetric"
        )
        assert np.all(np.diag(self.distance_matrix) == 0), (
            "distance_matrix diagonal must be zero"
        )

    @staticmethod
    def _random_tsp_distance(
        n: int,
        *,
        method: Literal["euclidean", "random_symmetric"] = "euclidean",
        coord_range: tuple[float, float] = (0.0, 100.0),
        dim: int = 2,
        as_int: bool = True,
        round_digits: int | None = None,
        weight_range: tuple[float, float] = (1.0, 100.0),
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a symmetric n×n distance matrix D with D[i,i]=0 and D[i,j]=D[j,i]>0.

        Args:
            n: Number of cities.
            method: ``"euclidean"`` uses random coordinates; ``"random_symmetric"``
                samples a symmetric matrix directly.
            coord_range: Range for coordinates in Euclidean mode.
            dim: Dimensionality for Euclidean coordinates.
            as_int: If True, round distances; off-diagonal zeros are clamped to 1.
            round_digits: When ``as_int=False``, number of decimals to round to.
            weight_range: Range for random-symmetric distances (lower bound must be > 0).
            seed: RNG seed.

        Returns:
            np.ndarray: Symmetric distance matrix.

        Raises:
            ValueError: If ``weight_range`` lower bound is invalid or method is unknown.
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
            if a <= 0:
                raise ValueError("weight_range lower bound must be > 0.")
            U = rng.uniform(a, b, size=(n, n))
            U = np.triu(U, k=1)
            D = U + U.T
            np.fill_diagonal(D, 0.0)
        else:
            raise ValueError('method must be "euclidean" or "random_symmetric"')

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
