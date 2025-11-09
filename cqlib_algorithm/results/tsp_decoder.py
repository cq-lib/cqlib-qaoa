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

"""Tsp decoder."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from cqlib_algorithm.results.utils import parse_probability
from cqlib_algorithm.visualization.tsp_plot import plot_tsp
from cqlib_algorithm.execution.objective import energy_of_bitstring
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian


def best_bitstring_from_probability(
    prob: dict[str, float] | str, ising: IsingHamiltonian
) -> str:
    """Select the lowest-energy bitstring under an Ising Hamiltonian.

    Args:
        prob: Probability map ``{bitstring: p}`` or a JSON string encoding it.
        ising: Target :class:`IsingHamiltonian` used to evaluate energies.

    Returns:
        str: Bitstring with minimal energy.
    """
    p = parse_probability(prob)
    best_b, best_E = None, None
    for b, pr in p.items():
        E = energy_of_bitstring(ising, b)
        if best_E is None or E < best_E:
            best_b, best_E = b, E
    return best_b


def _bitstr_to_assignment(bitstr: str, n: int) -> list[list[int]]:
    """Convert a length-N bitstring (N = n*n) into an n×n one-hot matrix.

    Indexing convention is ``idx = i*n + t`` for city ``i`` at position ``t``.
    If the bitstring is longer than ``n*n``, only the rightmost ``n*n`` bits are used.

    Args:
        bitstr: Flat assignment bitstring.
        n: Number of cities.

    Returns:
        list[list[int]]: One-hot assignment matrix ``X[i][t] ∈ {0,1}``.

    Raises:
        ValueError: If the bitstring is shorter than ``n*n``.
    """
    N = n * n
    if len(bitstr) < N:
        raise ValueError(f"bitstring length {len(bitstr)} does not match expected {N}.")
    if len(bitstr) > N:
        bitstr = bitstr[-N:]
    X = [[0] * n for _ in range(n)]
    for i in range(n):
        for t in range(n):
            idx = i * n + t
            X[i][t] = 1 if bitstr[idx] == "1" else 0
    return X


def _assignment_to_tour(X: list[list[int]]) -> list[int]:
    """Decode a one-hot matrix to a tour with simple repair.

    Args:
        X: One-hot assignment matrix ``n×n``.

    Returns:
        list[int]: Tour as a city sequence of length ``n``.
    """
    n = len(X)
    unused = set(range(n))
    tour: list[int] = []
    for t in range(n):
        cands = [i for i in unused if X[i][t] == 1]
        if len(cands) == 1:
            i = cands[0]
        elif len(cands) > 1:
            i = cands[0]
        else:
            i = min(unused)
        tour.append(i)
        if i in unused:
            unused.remove(i)
    for i in sorted(unused):
        tour.append(i)
    return tour[:n]


def decode_from_platform_result(
    result: dict, n: int, *, ising: IsingHamiltonian
) -> tuple[str, list[int], list[list[int]]]:
    """Decode a TSP tour from a platform result payload.

    Picks the best bitstring (minimum Ising energy), converts it to a one-hot
    matrix, then to a tour.

    Args:
        result: Platform result dict containing ``"probability"``.
        n: Number of cities.
        ising: Ising Hamiltonian used to score bitstrings.

    Returns:
        tuple[str, list[int], list[list[int]]]:
            - ``best_raw``: Best bitstring in platform order.
            - ``tour``: City sequence of length ``n`` (positions 0..n-1).
            - ``X``: One-hot assignment matrix ``n×n``.
    """
    prob = result.get("probability", {})
    best_raw = best_bitstring_from_probability(prob, ising)
    print("Best Qubit string: ", best_raw)

    X = _bitstr_to_assignment(best_raw, n)
    tour = _assignment_to_tour(X)
    return best_raw, tour, X


def plot_tsp_solution(
    distance_matrix,
    result: dict,
    *,
    title: str = "TSP Solution (QAOA)",
    show: bool = True,
    ising: IsingHamiltonian,
) -> list[int]:
    """Visualize a TSP tour decoded from the platform result.

    Args:
        distance_matrix: Either an ``n×n`` distance matrix or ``n×2`` coordinates.
        result: Platform result dict containing a ``"probability"`` field.
        title: Plot title.
        show: If True, call ``plt.show()`` after plotting.
        ising: Ising Hamiltonian used to score bitstrings.

    Returns:
        list[int]: The decoded tour (city order).
    """
    n = len(distance_matrix)
    _, tour, _ = decode_from_platform_result(result, n=n, ising=ising)
    print("Best solution:", tour)

    dm = np.asarray(distance_matrix, dtype=float)
    if dm.ndim == 2 and dm.shape[1] == 2 and dm.shape[0] == n:
        coords = dm
        dm = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                dm[i, j] = dm[j, i] = d
    elif dm.shape != (n, n):
        raise ValueError(
            "distance_matrix must be (n,n) distances or (n,2) coordinates."
        )

    total_dist = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        total_dist += float(dm[i, j])
    print(f"Best TSP value: {total_dist:.3f}")

    plot_tsp(distance_matrix, tour=tour, title=title)
    if show:
        plt.show()
    return tour
