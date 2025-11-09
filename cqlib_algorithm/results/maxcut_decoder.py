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

"""MaxCut decoder."""

from __future__ import annotations
import matplotlib.pyplot as plt

from cqlib_algorithm.results.utils import parse_probability
from cqlib_algorithm.visualization.maxcut_plot import plot_maxcut
from cqlib_algorithm.execution.objective import energy_of_bitstring
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian


Edge = tuple[int, int]


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


def decode_from_platform_result(
    result: dict, n: int, *, choose_ones: bool = True, ising: IsingHamiltonian
) -> tuple[str, str, list[int]]:
    """Decode a MaxCut partition from a platform result payload.

    Picks the best bitstring (minimum Ising energy) and derives the partition.

    Args:
        result: Platform result dict containing ``"probability"``.
        n: Number of vertices/qubits.
        choose_ones: If True, the partition is the set of indices with bit '1';
            otherwise choose the '0' side.
        ising: Ising Hamiltonian used to score bitstrings.

    Returns:
        tuple[str, str, list[int]]:
            - ``best_raw``: Best bitstring in platform order.
            - ``best_std``: Best bitstring in standard order (here same as raw).
            - ``partition``: Indices in the selected side of the cut.
    """
    prob = result.get("probability", {})
    best_raw = best_bitstring_from_probability(prob, ising)
    print("Best Qubit string: ", best_raw)

    part: list[int] = []
    for i, b in enumerate(best_raw):
        if (b == "1") == choose_ones:
            part.append(i)
    return best_raw, best_raw, part


def plot_maxcut_solution(
    n: int,
    weights: dict[Edge, float],
    result: dict,
    *,
    choose_ones: bool = True,
    title: str = "MaxCut Solution(QAOA)",
    pos: dict[int, tuple[float, float]] | None = None,
    show: bool = True,
    ising: IsingHamiltonian,
) -> list[int]:
    """Visualize a MaxCut solution decoded from the platform result.

    Args:
        n: Number of vertices.
        weights: Edge weights mapping ``{(i, j): w}`` with ``i < j``.
        result: Platform result dict containing a ``"probability"`` field.
        choose_ones: Select the subset where bit == '1' (or '0' if False).
        title: Plot title.
        pos: Optional fixed positions for nodes (for deterministic layouts).
        show: If True, call ``plt.show()``.
        ising: Ising Hamiltonian used to score bitstrings.

    Returns:
        list[int]: The decoded vertex set for one side of the cut.
    """
    best_raw, _, part = decode_from_platform_result(
        result, n=n, choose_ones=choose_ones, ising=ising
    )
    print("Best solution:", part)

    cut_value = 0.0
    for (i, j), w in weights.items():
        ii, jj = (i, j) if i < j else (j, i)
        if best_raw[ii] != best_raw[jj]:
            cut_value += float(w)
    print(f"Max-Cut value: {cut_value:.3f}")

    plot_maxcut(n=n, weights=weights, partition=part, pos=pos, title=title)
    if show:
        plt.show()
    return part
