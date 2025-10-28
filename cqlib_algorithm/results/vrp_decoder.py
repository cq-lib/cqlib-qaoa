# This code is part of cqlib-algorithm.
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

"""Vrp decoder."""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from cqlib_algorithm.results.utils import parse_probability
from cqlib_algorithm.visualization.vrp_plot import plot_vrp
from cqlib_algorithm.execution.objective import energy_of_bitstring
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian


def best_bitstring_from_probability(
    prob: Dict[str, float] | str, ising: IsingHamiltonian
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


def _bitstr_to_assignment(bitstr: str, n: int, K: int, P: int) -> List[List[List[int]]]:
    """Convert a flat bitstring into a 3D one-hot tensor for VRP.

    The tensor layout is ``X[I][P][K]`` where ``I = n-1`` (customers 1..n-1),
    ``P`` is positions per vehicle, and ``K`` is number of vehicles.

    Indexing convention (flattened):
        ``flat_idx = ii*(P*K) + p*K + k``

    If the bitstring is longer than ``I*P*K``, the rightmost bits are used.

    Args:
        bitstr: Flat assignment bitstring.
        n: Total nodes including depot (so customers are 1..n-1).
        K: Number of vehicles.
        P: Positions per vehicle.

    Returns:
        list[list[list[int]]]: One-hot assignment tensor ``X[I][P][K]``.

    Raises:
        ValueError: If the bitstring is shorter than ``I*P*K``.
    """
    I = n - 1
    N = I * P * K
    if len(bitstr) < N:
        raise ValueError(f"bitstring length {len(bitstr)} does not match expected {N}.")
    if len(bitstr) > N:
        bitstr = bitstr[-N:]

    X = [[[0 for _ in range(K)] for _ in range(P)] for _ in range(I)]
    for ii in range(I):
        for p in range(P):
            for k in range(K):
                idx = ii * (P * K) + p * K + k
                X[ii][p][k] = 1 if bitstr[idx] == "1" else 0
    return X


def _assignment_to_routes(X: List[List[List[int]]]) -> List[List[int]]:
    """Decode a 3D one-hot assignment tensor into vehicle routes.

    Strategy:
        1) Global uniqueness: each customer is assigned at most once.
        2) Slot preference: among multiple candidate slots for a customer,
           choose the smallest ``p``, then smallest ``k``.
        3) Backlog repair: unassigned customers are appended to the currently
           shortest route.

    Args:
        X: One-hot tensor ``X[I][P][K]``.

    Returns:
        list[list[int]]: Routes per vehicle; customers are labeled ``1..n-1``.
    """
    I = len(X)
    P = len(X[0])
    K = len(X[0][0])

    assigned_slot: List[List[Optional[int]]] = [
        [None for _ in range(K)] for _ in range(P)
    ]
    used_in_vehicle = [set() for _ in range(K)]
    used_global = set()

    backlog = []
    for ii in range(I):
        cands: List[Tuple[int, int]] = []
        for p in range(P):
            for k in range(K):
                if (
                    X[ii][p][k] == 1
                    and assigned_slot[p][k] is None
                    and (ii not in used_in_vehicle[k])
                ):
                    cands.append((p, k))

        if cands:
            cands.sort(key=lambda t: (t[0], t[1]))
            p_sel, k_sel = cands[0]
            assigned_slot[p_sel][k_sel] = ii
            used_in_vehicle[k_sel].add(ii)
            used_global.add(ii)
        else:
            backlog.append(ii)

    routes = [[] for _ in range(K)]
    for k in range(K):
        for p in range(P):
            ii = assigned_slot[p][k]
            if ii is not None:
                routes[k].append(ii + 1)

    def _argmin_route_len() -> int:
        lengths = [len(r) for r in routes]
        return min(range(K), key=lambda kk: lengths[kk])

    for ii in backlog:
        if ii in used_global:
            continue
        k_sel = _argmin_route_len()
        routes[k_sel].append(ii + 1)
        used_in_vehicle[k_sel].add(ii)
        used_global.add(ii)
    return routes


def decode_from_platform_result(
    result: Dict,
    n: int,
    *,
    vehicle_count: int,
    positions_per_vehicle: int,
    ising: IsingHamiltonian,
) -> Tuple[str, List[List[int]], List[List[List[int]]]]:
    """Decode VRP routes from a platform result payload.

    Picks the best bitstring (minimum Ising energy), converts to a 3D one-hot
    tensor, then derives the per-vehicle routes.

    Args:
        result: Platform result dict containing ``"probability"``.
        n: Total number of nodes including depot.
        vehicle_count: Number of vehicles.
        positions_per_vehicle: Slots per vehicle.
        ising: Ising Hamiltonian used to score bitstrings.

    Returns:
        Tuple[str, list[list[int]], list[list[list[int]]]]:
            - ``best_raw``: Best bitstring in platform order.
            - ``routes``: Per-vehicle customer sequences (labels 1..n-1).
            - ``X``: One-hot tensor ``X[I][P][K]``.
    """
    prob = result.get("probability", {})
    best_raw = best_bitstring_from_probability(prob, ising)
    print("Best Qubit string:", best_raw)

    X = _bitstr_to_assignment(best_raw, n=n, K=vehicle_count, P=positions_per_vehicle)
    routes = _assignment_to_routes(X)
    return best_raw, routes, X


def plot_vrp_solution(
    distance,
    result: Dict,
    *,
    n: int,
    vehicle_count: int,
    positions_per_vehicle: int,
    depot: int = 0,
    title: str = "VRP Solution (QAOA)",
    show: bool = True,
    ising: IsingHamiltonian,
) -> List[List[int]]:
    """Visualize VRP routes decoded from the platform result.

    Args:
        distance: Either an ``(n,n)`` distance matrix or ``(n,2)`` coordinates.
        result: Platform result dict containing ``"probability"``.
        n: Total number of nodes including depot.
        vehicle_count: Number of vehicles.
        positions_per_vehicle: Slots per vehicle.
        depot: Index of the depot (default 0).
        title: Plot title.
        show: If True, call ``plt.show()`` after plotting.
        ising: Ising Hamiltonian used to score bitstrings.

    Returns:
        list[list[int]]: Per-vehicle customer sequences.
    """
    _, routes, _ = decode_from_platform_result(
        result,
        n=n,
        vehicle_count=vehicle_count,
        positions_per_vehicle=positions_per_vehicle,
        ising=ising,
    )
    print("Best solution:", routes)

    dm = np.asarray(distance, dtype=float)
    if dm.ndim == 2 and dm.shape == (n, n):
        pass
    elif dm.ndim == 2 and dm.shape == (n, 2):
        coords = dm
        dm = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                dm[i, j] = dm[j, i] = d
    else:
        raise ValueError("distance must be (n,n) distances or (n,2) coordinates.")

    def route_length(route: List[int]) -> float:
        """Compute closed-route length: depot → route → depot."""
        if not route:
            return 0.0
        length = 0.0
        length += dm[depot, route[0]]
        for a, b in zip(route, route[1:]):
            length += dm[a, b]
        length += dm[route[-1], depot]
        return float(length)

    per_vehicle_len: List[float] = []
    total_len = 0.0
    for vid, r in enumerate(routes):
        r_clean = [u for u in r if u != depot]
        L = route_length(r_clean)
        per_vehicle_len.append(L)
        total_len += L
        print(f"[Vehicle {vid}] length = {L:.3f} | route = {r_clean}")

    print(f"Total VRP distance: {total_len:.3f}")

    plot_vrp(distance, routes=routes, depot=depot, title=title)
    if show:
        plt.show()
    return routes
