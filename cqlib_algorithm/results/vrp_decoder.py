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

"""Vrp decoder."""

import numpy as np
import matplotlib.pyplot as plt
from cqlib_algorithm.results.utils import parse_probability
from cqlib_algorithm.visualization.vrp_plot import plot_vrp
from cqlib_algorithm.execution.objective import energy_of_bitstring
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian


def best_bitstring_from_probability(
    prob: dict[str, float] | str, ising: IsingHamiltonian
) -> str:
    """Selects the bitstring with minimum Ising energy.

    Args:
        prob: Probability dictionary mapping bitstrings to probabilities,
            or a JSON-encoded string of such a dictionary.
        ising: Ising Hamiltonian used to evaluate bitstring energies.

    Returns:
        str: Bitstring corresponding to the lowest Ising energy.
    """
    p = parse_probability(prob)
    best_b, best_E = None, None
    for b, pr in p.items():
        E = energy_of_bitstring(ising, b)
        if best_E is None or E < best_E:
            best_b, best_E = b, E
    return best_b

def _bitstr_to_assignment(bitstr: str, n: int, K: int, P: int) -> list[list[list[int]]]:
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
                X[ii][p][k] = 1 if bitstr[idx] == '1' else 0
    return X

def _assignment_to_routes(X: list[list[list[int]]]) -> list[list[int]]:
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

    assigned_slot: list[list[int | None]] = [[None for _ in range(K)] for _ in range(P)]
    used_in_vehicle = [set() for _ in range(K)]
    used_global = set()

    backlog = [] 
    for ii in range(I):
        cands: list[tuple[int, int]] = []
        for p in range(P):
            for k in range(K):
                if X[ii][p][k] == 1 and assigned_slot[p][k] is None and (ii not in used_in_vehicle[k]):
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

def _bitstr_to_arcs(bitstr: str, n: int) -> np.ndarray:
    """Converts a flat bitstring into a directed edge selection matrix.

    Each bit represents whether the directed arc i→j is selected (i ≠ j).
    The flattening index follows the same convention as vrp_to_qubo:
        eid(i, j) = i * (n - 1) + (j - 1 if j > i else j)

    Args:
        bitstr: Flat bitstring of length n*(n-1).
        n: Total number of nodes including the depot.

    Returns:
        np.ndarray: (n × n) binary adjacency matrix Y where Y[i, j] = 1
            if arc i→j is active, else 0.

    Raises:
        ValueError: If the bitstring length does not match n*(n-1).
    """
    if len(bitstr) < n * (n - 1):
        raise ValueError(f"Bitstring length {len(bitstr)} < n*(n-1) = {n*(n-1)}.")
    if len(bitstr) > n * (n - 1):
        bitstr = bitstr[-n * (n - 1):]

    def eid(i: int, j: int) -> int:
        return i * (n - 1) + (j - 1 if j > i else j)

    Y = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            idx = eid(i, j)
            Y[i, j] = 1 if bitstr[idx] == "1" else 0
    return Y


def _arcs_to_routes(Y: np.ndarray, depot: int, K: int) -> list[list[int]]:
    """Extracts vehicle routes from the directed edge matrix.

    The algorithm traces arcs starting from the depot to reconstruct K
    routes that each form a (nearly) closed loop returning to the depot.

    Args:
        Y: (n × n) binary matrix representing selected arcs.
        depot: Index of the depot node (commonly 0).
        K: Number of vehicles.

    Returns:
        list[list[int]]: A list of K routes, each a list of customer indices.
            Customers are labeled 1..n-1.

    Notes:
        - Subtours or disconnected routes are heuristically repaired.
        - If fewer than K routes are found, empty routes are appended.
    """
    n = Y.shape[0]
    routes: list[list[int]] = []
    succ = {i: [j for j in range(n) if i != j and Y[i, j] == 1] for i in range(n)}
    used_edges = set()

    def pop_next(u: int) -> int | None:
        """Finds the next unused successor of node u."""
        for v in succ[u]:
            if (u, v) not in used_edges:
                return v
        return None

    for _ in range(K):
        route = []
        u = depot
        visited = {depot}
        for _step in range(n + K + 5):
            v = pop_next(u)
            if v is None:
                break
            used_edges.add((u, v))
            if v == depot:
                break
            if v in visited:
                break
            if v != depot:
                route.append(v)
            visited.add(v)
            u = v
        routes.append(route)

    while len(routes) < K:
        routes.append([])

    covered = {v for r in routes for v in r}
    all_customers = set(range(n)) - {depot}
    backlog = sorted(list(all_customers - covered))

    def _argmin_route_len() -> int:
        lengths = [len(r) for r in routes]
        return min(range(K), key=lambda i: lengths[i])

    for v in backlog:
        routes[_argmin_route_len()].append(v)
    return routes


def decode_from_platform_result(
    result: dict,
    n: int,
    *,
    vehicle_count: int,
    ising: IsingHamiltonian,
    depot: int = 0,
) -> tuple[str, list[list[int]], np.ndarray]:
    """Decode VRP routes from platform result (auto detect capacity).

    Automatically detects whether the bitstring represents:
      - Uncapacitated (arc-based): N = n*(n-1)
      - Capacitated (assignment-based): N = (n-1)*P*K for some integer P>0

    Args:
        result: Platform result dictionary containing the key ``"probability"``.
        n: Total number of nodes including the depot.
        vehicle_count: Number of vehicles (K).
        ising: Ising Hamiltonian used for energy evaluation.
        depot: Index of the depot node (default: 0).

    Returns:
        tuple:
            - str: Best bitstring with minimum Ising energy.
            - list[list[int]]: Per-vehicle customer sequences.
            - np.ndarray: Binary arc matrix for arc case,
                          or assignment tensor for capacitated case.
    """
    prob = result.get("probability", {})
    best_raw = best_bitstring_from_probability(prob, ising)
    print("Best Qubit string:", best_raw)

    K = int(vehicle_count)
    N = len(best_raw)

    expect_arc = n * (n - 1)
    if N == expect_arc:
        Y = _bitstr_to_arcs(best_raw, n=n)
        routes = _arcs_to_routes(Y, depot=depot, K=K)
        return best_raw, routes, Y
    else:
        denom = (n - 1) * K
        P = N // denom
        X = _bitstr_to_assignment(best_raw, n=n, K=K, P=P)
        routes = _assignment_to_routes(X)
        return best_raw, routes, X


def plot_vrp_solution(
    distance,
    result: dict,
    *,
    n: int,
    vehicle_count: int,
    depot: int = 0,
    title: str = "VRP Solution (QAOA)",
    show: bool = True,
    ising: IsingHamiltonian,
) -> list[list[int]]:
    """Visualizes decoded VRP routes using arc-based decoding.

    Args:
        distance: Either (n×n) distance matrix or (n×2) coordinate array.
        result: Platform result dictionary containing ``"probability"``.
        n: Total number of nodes including depot.
        vehicle_count: Number of vehicles.
        depot: Index of the depot node (default: 0).
        title: Plot title.
        show: Whether to call plt.show() after plotting (default: True).
        ising: Ising Hamiltonian used for energy scoring.

    Returns:
        list[list[int]]: Decoded per-vehicle customer sequences.
    """
    _, routes, Y = decode_from_platform_result(
        result, n=n, vehicle_count=vehicle_count, ising=ising, depot=depot
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
        raise ValueError("distance must be (n,n) matrix or (n,2) coordinates.")

    def route_length(route: list[int]) -> float:
        """Computes the total route length for a single vehicle."""
        if not route:
            return 0.0
        length = dm[depot, route[0]]
        for a, b in zip(route, route[1:]):
            length += dm[a, b]
        length += dm[route[-1], depot]
        return float(length)

    total_len = 0.0
    for vid, r in enumerate(routes):
        r_clean = [u for u in r if u != depot]
        L = route_length(r_clean)
        total_len += L
        print(f"[Vehicle {vid}] length = {L:.3f} | route = {r_clean}")

    print(f"Total VRP distance: {total_len:.3f}")
    plot_vrp(distance, routes=routes, depot=depot, title=title)
    if show:
        plt.show()
    return routes
