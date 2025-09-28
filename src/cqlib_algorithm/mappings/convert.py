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

"""Problem mappings between MaxCut/TSP/VRP ↔ QUBO and generic QUBO → Ising.

This module provides:
- ``maxcut_to_qubo``: MaxCut → QUBO
- ``tsp_to_qubo``: TSP → QUBO
- ``vrp_to_qubo``: VRP → QUBO
- ``qubo_to_ising``: Generic QUBO → Ising
"""

import numpy as np
from typing import Dict, Tuple

from cqlib_algorithm.problems.maxcut import MaxCut
from cqlib_algorithm.problems.tsp import TSP
from cqlib_algorithm.problems.vrp import VRP
from cqlib_algorithm.mappings.qubo import QUBO
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian

Edge = Tuple[int, int]

# ------- MaxCut → QUBO -------
def maxcut_to_qubo(problem: MaxCut) -> QUBO:
    """Convert a MaxCut instance into a QUBO.

    The mapping follows the common binary formulation where the objective is to
    maximize the cut weight. The returned QUBO has its ``sense`` set to ``"max"``.

    Args:
        problem: MaxCut problem with:
            - ``n``: number of nodes
            - ``weights``: edge weight mapping ``{(i, j): w}`` (i ≠ j)

    Returns:
        QUBO: Quadratic model with fields:
            - ``Q``: quadratic coefficient matrix (n × n)
            - ``c``: linear coefficient vector (n,)
            - ``offset``: constant term
    """ 
    n = problem.n
    Q = np.zeros((n, n), dtype=float)
    deg = np.zeros(n, dtype=float)
    total_w = 0.0

    for (i, j), w in problem.weights.items():
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        w = float(w)
        Q[a, b] += -2.0 * w
        Q[b, a] += -2.0 * w
        deg[a] += w
        deg[b] += w
        total_w += w

    c = deg
    offset = 0
    qubo = QUBO(Q=Q, c=c, offset=offset)
    setattr(qubo, "sense", "max")
    return qubo

# # ------- TSP → QUBO -------
def tsp_to_qubo(problem: TSP, A: float = 10000) -> QUBO:
    """Convert a TSP instance into a QUBO.

    Variables: x_{i,t} indicates city i is at tour position t.

    Constraints (penalized):
        1) Each position has exactly one city.
        2) Each city appears exactly once.

    Objective: Minimize total tour length (including wrap-around).

    Args:
        problem: TSP instance with ``n`` and ``distance_matrix`` (n × n).
        A: Penalty strength for hard constraints.

    Returns:
        QUBO: Quadratic model with ``sense="min"``.
    """
    n = problem.n
    D = np.asarray(problem.distance_matrix, dtype=float)
    N = n * n
    Q = np.zeros((N, N), dtype=float)
    c = np.zeros(N, dtype=float)
    offset = 0.0

    A_pen = 2.0 * A 

    def idx(i, j):
        """Linear index for variable x_{i,j}."""
        return i * n + j

    def add_sym(u, v, coef):
        """Accumulate symmetric quadratic term."""
        if u == v:
            Q[u, u] += coef
        else:
            Q[u, v] += coef 
            Q[v, u] += coef 

    # Constraint 1: each position has exactly one city => A_pen * (1 - sum_i x_{i,j})^2
    for j in range(n):
        for i in range(n):
            u = idx(i, j)
            add_sym(u, u, A_pen) 
            c[u] += -2.0 * A_pen 
        offset += A_pen     

        for i in range(n):
            for k in range(i + 1, n):
                add_sym(idx(i, j), idx(k, j), 2.0 * A_pen)

    # Constraint 2: each city appears exactly once => A_pen * (1 - sum_j x_{i,j})^2
    for i in range(n):
        for j in range(n):
            u = idx(i, j)
            add_sym(u, u, A_pen) 
            c[u] += -2.0 * A_pen
        offset += A_pen

        for j in range(n):
            for k in range(j + 1, n):
                add_sym(idx(i, j), idx(i, k), 2.0 * A_pen)

    # Objective: path distance (adjacent positions t -> t+1 with wrap-around)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for t in range(n):
                u = idx(i, t)
                v = idx(j, (t + 1) % n)
                add_sym(u, v, D[i, j]) 

    qubo = QUBO(Q=Q, c=c, offset=offset)
    setattr(qubo, "sense", "min")
    return qubo


def vrp_to_qubo(vrp: VRP, A_assign: float = 10000, A_pos: float = 10000) -> QUBO:
    """Convert a capacitated VRP into a QUBO.

    Variables: x_{i,p,k} indicates customer i is placed at position p of vehicle k.
    Depot is node 0; customers are 1..n-1.

    Constraints (penalized):
        1) Each customer appears exactly once.
        2) Each (vehicle, position) holds at most one customer (pairwise penalty).

    Objective:
        Minimize total traveled distance including depot→first, neighbors, and last→depot.

    Args:
        vrp: VRP instance with fields:
            - ``n``: number of nodes including depot (0)
            - ``distance``: (n × n) symmetric distance matrix or coordinates
            - ``vehicle_count``: number of vehicles K
            - ``positions_per_vehicle``: positions per vehicle P (capacity)
        A_assign: Penalty for the assignment (exactly-once) constraint.
        A_pos: Penalty for the per-slot uniqueness constraint.

    Returns:
        QUBO: Quadratic model with ``sense="min"``.
    """
    n, K, P = vrp.n, vrp.vehicle_count, vrp.positions_per_vehicle
    D = vrp.distance
    customers = list(range(1, n))         
    I = len(customers)
    N = I * P * K                          

    Q = np.zeros((N, N), dtype=float)
    c = np.zeros(N, dtype=float)
    offset = 0.0

    # Variable indexing: i in customers → (i-1) compact index
    def vid(i: int, p: int, k: int) -> int:
        ii = i - 1
        return ii * (P * K) + p * K + k 

    def add_sym(u: int, v: int, val: float):
        if u == v:
            Q[u, u] += val
        else:
            Q[u, v] += val
            Q[v, u] += val

    # Constraint 1: each customer exactly once  A_assign * (Σ_{p,k} x - 1)^2
    for i in customers:
        vars_i = [vid(i, p, k) for p in range(P) for k in range(K)]
        for u in vars_i:
            add_sym(u, u, A_assign)    
            c[u]    += -2.0 * A_assign
        offset += A_assign
        for a in range(len(vars_i)):
            for b in range(a + 1, len(vars_i)):
                add_sym(vars_i[a], vars_i[b], 2 * A_assign)

    # Constraint 2: each (vehicle, position) at most one customer
    for k in range(K):
        for p in range(P):
            vars_pk = [vid(i, p, k) for i in customers]
            for a in range(I):
                for b in range(a + 1, I):
                    add_sym(vars_pk[a], vars_pk[b], 2 * A_pos)

    # Objective: total distance
    # depot -> first
    for k in range(K):
        for i in customers:
            u = vid(i, 0, k)
            c[u] += D[0, i]

    # adjacent positions
    for k in range(K):
        for p in range(P - 1):
            for i in customers:
                for j in customers:
                    if i == j: 
                        continue
                    u = vid(i, p, k)
                    v = vid(j, p + 1, k)
                    add_sym(u, v, D[i, j])

    # last -> depot
    for k in range(K):
        for i in customers:
            u = vid(i, P - 1, k)
            c[u] += D[i, 0]

    qubo = QUBO(Q=Q, c=c, offset=offset)
    setattr(qubo, "sense", "min") 
    return qubo

# ------- QUBO → Ising -------
def qubo_to_ising(qubo: QUBO, *, zero_tol: float = 1e-12) -> IsingHamiltonian:
    """Convert a generic QUBO to an Ising Hamiltonian.

    The conversion supports linear and quadratic terms. If ``qubo.sense == "max"``,
    the objective is flipped to minimization before mapping, i.e., Q, c, offset
    are multiplied by -1 so that the resulting Ising encodes a minimization task.

    Args:
        qubo: QUBO model with fields:
            - ``Q``: (n × n) quadratic matrix
            - ``c``: (n,) linear vector
            - ``offset``: scalar constant
            - optional ``sense``: "min" (default) or "max"
        zero_tol: Numerical tolerance below which coefficients are dropped.

    Returns:
        IsingHamiltonian: ``(n, h, J, offset)`` where
            - ``h``: dict ``{i: h_i}`` local fields
            - ``J``: dict ``{(i, j): J_ij}`` couplings with i < j
            - ``offset``: constant energy term
    """
    # Normalize objective direction
    sense = getattr(qubo, "sense", "min")
    sign = -1.0 if sense == "max" else 1.0

    Q_eff = sign * qubo.Q
    c_eff = sign * qubo.c
    off_eff = sign * float(qubo.offset)

    # Symmetrize and map
    Qs = 0.5 * (Q_eff + Q_eff.T)
    n = Qs.shape[0]
    one = np.ones(n)

    J_full = Qs / 4.0
    h_vec  = -0.25 * (Qs @ one) - 0.5 * c_eff
    diag_Q = Qs.diagonal() if hasattr(Qs, "diagonal") else np.diag(Qs)
    h_vec = np.asarray(h_vec).reshape(-1) - 0.25 * np.asarray(diag_Q).reshape(-1)

    offset = 0.25 * 0.5 * float(one @ Qs @ one) + 0.5 * float(c_eff @ one) + off_eff
    diag_sum = float(Qs.diagonal().sum()) if hasattr(Qs, "diagonal") else float(np.trace(Qs))
    offset += 0.25 * 0.5 * diag_sum

    J: Dict[Edge, float] = {}
    for i in range(n):
        offset += float(J_full[i, i]) 
        for j in range(i + 1, n):
            if abs(J_full[i, j]) > zero_tol:
                J[(i, j)] = float(J_full[i, j])

    h = {i: float(h_vec[i]) for i in range(n) if abs(h_vec[i]) > zero_tol}
    return IsingHamiltonian(n=n, h=h, J=J, offset=offset)
