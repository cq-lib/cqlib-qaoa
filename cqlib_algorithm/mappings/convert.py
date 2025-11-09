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

"""Problem mappings between MaxCut/TSP/VRP ↔ QUBO and generic QUBO → Ising.

This module provides:
- ``maxcut_to_qubo``: MaxCut → QUBO
- ``tsp_to_qubo``: TSP → QUBO
- ``vrp_to_qubo``: VRP → QUBO
- ``qubo_to_ising``: Generic QUBO → Ising
"""

import numpy as np

from cqlib_algorithm.problems.maxcut import MaxCut
from cqlib_algorithm.problems.tsp import TSP
from cqlib_algorithm.problems.vrp import VRP
from cqlib_algorithm.mappings.qubo import QUBO
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian

Edge = tuple[int, int]


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
        Q[a, b] += -1.0 * w
        Q[b, a] += -1.0 * w
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
            Q[u, v] += 0.5 * coef 
            Q[v, u] += 0.5 * coef 

    for j in range(n):
        for i in range(n):
            u = idx(i, j)
            add_sym(u, u, A_pen) 
            c[u] += -2.0 * A_pen 
        offset += A_pen     

        for i in range(n):
            for k in range(i + 1, n):
                add_sym(idx(i, j), idx(k, j), 2.0 * A_pen)

    for i in range(n):
        for j in range(n):
            u = idx(i, j)
            add_sym(u, u, A_pen) 
            c[u] += -2.0 * A_pen
        offset += A_pen

        for j in range(n):
            for k in range(j + 1, n):
                add_sym(idx(i, j), idx(i, k), 2.0 * A_pen)

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


# # ------- VRP → QUBO -------
def vrp_to_qubo(problem: VRP, A_assign: float = 10000, A_pos: float = 10000) -> "QUBO":
    """Convert a Vehicle Routing Problem (VRP) instance into a QUBO model.

    This function automatically detects whether the VRP instance includes
    vehicle capacity information (i.e., `positions_per_vehicle` or `capacity` > 0).
    If so, it uses a **position-based capacitated formulation**; otherwise,
    it uses a **uncapacitated arc-based formulation**.

    When capacity/position information is provided, binary variables are:
        x_{i,p,k} = 1 if customer i is served at position p by vehicle k.

    Otherwise (uncapacitated case), binary variables are:
        y_{i,j} = 1 if the directed arc i -> j is traversed.

    Args:
        vrp: A VRP instance with the following attributes:
            - n: int, number of nodes including depot (0).
            - distance: np.ndarray, (n × n) distance matrix.
            - vehicle_count: int, number of vehicles.
            - capacity (optional): int, vehicle capacity.
        A_assign: float, penalty coefficient for assignment or degree constraints.
        A_pos: float, penalty coefficient for position uniqueness constraints.

    Returns:
        QUBO: A quadratic unconstrained binary optimization (QUBO) model
        with attributes:
            - Q: np.ndarray, quadratic coefficient matrix.
            - c: np.ndarray, linear coefficient vector.
            - offset: float, constant offset term.
            - sense: "min", objective is to be minimized.
    """
    n = int(problem.n)
    K = int(problem.vehicle_count)
    D = problem.distance

    # ---------------------------------------------------------------------
    # Detect whether the instance contains capacity information.
    # ---------------------------------------------------------------------
    P = None
    if getattr(problem, "positions_per_vehicle", None) and int(problem.positions_per_vehicle) > 0:
        P = int(problem.positions_per_vehicle)
    elif getattr(problem, "capacity", None) and int(problem.capacity) > 0:
        P = int(problem.capacity)

    # =====================================================================
    # Case 1: Capacitated VRP using position-based binary variables x_{i,p,k}
    # =====================================================================
    if P is not None and P > 0:
        customers = list(range(1, n))         
        I = len(customers)
        N = I * P * K                          

        Q = np.zeros((N, N), dtype=float)
        c = np.zeros(N, dtype=float)
        offset = 0.0

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

        for k in range(K):
            for i in customers:
                u = vid(i, 0, k)
                c[u] += D[0, i]

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

    # =====================================================================
    # Case 2: Uncapacitated VRP using arc-based binary variables y_{i,j}
    # =====================================================================
    depot = 0
    
    def eid(i: int, j: int) -> int:
        return i * (n - 1) + (j - 1 if j > i else j)

    N = n * (n - 1)
    Q = np.zeros((N, N), dtype=float)
    c = np.zeros(N, dtype=float)
    offset = 0.0

    def add_sym(u: int, v: int, val: float):
        if u == v:
            Q[u, u] += val
        else:
            Q[u, v] += 0.5 * val
            Q[v, u] += 0.5 * val

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c[eid(i, j)] += float(D[i, j])

    def add_equal_sum_squared(vars_idx: list[int], t: float, A: float):
        nonlocal offset
        m = len(vars_idx)
        if m == 0:
            offset += A * (t ** 2)
            return
        for u in vars_idx:
            add_sym(u, u, A)      
            c[u] += -2.0 * A * t
        for a in range(m):
            ua = vars_idx[a]
            for b in range(a + 1, m):
                ub = vars_idx[b]
                add_sym(ua, ub, 2.0 * A) 
        offset += A * (t ** 2)

    for v in range(n):
        if v == depot:
            continue
        out_vars = [eid(v, j) for j in range(n) if j != v]
        in_vars  = [eid(i, v) for i in range(n) if i != v]
        add_equal_sum_squared(out_vars, 1.0, A_assign)
        add_equal_sum_squared(in_vars,  1.0, A_assign)

    out_vars_dep = [eid(depot, j) for j in range(n) if j != depot]
    in_vars_dep  = [eid(i, depot) for i in range(n) if i != depot]
    add_equal_sum_squared(out_vars_dep, float(K), A_assign)
    add_equal_sum_squared(in_vars_dep,  float(K), A_assign)

    S_max = 0
    if S_max >= 2:
        import itertools
        for s in range(2, min(S_max, n - 1) + 1):
            for S in itertools.combinations([v for v in range(n) if v != depot], s):
                idxs = [eid(i, j) for i in S for j in S if i != j]
                add_equal_sum_squared(idxs, float(s - 1), A_pos)

    qubo = QUBO(Q=Q, c=c, offset=offset)
    setattr(qubo, "sense", "min")
    return qubo


# ------- QUBO → Ising -------
def qubo_to_ising(qubo: QUBO, *, zero_tol: float = 1e-12) -> IsingHamiltonian:
    """Convert a QUBO to Ising.
    """
    sense = getattr(qubo, "sense", "min")
    sign = -1.0 if sense == "max" else 1.0

    Q_eff = sign * np.asarray(qubo.Q, dtype=float)
    c_eff = sign * np.asarray(qubo.c, dtype=float)
    off_eff = sign * float(getattr(qubo, "offset", 0.0))

    Qs = 0.5 * (Q_eff + Q_eff.T)
    n = Qs.shape[0]
    one = np.ones(n)

    h_vec = -0.5 * (Qs @ one + c_eff)
    h = {i: float(h_vec[i]) for i in range(n) if abs(h_vec[i]) > zero_tol}

    offset = 0.25 * float(one @ (Qs @ one)) + 0.5 * float(c_eff @ one) + off_eff

    J: dict[tuple[int, int], float] = {}
    for i in range(n):
        offset += 0.25 * float(Qs[i, i]) 
        for j in range(i + 1, n):
            val = 0.5 * Qs[i, j]
            if abs(val) > zero_tol:
                J[(i, j)] = float(val)

    return IsingHamiltonian(n=n, h=h, J=J, offset=offset)
