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

"""Unit tests for QUBO/Ising mappings and classic problem converters."""

import importlib
import numpy as np
import pytest

MODULE_M = "cqlib_algorithm.mappings.convert"
MODULE_H = "cqlib_algorithm.mappings.hamiltonian"
mappings = importlib.import_module(MODULE_M)
ham_mod = importlib.import_module(MODULE_H)
IsingHamiltonian = ham_mod.IsingHamiltonian


# ---------------------------------------------------------------------
# Lightweight stubs for problem inputs
# ---------------------------------------------------------------------
class _MaxCutStub:
    """Minimal MaxCut holder with node count and edge weights."""

    def __init__(self, n, weights):
        self.n = n
        self.weights = weights


def test_maxcut_to_qubo():
    """MaxCut → QUBO: verify symmetry, linear degree terms, and sense=max."""
    mc = _MaxCutStub(
        n=4,
        weights={(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (0, 3): 1.0},
    )
    qubo = mappings.maxcut_to_qubo(mc)

    Q, c, offset = qubo.Q, qubo.c, qubo.offset
    assert Q.shape == (4, 4)
    assert c.shape == (4,)
    assert offset == 0
    assert getattr(qubo, "sense") == "max"
    assert np.allclose(Q, Q.T)
    assert Q[0, 1] == Q[1, 0] == -1.0
    assert Q[1, 2] == Q[2, 1] == -1.0
    assert Q[2, 3] == Q[3, 2] == -1.0
    assert Q[0, 3] == Q[3, 0] == -1.0
    assert np.allclose(c, np.array([2.0, 2.0, 2.0, 2.0]))


class _TSPStub:
    """Minimal TSP holder with city count and distance matrix."""

    def __init__(self, n, distance_matrix):
        self.n = n
        self.distance_matrix = distance_matrix


def test_tsp_to_qubo():
    """TSP → QUBO: verify hard-constraint structure and distance coupling."""
    n = 3
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    A = 10.0
    A_pen = 2 * A
    tsp = _TSPStub(n=n, distance_matrix=D)
    qubo = mappings.tsp_to_qubo(tsp, A=A)

    Q, c, offset = qubo.Q, qubo.c, qubo.offset
    N = n * n
    assert Q.shape == (N, N)
    assert c.shape == (N,)
    assert getattr(qubo, "sense") == "min"
    assert np.allclose(Q, Q.T)

    def idx(i, j):
        return i * n + j
    
    diag_min = 2 * A_pen
    for u in range(N):
        assert Q[u, u] >= diag_min - 1e-12

    assert np.allclose(c, -4 * A_pen * np.ones(N), atol=1e-12)
    assert offset == pytest.approx(2 * n * A_pen, rel=0, abs=1e-12)

    u = idx(0, 2)
    v = idx(1, 0)
    assert Q[u, v] == pytest.approx(0.5 * D[0, 1], abs=1e-12)
    assert Q[v, u] == pytest.approx(0.5 * D[0, 1], abs=1e-12)


class _VRPStub:
    """Minimal VRP holder with distance matrix and vehicle count."""

    def __init__(self, n, distance, vehicle_count):
        self.n = n
        self.distance = np.asarray(distance, dtype=float)
        self.vehicle_count = vehicle_count


def test_vrp_to_qubo():
    """VRP → QUBO (K=1): verify linear/diagonal penalties and offset."""
    D = np.array(
        [
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 2.0],
            [4.0, 2.0, 0.0],
        ]
    )
    A = 10.0
    vrp = _VRPStub(n=3, distance=D, vehicle_count=1)
    qubo = mappings.vrp_to_qubo(vrp, A_assign=A, A_pos=A)

    Q, c, offset = qubo.Q, qubo.c, qubo.offset
    n = vrp.n
    N = n * (n - 1)

    assert Q.shape == (N, N)
    assert c.shape == (N,)
    assert getattr(qubo, "sense") == "min"
    assert np.allclose(Q, Q.T)


    def eid(i, j):
        """Index of binary var x_{i->j} flattened by excluding self-loops."""
        return i * (n - 1) + (j - 1 if j > i else j)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u = eid(i, j)
            assert c[u] == pytest.approx(D[i, j] - 4 * A, abs=1e-12)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u = eid(i, j)
            assert Q[u, u] == pytest.approx(2 * A, abs=1e-12)

    u = eid(1, 0)
    v = eid(1, 2)
    assert Q[u, v] == pytest.approx(A, abs=1e-12)
    assert Q[v, u] == pytest.approx(A, abs=1e-12)
    assert offset == pytest.approx(6 * A, abs=1e-12)


# ---------------------------------------------------------------------
# QUBO container and QUBO → Ising tests
# ---------------------------------------------------------------------
class _QUBOStub:
    """Simple container to mimic a QUBO object."""

    def __init__(self, Q, c, offset=0.0, sense="min"):
        self.Q = np.asarray(Q, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.offset = float(offset)
        setattr(self, "sense", sense)


def test_qubo_to_ising_simple_min():
    """QUBO → Ising (min): verify h, J, and offset on a small example."""
    Q = np.array([[0.0, 2.0], [0.0, 0.0]])
    c = np.array([0.0, 0.0])
    qubo = _QUBOStub(Q=Q, c=c, offset=0.0, sense="min")

    H = mappings.qubo_to_ising(qubo)
    assert H.n == 2
    assert H.h[0] == pytest.approx(-0.5, abs=1e-12)
    assert H.h[1] == pytest.approx(-0.5, abs=1e-12)
    assert H.J[(0, 1)] == pytest.approx(0.5, abs=1e-12)
    assert H.offset == pytest.approx(0.5, abs=1e-12)


def test_qubo_to_ising_simple_max_sign_flip():
    """QUBO → Ising (max): verify sign flip for h, J, and offset."""
    Q = np.array([[0.0, 2.0], [0.0, 0.0]])
    c = np.array([0.0, 0.0])
    qubo = _QUBOStub(Q=Q, c=c, offset=0.0, sense="max")

    H = mappings.qubo_to_ising(qubo)
    assert H.h[0] == pytest.approx(0.5, abs=1e-12)
    assert H.h[1] == pytest.approx(0.5, abs=1e-12)
    assert H.J[(0, 1)] == pytest.approx(-0.5, abs=1e-12)
    assert H.offset == pytest.approx(-0.5, abs=1e-12)


def test_qubo_to_ising_zero_tol_filters_small_terms():
    """Zero-tolerance should filter tiny h/J entries while preserving type."""
    eps = 1e-14
    Q = np.array([[eps, eps], [eps, eps]])
    c = np.array([eps, -eps])
    qubo = _QUBOStub(Q=Q, c=c, offset=eps, sense="min")
    H = mappings.qubo_to_ising(qubo, zero_tol=1e-12)
    assert isinstance(H.h, dict) and len(H.h) in (0, 2)
    assert isinstance(H.J, dict) and len(H.J) == 0


def test_end_to_end_maxcut_qubo_to_ising_single_edge():
    """End-to-end sanity check: 1-edge MaxCut → QUBO → Ising."""

    class _MaxCutStub:
        def __init__(self, n, weights):
            self.n = n
            self.weights = weights

    w = 4.0
    mc = _MaxCutStub(n=2, weights={(0, 1): w})
    qubo = mappings.maxcut_to_qubo(mc) 
    H = mappings.qubo_to_ising(qubo) 

    assert H.n == 2
    assert H.h == {}
    assert H.J[(0, 1)] == pytest.approx(w / 2.0, abs=1e-12)
    assert H.offset == pytest.approx(-w / 2.0, abs=1e-12)
