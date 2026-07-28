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

"""Unit tests for VRP decoding and plotting utilities."""

import importlib
from types import SimpleNamespace
import numpy as np
import pytest

MODULE_DEC = "cqlib_qaoa.results.vrp_decoder"
MODULE_VIS = "cqlib_qaoa.visualization.vrp_plot"

dec_mod = importlib.import_module(MODULE_DEC)
m_vis = importlib.import_module(MODULE_VIS)

best_bitstring_from_probability = getattr(dec_mod, "best_bitstring_from_probability")
_bitstr_to_arcs = getattr(dec_mod, "_bitstr_to_arcs")
_arcs_to_routes = getattr(dec_mod, "_arcs_to_routes")
decode_from_platform_result = getattr(dec_mod, "decode_from_platform_result")
plot_vrp_solution = getattr(dec_mod, "plot_vrp_solution")

m_vis = importlib.import_module("cqlib_qaoa.visualization.vrp_plot")


def _eid(n: int, i: int, j: int) -> int:
    """Linear index for directed edge (i, j) under arc encoding."""
    return i * (n - 1) + (j - 1 if j > i else j)


def make_bitstring_from_arcs(n: int, arcs: list[tuple[int, int]]) -> str:
    """Build an n*(n-1)-length bitstring from a set of directed arcs."""
    N = n * (n - 1)
    b = ["0"] * N
    for (i, j) in arcs:
        assert i != j
        b[_eid(n, i, j)] = "1"
    return "".join(b)


def test_arcs_to_routes_two_routes_from_depot_basic():
    """Reconstruct two routes from depot with backfilling of uncovered customers."""
    n = 5
    depot = 0
    arcs = [(0, 1), (1, 0), (0, 2), (2, 3), (3, 0)]
    b = make_bitstring_from_arcs(n, arcs)
    Y = _bitstr_to_arcs(b, n=n)

    routes = _arcs_to_routes(Y, depot=depot, K=2)
    assert sorted([sorted(r) for r in routes]) == [[1, 4], [2, 3]]


def test_decode_pipeline_with_monkeypatched_best_bitstring(monkeypatch, capsys):
    """Decode pipeline: fixed best bitstring, verify routes and arc matrix."""
    n = 5
    depot = 0
    K = 2
    arcs = [(0, 1), (1, 0), (0, 2), (2, 3), (3, 0)]
    b = make_bitstring_from_arcs(n, arcs)

    monkeypatch.setattr(dec_mod, "best_bitstring_from_probability", lambda prob, ising: b)
    res = {"probability": {"dummy": 1.0}}
    ising = SimpleNamespace()

    best_raw, routes, Y = decode_from_platform_result(
        res, n=n, vehicle_count=K, ising=ising, depot=depot
    )
    assert best_raw == b
    assert Y.shape == (n, n)
    assert sorted([sorted(r) for r in routes]) == [[1, 4], [2, 3]]

    out = capsys.readouterr().out
    assert "Best Qubit string:" in out


def test_decode_pipeline_with_nonzero_depot(monkeypatch):
    """Decoding with a nonzero depot; uncovered customers are backfilled."""
    n = 5
    depot = 2
    K = 2
    arcs = [(2, 1), (1, 2), (2, 3), (3, 4), (4, 2)]
    b = make_bitstring_from_arcs(n, arcs)
    monkeypatch.setattr(dec_mod, "best_bitstring_from_probability", lambda prob, ising: b)

    res = {"probability": {"y": 1.0}}
    ising = SimpleNamespace()
    best_raw, routes, Y = decode_from_platform_result(
        res, n=n, vehicle_count=K, ising=ising, depot=depot
    )
    assert sorted([sorted(r) for r in routes]) == [[0, 1], [3, 4]]


def test_plot_vrp_solution_with_distance_matrix_smoke(monkeypatch):
    """Smoke test: plotting with a distance matrix and patched decoder/plot."""
    n = 5
    depot = 0
    K = 2
    D = np.array(
        [
            [0, 2, 6, 7, 3],
            [2, 0, 4, 3, 5],
            [6, 4, 0, 6, 2],
            [7, 3, 6, 0, 4],
            [3, 5, 2, 4, 0],
        ],
        dtype=float,
    )

    fake_routes = [[1], [2, 3]]
    fake_Y = np.zeros((n, n), dtype=int)
    monkeypatch.setattr(
        dec_mod,
        "decode_from_platform_result",
        lambda result, n, vehicle_count, ising, depot=0: ("b", fake_routes, fake_Y),
    )

    monkeypatch.setattr(dec_mod, "plot_vrp", lambda *a, **k: None)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=True)

    res_routes = plot_vrp_solution(
        D,
        result={"probability": {"dummy": 1.0}},
        n=n,
        vehicle_count=K,
        depot=depot,
        title="VRP Solution (QAOA)",
        ising=SimpleNamespace(),
        show=True,
    )
    assert res_routes == fake_routes


def test_plot_vrp_solution_with_coordinates_smoke(monkeypatch):
    """Smoke test: plotting with (n,2) coordinates; ensure no GUI and correct return."""
    n = 4
    depot = 0
    K = 1
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    fake_routes = [[1, 2, 3]]
    fake_Y = np.zeros((n, n), dtype=int)
    monkeypatch.setattr(
        dec_mod,
        "decode_from_platform_result",
        lambda result, n, vehicle_count, ising, depot=0: ("b", fake_routes, fake_Y),
    )
    monkeypatch.setattr(dec_mod, "plot_vrp", lambda *a, **k: None)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=True)

    res_routes = plot_vrp_solution(
        coords,
        result={"probability": {"dummy": 1.0}},
        n=n,
        vehicle_count=K,
        depot=depot,
        title="VRP Solution (QAOA)",
        ising=SimpleNamespace(),
        show=True,
    )
    assert res_routes == fake_routes
