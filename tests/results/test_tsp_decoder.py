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

"""Unit tests for TSP decoding and plotting utilities."""

import importlib
import json
import numpy as np
import pytest
from types import SimpleNamespace

MODULE_DEC = "cqlib_qaoa.results.tsp_decoder"
dec_mod = importlib.import_module(MODULE_DEC)

best_bitstring_from_probability = dec_mod.best_bitstring_from_probability
decode_from_platform_result = dec_mod.decode_from_platform_result
plot_tsp_solution = dec_mod.plot_tsp_solution
_bitstr_to_assignment = dec_mod._bitstr_to_assignment
_assignment_to_tour = dec_mod._assignment_to_tour


# ---------------------------------------------------------------------
# best_bitstring_from_probability
# ---------------------------------------------------------------------
def test_best_bitstring_selects_min_energy_from_dict(monkeypatch):
    """Select the bitstring with minimum energy from a dict payload."""
    monkeypatch.setattr(dec_mod, "parse_probability", lambda p: p)

    energy = {"0000": 0.0, "0110": -2.0, "1111": 5.0}
    monkeypatch.setattr(dec_mod, "energy_of_bitstring", lambda ising, b: energy[b])

    ising = SimpleNamespace()
    probs = {"0000": 0.2, "0110": 0.1, "1111": 0.7}
    b = best_bitstring_from_probability(probs, ising)
    assert b == "0110"


def test_best_bitstring_accepts_json_string(monkeypatch):
    """Accept a JSON string and ensure parse_probability is invoked."""
    captured = {}

    def fake_parse(p):
        captured["type"] = type(p)
        return {"0": 0.4, "1": 0.6}

    monkeypatch.setattr(dec_mod, "parse_probability", fake_parse)
    monkeypatch.setattr(dec_mod, "energy_of_bitstring", lambda ising, b: {"0": 1.0, "1": -1.0}[b])

    payload = json.dumps({"0": 0.4, "1": 0.6})
    ising = SimpleNamespace()
    b = best_bitstring_from_probability(payload, ising)
    assert b == "1"
    assert captured["type"] is str


# ---------------------------------------------------------------------
# _bitstr_to_assignment
# ---------------------------------------------------------------------
def test_bitstr_to_assignment_basic_and_trim_right():
    """Convert a bitstring to an n×n assignment and trim extra left bits."""
    n = 3
    bitstr = "xx" + "100010001" 
    X = _bitstr_to_assignment(bitstr, n)
    assert X == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_bitstr_to_assignment_too_short_raises():
    """Raise ValueError if bitstring is shorter than n*n."""
    with pytest.raises(ValueError):
        _ = _bitstr_to_assignment("10101", n=3) 


# ---------------------------------------------------------------------
# _assignment_to_tour
# ---------------------------------------------------------------------
def test_assignment_to_tour_unique_one_hot():
    """Extract tour from a valid one-hot assignment matrix."""
    X = [
        [1, 0, 0], 
        [0, 1, 0],
        [0, 0, 1], 
    ]
    tour = _assignment_to_tour(X)
    assert tour == [0, 1, 2]


def test_assignment_to_tour_multi_and_zero_columns_repair():
    """Repair tour when a column has multiple ones or is all zeros."""
    X = [
        [1, 0, 0], 
        [1, 0, 0], 
        [0, 0, 1], 
    ]
    tour = _assignment_to_tour(X)
    assert tour == [0, 1, 2]


# ---------------------------------------------------------------------
# decode_from_platform_result
# ---------------------------------------------------------------------
def test_decode_from_platform_result_pipeline(monkeypatch, capsys):
    """Run the decode pipeline with a fixed diagonal one-hot best bitstring."""
    monkeypatch.setattr(
        dec_mod, "best_bitstring_from_probability", lambda prob, ising: "100010001"
    )
    res = {"probability": {"100010001": 1.0}}
    ising = SimpleNamespace()

    best_raw, tour, X = decode_from_platform_result(res, n=3, ising=ising)
    assert best_raw == "100010001"
    assert tour == [0, 1, 2]
    assert X == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    out = capsys.readouterr().out
    assert "Best Qubit string" in out


# ---------------------------------------------------------------------
# plot_tsp_solution
# ---------------------------------------------------------------------
def test_plot_tsp_solution_with_distance_matrix_calls_plot_and_show(monkeypatch, capsys):
    """Plot with a distance matrix and verify plot/show interactions."""
    monkeypatch.setattr(
        dec_mod,
        "decode_from_platform_result",
        lambda result, n, ising: ("100", [0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )

    plotted = {}

    def fake_plot_tsp(distance_matrix, *, tour, title):
        plotted["tour"] = list(tour)
        plotted["title"] = title
        plotted["shape"] = np.asarray(distance_matrix).shape

    monkeypatch.setattr(dec_mod, "plot_tsp", fake_plot_tsp)

    show_called = {}
    monkeypatch.setattr(dec_mod.plt, "show", lambda: show_called.setdefault("called", True))

    dm = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    tour = plot_tsp_solution(
        dm, result={"probability": {}}, title="TSP Plot", show=True, ising=SimpleNamespace()
    )

    assert tour == [0, 1, 2]
    assert plotted["tour"] == [0, 1, 2]
    assert plotted["title"] == "TSP Plot"
    assert plotted["shape"] == (3, 3)
    assert show_called.get("called", False) is True
    assert "Best TSP value" in capsys.readouterr().out


def test_plot_tsp_solution_with_coords_matrix(monkeypatch):
    """Plot with coordinate matrix input (Nx2) and ensure no errors."""
    monkeypatch.setattr(
        dec_mod,
        "decode_from_platform_result",
        lambda result, n, ising: ("100", [0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    called = {}
    monkeypatch.setattr(
        dec_mod, "plot_tsp", lambda distance_matrix, *, tour, title: called.setdefault("ok", True)
    )
    monkeypatch.setattr(dec_mod.plt, "show", lambda: None)

    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    tour = plot_tsp_solution(
        coords, result={"probability": {}}, show=False, ising=SimpleNamespace()
    )
    assert tour == [0, 1, 2]
    assert called.get("ok", False) is True 


def test_plot_tsp_solution_invalid_shape_raises(monkeypatch):
    """Raise ValueError when distance/coords matrix shape is invalid."""
    monkeypatch.setattr(
        dec_mod, "decode_from_platform_result", lambda result, n, ising: ("10", [0, 1], [[1, 0], [0, 1]])
    )
    with pytest.raises(ValueError):
        _ = plot_tsp_solution(np.ones((3, 4)), result={"probability": {}}, ising=SimpleNamespace())
