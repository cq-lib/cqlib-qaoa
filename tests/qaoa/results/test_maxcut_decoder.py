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

"""Unit tests for MaxCut decoding and plotting utilities."""

import importlib
import json
from types import SimpleNamespace
import pytest

MODULE_DEC = "cqlib_algorithm.results.maxcut_decoder"
dec_mod = importlib.import_module(MODULE_DEC)

best_bitstring_from_probability = dec_mod.best_bitstring_from_probability
decode_from_platform_result = dec_mod.decode_from_platform_result
plot_maxcut_solution = dec_mod.plot_maxcut_solution


# ---------------------------------------------------------------------
# best_bitstring_from_probability
# ---------------------------------------------------------------------
def test_best_bitstring_selects_min_energy_from_dict(monkeypatch):
    """Select the bitstring with the minimum energy from a dict input."""
    monkeypatch.setattr(dec_mod, "parse_probability", lambda p: p)

    energy_table = {"00": 1.0, "01": -0.5, "10": 0.0, "11": 2.0}

    def fake_energy(ising, b):
        return energy_table[b]

    monkeypatch.setattr(dec_mod, "energy_of_bitstring", fake_energy)

    probs = {"00": 0.1, "01": 0.2, "10": 0.7, "11": 0.0}
    ising = SimpleNamespace()
    b = best_bitstring_from_probability(probs, ising)
    assert b == "01" 


def test_best_bitstring_accepts_json_string_and_calls_parse(monkeypatch):
    """Accept JSON string payloads and ensure parse_probability is invoked."""
    called = {}

    def fake_parse(p):
        called["type"] = type(p)
        return {"000": 0.4, "111": 0.6}

    monkeypatch.setattr(dec_mod, "parse_probability", fake_parse)

    def fake_energy(ising, b):
        return {"000": 5.0, "111": -1.0}[b]

    monkeypatch.setattr(dec_mod, "energy_of_bitstring", fake_energy)

    payload = json.dumps({"000": 0.4, "111": 0.6})
    ising = SimpleNamespace()
    b = best_bitstring_from_probability(payload, ising)
    assert b == "111"
    assert called["type"] is str


# ---------------------------------------------------------------------
# decode_from_platform_result
# ---------------------------------------------------------------------
def test_decode_from_platform_result_partition_choose_ones_true(monkeypatch, capsys):
    """Decode result with choose_ones=True and check partition indices."""
    monkeypatch.setattr(dec_mod, "best_bitstring_from_probability", lambda prob, ising: "1010")
    res = {"probability": {"1010": 1.0}}
    ising = SimpleNamespace()
    best_raw, best_std, part = decode_from_platform_result(res, n=4, choose_ones=True, ising=ising)
    assert best_raw == "1010" and best_std == "1010"
    assert part == [0, 2]

    out = capsys.readouterr().out
    assert "Best Qubit string" in out


def test_decode_from_platform_result_partition_choose_ones_false(monkeypatch):
    """Decode result with choose_ones=False and check complement partition."""
    monkeypatch.setattr(dec_mod, "best_bitstring_from_probability", lambda prob, ising: "1010")
    res = {"probability": {"1010": 1.0}}
    ising = SimpleNamespace()
    best_raw, best_std, part = decode_from_platform_result(res, n=4, choose_ones=False, ising=ising)
    assert part == [1, 3] 


# ---------------------------------------------------------------------
# plot_maxcut_solution
# ---------------------------------------------------------------------
def test_plot_maxcut_solution_calls_plot_and_show(monkeypatch, capsys):
    """Plot a decoded MaxCut solution and verify plotting/show calls."""
    def fake_decode(result, n, choose_ones, ising):
        return "1010", "1010", [0, 2]

    monkeypatch.setattr(dec_mod, "decode_from_platform_result", fake_decode)

    plotted = {}

    def fake_plot_maxcut(*, n, weights, partition, pos, title):
        plotted["n"] = n
        plotted["weights"] = dict(weights)
        plotted["partition"] = list(partition)
        plotted["title"] = title

    monkeypatch.setattr(dec_mod, "plot_maxcut", fake_plot_maxcut)

    show_called = {}
    monkeypatch.setattr(dec_mod.plt, "show", lambda: show_called.setdefault("called", True))

    weights = {(0, 1): 1.0, (0, 2): 2.0, (1, 3): 3.0, (2, 3): 4.0}
    ising = SimpleNamespace()
    part = plot_maxcut_solution(
        n=4,
        weights=weights,
        result={"probability": {"1010": 1.0}},
        choose_ones=True,
        title="MyCut",
        pos=None,
        show=True,
        ising=ising,
    )

    assert part == [0, 2]
    assert plotted["n"] == 4
    assert plotted["partition"] == [0, 2]
    assert plotted["title"] == "MyCut"
    assert show_called.get("called", False) is True

    out = capsys.readouterr().out
    assert "Max-Cut value" in out


def test_plot_maxcut_solution_show_false_disables_show(monkeypatch):
    """Ensure plt.show is not called when show=False."""
    monkeypatch.setattr(
        dec_mod,
        "decode_from_platform_result",
        lambda result, n, choose_ones, ising: ("0101", "0101", [1, 3]),
    )
    monkeypatch.setattr(dec_mod, "plot_maxcut", lambda **kwargs: None)

    called = {"show": False}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(dec_mod.plt, "show", fake_show)

    weights = {(0, 1): 1.0, (2, 3): 1.0}
    ising = SimpleNamespace()
    part = plot_maxcut_solution(
        n=4,
        weights=weights,
        result={"probability": {"0101": 1.0}},
        choose_ones=True,
        title="no_show",
        show=False,
        ising=ising,
    )

    assert part == [1, 3]
    assert called["show"] is False 
