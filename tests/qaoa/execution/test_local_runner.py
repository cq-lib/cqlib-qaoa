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

"""Unit tests for LocalRunner (local execution path)."""

import importlib
import re
from types import SimpleNamespace

import pytest

MODULE_E = "cqlib_algorithm.execution.local_runner"
lr_mod = importlib.import_module(MODULE_E)
LocalRunner = lr_mod.LocalRunner
SubmitResult = lr_mod.SubmitResult


# ---------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------
class FakeSimulator:
    """Injectable fake StatevectorSimulator to capture shots and control counts."""

    def __init__(self, circ):
        self.circ = circ
        self.last_shots = None
        self._counts = {"01": 30, "10": 70}

    def sample(self, shots: int):
        """Return a copy of the counts and record the requested shot count."""
        self.last_shots = shots
        return dict(self._counts)


def _patch_simulator(monkeypatch, counts=None):
    """Patch StatevectorSimulator with FakeSimulator and optional custom counts."""
    def _factory(circ):
        sim = FakeSimulator(circ)
        if counts is not None:
            sim._counts = dict(counts)
        return sim

    monkeypatch.setattr(lr_mod, "StatevectorSimulator", _factory)


def _patch_draw_probability(monkeypatch, bucket: dict):
    """Patch draw_probability to capture its invocation and arguments."""
    def _fake_draw(probs, title, topk):
        bucket["called"] = True
        bucket["probs"] = probs
        bucket["title"] = title
        bucket["topk"] = topk

    monkeypatch.setattr(lr_mod, "draw_probability", _fake_draw)


# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------
def test_infer_num_qubits_prefers_num_qubits():
    """_infer_num_qubits should return circ.num_qubits when present."""
    lr = LocalRunner()
    circ = SimpleNamespace(num_qubits=5, qubits=[0, 1, 2])
    assert lr._infer_num_qubits(circ) == 5


def test_infer_num_qubits_falls_back_to_qubits_len():
    """_infer_num_qubits should fall back to len(circ.qubits) if needed."""
    lr = LocalRunner()
    circ = SimpleNamespace(qubits=[0, 1, 2, 3])
    assert lr._infer_num_qubits(circ) == 4


def test_identity_mapping_returns_i_to_i():
    """_identity_mapping should map i -> i for the specified size."""
    lr = LocalRunner()
    mp = lr._identity_mapping(4)
    assert mp == {0: 0, 1: 1, 2: 2, 3: 3}


def test_run_circuit_basic_probabilities_and_reverse_sort(monkeypatch):
    """Verify bitstring reversal, normalization, and descending probability order."""
    _patch_simulator(monkeypatch, counts={"01": 30, "10": 70})

    lr = LocalRunner()
    circ = SimpleNamespace() 
    submit, result = lr.run_circuit(circ, num_shots=100)

    assert isinstance(submit, SubmitResult)
    assert submit.num_shots == 100
    assert re.fullmatch(r"\d{14}", submit.query_id) 

    probs = result["probability"]
    assert list(probs.keys()) == ["01", "10"]
    assert probs["01"] == pytest.approx(0.7, abs=1e-12)
    assert probs["10"] == pytest.approx(0.3, abs=1e-12)


def test_run_circuit_passes_shots_to_simulator(monkeypatch):
    """Ensure the requested shot count is passed through to simulator.sample()."""
    captured = {}

    class _Sim(FakeSimulator):
        def sample(self, shots: int):
            captured["shots"] = shots
            return super().sample(shots)

    def _factory(circ):
        return _Sim(circ)

    monkeypatch.setattr(lr_mod, "StatevectorSimulator", _factory)

    lr = LocalRunner()
    _ = lr.run_circuit(SimpleNamespace(), num_shots=256)
    assert captured["shots"] == 256


def test_run_circuit_empty_counts_safe(monkeypatch):
    """LocalRunner should handle empty simulator counts without raising."""
    _patch_simulator(monkeypatch, counts={})
    lr = LocalRunner()
    _, result = lr.run_circuit(SimpleNamespace(), num_shots=10)
    assert result["probability"] == {}


def test_print_result_calls_draw_and_prints(monkeypatch, capsys):
    """print_result should call draw_probability and print key sections."""
    _patch_simulator(monkeypatch, counts={"00": 5, "11": 5})
    lr = LocalRunner()
    submit, result = lr.run_circuit(SimpleNamespace(), num_shots=10)

    bucket = {}
    _patch_draw_probability(monkeypatch, bucket)

    lr.print_result(submit, result, topk=8)
    out = capsys.readouterr().out

    assert "========== [ Experiment Information ] ==========" in out
    assert "Task ID" in out and submit.query_id in out
    assert "Shots" in out and str(submit.num_shots) in out
    assert "========== [ Measurement Results ] ==========" in out
    assert "probability" in out 

    assert bucket.get("called") is True
    assert bucket.get("probs") == result["probability"]
    assert bucket.get("topk") == 8
    assert "TaskID:" in bucket.get("title", "")
