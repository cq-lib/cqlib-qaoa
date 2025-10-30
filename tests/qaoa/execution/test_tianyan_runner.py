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

"""Unit tests for TianYanRunner (platform execution path)."""

import importlib
import pytest

MODULE_E = "cqlib_algorithm.execution.platform_runner"
ty_mod = importlib.import_module(MODULE_E)
TianYanRunner = ty_mod.TianYanRunner
SubmitResult = ty_mod.SubmitResult


# ---------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------
class FakePlatform:
    """Injectable fake TianYanPlatform for deterministic I/O behavior."""

    def __init__(self, login_key):
        self.login_key = login_key
        self.machine = None
        self.created_labs = []
        self.submissions = []
        self.queries = []
        self._query_result = [{"probability": {"0": 0.6, "1": 0.4}}]

    def set_machine(self, machine):
        self.machine = machine

    def create_lab(self, name, remark):
        """Create a lab and return a fixed lab_id for testing."""
        self.created_labs.append((name, remark))
        return 42 

    def submit_job(self, circuit, exp_name, lab_id, num_shots):
        """Record a job submission and return a fixed query_id."""
        self.submissions.append(
            {"circuit": circuit, "exp_name": exp_name, "lab_id": lab_id, "shots": num_shots}
        )
        return "Q-123456" 

    def query_experiment(self, query_id, max_wait_time, sleep_time):
        """Record query parameters and return the configured query result list."""
        self.queries.append(
            {"query_id": query_id, "max_wait_time": max_wait_time, "sleep_time": sleep_time}
        )
        return list(self._query_result)


class FakeCircuit:
    """Minimal circuit holder for QCIS payload tests."""

    def __init__(self, qcis="ORIGINAL_QCIS"):
        self.qcis = qcis
        self.name = "fake"


def _patch_platform(monkeypatch, platform_instance: FakePlatform):
    """Patch TianYanPlatform factory to return the provided fake instance."""
    def _factory(login_key):
        assert login_key 
        return platform_instance

    monkeypatch.setattr(ty_mod, "TianYanPlatform", _factory)


def _patch_transpile(monkeypatch, returns=None, recorder: dict | None = None):
    """Patch QCIS transpilation and optionally record invocation details."""
    if returns is None:
        tcirc = FakeCircuit("TRANSPILED_QCIS")
        returns = (tcirc, "INIT_LAYOUT", "SWAP_MAPPING", {0: 1, 1: 0})

    def _fake_transpile(qcis_text, platform):
        if recorder is not None:
            recorder["called"] = True
            recorder["arg_qcis"] = qcis_text
            recorder["machine"] = getattr(platform, "machine", None)
        return returns

    monkeypatch.setattr(ty_mod, "transpile_qcis", _fake_transpile)


def _patch_draw(monkeypatch, bucket: dict):
    """Patch probability plotting to capture arguments and call site."""
    def _fake_draw(probs, title, topk):
        bucket["called"] = True
        bucket["probs"] = probs
        bucket["title"] = title
        bucket["topk"] = topk

    monkeypatch.setattr(ty_mod, "draw_probability", _fake_draw)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_init_requires_login_key():
    """TianYanRunner must validate that a non-empty login_key is provided."""
    with pytest.raises(ValueError):
        _ = TianYanRunner(login_key=None)
    with pytest.raises(ValueError):
        _ = TianYanRunner(login_key="")


def test_init_sets_machine(monkeypatch):
    """Constructor should set platform machine selection."""
    fp = FakePlatform("KEY")
    _patch_platform(monkeypatch, fp)

    r = TianYanRunner(login_key="KEY", machine="tianyan_sw")
    assert r.machine == "tianyan_sw"
    assert fp.machine == "tianyan_sw"


def test_run_simulator_no_transpile(monkeypatch):
    """Simulator path should submit original QCIS without transpilation."""
    fp = FakePlatform("KEY")
    _patch_platform(monkeypatch, fp)
    _patch_transpile(monkeypatch, recorder={}) 

    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw") 
    circ = FakeCircuit("ORIGINAL_QCIS")

    submit, single = runner.run_circuit(circ, num_shots=256)

    assert fp.created_labs and isinstance(fp.created_labs[0][0], str)
    assert fp.submissions[-1]["circuit"] == "ORIGINAL_QCIS"
    assert fp.submissions[-1]["shots"] == 256
    assert isinstance(submit, SubmitResult)
    assert submit.lab_id == 42
    assert submit.machine == "tianyan_sw"
    assert submit.num_shots == 256
    assert submit.used_circuit is circ
    assert submit.mapping_virtual_to_final is None
    assert submit.initial_layout is None
    assert submit.swap_mapping is None
    assert single == {"probability": {"0": 0.6, "1": 0.4}}


def test_run_hardware_with_transpile_inferred(monkeypatch):
    """Hardware path should transpile and submit the transpiled QCIS."""
    fp = FakePlatform("KEY")
    _patch_platform(monkeypatch, fp)
    rec = {}
    _patch_transpile(monkeypatch, recorder=rec)

    runner = TianYanRunner(login_key="KEY", machine="tianyan_qpu") 
    circ = FakeCircuit("ORIGINAL_QCIS")

    submit, single = runner.run_circuit(circ, num_shots=100)

    assert rec.get("called") is True
    assert rec.get("arg_qcis") == "ORIGINAL_QCIS"
    assert fp.submissions[-1]["circuit"] == "TRANSPILED_QCIS"
    assert submit.used_circuit.qcis == "TRANSPILED_QCIS"
    assert submit.mapping_virtual_to_final == {0: 1, 1: 0}
    assert submit.initial_layout == "INIT_LAYOUT"
    assert submit.swap_mapping == "SWAP_MAPPING"
    assert single == {"probability": {"0": 0.6, "1": 0.4}}


def test_run_requires_lab_if_disabled(monkeypatch):
    """If lab auto-creation is disabled, missing lab should raise ValueError."""
    fp = FakePlatform("KEY")
    _patch_platform(monkeypatch, fp)
    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")

    with pytest.raises(ValueError):
        runner.run_circuit(FakeCircuit(), create_lab_if_missing=False)


def test_run_raises_on_empty_result(monkeypatch):
    """Empty platform result list should raise RuntimeError."""
    fp = FakePlatform("KEY")
    fp._query_result = []
    _patch_platform(monkeypatch, fp)
    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")

    with pytest.raises(RuntimeError):
        runner.run_circuit(FakeCircuit())


def test_print_result_calls_draw(monkeypatch, capsys):
    """print_result should call draw_probability and print key metadata."""
    fp = FakePlatform("KEY")
    _patch_platform(monkeypatch, fp)

    bucket = {}
    _patch_draw(monkeypatch, bucket)

    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")
    submit, result = runner.run_circuit(FakeCircuit(), num_shots=32)

    runner.print_result(submit, result, topk=5)
    out = capsys.readouterr().out

    assert "========== [ Experiment Information ] ==========" in out
    assert "Lab ID" in out and str(submit.lab_id) in out
    assert "Task ID" in out and submit.query_id in out
    assert "Machine" in out and submit.machine in out
    assert "Shots" in out and str(submit.num_shots) in out
    assert "Mapping" in out
    assert "========== [ Measurement Results ] ==========" in out
    assert "probability" in out
    assert bucket.get("called") is True
    assert bucket.get("probs") == result["probability"]
    assert bucket.get("topk") == 5
    assert "TaskID:" in bucket.get("title", "")
