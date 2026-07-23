# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
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
from types import SimpleNamespace

import pytest

MODULE_E = "cqlib_algorithm.execution.platform_runner"
ty_mod = importlib.import_module(MODULE_E)
TianYanRunner = ty_mod.TianYanRunner
SubmitResult = ty_mod.SubmitResult


# ---------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------
class FakePlatform:
    """Injectable fake cqlib_tianyan platform for deterministic I/O."""

    def __init__(self):
        self.login_kwargs = None
        self.machine = None
        self.backend = FakeBackend()

    def get_backend(self, machine):
        self.machine = machine
        self.backend.machine = machine
        return self.backend


class FakeBackend:
    """Minimal backend with run/wait and device_config hooks."""

    def __init__(self):
        self.machine = None
        self.submissions = []
        self.device_config_called = False
        self._results = [FakeExecutionResult(probabilities={"0": 0.6, "1": 0.4})]

    def device_config(self):
        self.device_config_called = True
        return "DEVICE_CONFIG"

    def run(self, circuits, shots):
        self.submissions.append({"circuits": list(circuits), "shots": shots})
        return FakeTask(self._results)


class FakeTask:
    """Minimal task handle returned by backend.run."""

    def __init__(self, results):
        self.task_ids = ["Q-123456"]
        self.results = results
        self.wait_calls = []

    def wait(self, timeout_secs, poll_interval_secs):
        self.wait_calls.append(
            {"timeout_secs": timeout_secs, "poll_interval_secs": poll_interval_secs}
        )
        return list(self.results)


class FakeExecutionResult:
    """Minimal execution result with cqlib_tianyan-like fields."""

    def __init__(self, probabilities=None, counts=None):
        self.probabilities = probabilities
        self.counts = counts or {}


class FakeCircuit:
    """Minimal circuit holder for QCIS payload tests."""

    def __init__(self, qcis="ORIGINAL_QCIS"):
        self.qcis = qcis
        self.name = "fake"


def _patch_platform(monkeypatch, platform_instance: FakePlatform):
    """Patch TianyanPlatform.login to return the provided fake instance."""
    class _Platform:
        @staticmethod
        def login(**kwargs):
            assert kwargs["api_key"]
            platform_instance.login_kwargs = dict(kwargs)
            return platform_instance

    monkeypatch.setattr(ty_mod, "TianyanPlatform", _Platform)


def _patch_compile(monkeypatch, returns=None, recorder: dict | None = None):
    """Patch cqlib.compile.compile and optionally record invocation details."""
    if returns is None:
        returns = SimpleNamespace(
            circuit=FakeCircuit("TRANSPILED_QCIS"),
            initial_layout="INIT_LAYOUT",
            steps="COMPILE_STEPS",
        )

    def _fake_compile(circuit, device):
        if recorder is not None:
            recorder["called"] = True
            recorder["circuit"] = circuit
            recorder["device"] = device
        return returns

    monkeypatch.setattr(ty_mod, "compile_circuit", _fake_compile)


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


def test_init_logs_in_and_gets_backend(monkeypatch):
    """Constructor should log in through cqlib_tianyan and select a backend."""
    fp = FakePlatform()
    _patch_platform(monkeypatch, fp)

    r = TianYanRunner(login_key="KEY", machine="tianyan_sw", domain="https://api")
    assert r.machine == "tianyan_sw"
    assert fp.machine == "tianyan_sw"
    assert fp.login_kwargs["api_key"] == "KEY"
    assert fp.login_kwargs["domain"] == "https://api"


def test_run_simulator_no_compile(monkeypatch):
    """Simulator path should submit original QCIS without compilation."""
    fp = FakePlatform()
    _patch_platform(monkeypatch, fp)

    def _compile_should_not_run(circuit, device):
        raise AssertionError("compile should not run for tianyan_sw")

    monkeypatch.setattr(ty_mod, "compile_circuit", _compile_should_not_run)

    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")
    circ = FakeCircuit("ORIGINAL_QCIS")

    submit, single = runner.run_circuit(circ, num_shots=256)

    assert fp.backend.submissions[-1]["circuits"] == ["ORIGINAL_QCIS"]
    assert fp.backend.submissions[-1]["shots"] == 256
    assert isinstance(submit, SubmitResult)
    assert submit.query_id == "Q-123456"
    assert submit.lab_id is None
    assert submit.machine == "tianyan_sw"
    assert submit.num_shots == 256
    assert submit.used_circuit is circ
    assert submit.mapping_virtual_to_final is None
    assert submit.initial_layout is None
    assert submit.swap_mapping is None
    assert single["probability"] == {"0": 0.6, "1": 0.4}


def test_run_hardware_with_compile_inferred(monkeypatch):
    """Hardware path should compile and submit the compiled QCIS."""
    fp = FakePlatform()
    _patch_platform(monkeypatch, fp)
    rec = {}
    _patch_compile(monkeypatch, recorder=rec)

    runner = TianYanRunner(login_key="KEY", machine="tianyan_qpu")
    circ = FakeCircuit("ORIGINAL_QCIS")

    submit, single = runner.run_circuit(circ, num_shots=100)

    assert rec.get("called") is True
    assert rec.get("circuit") is circ
    assert rec.get("device") == "DEVICE_CONFIG"
    assert fp.backend.device_config_called is True
    assert fp.backend.submissions[-1]["circuits"] == ["TRANSPILED_QCIS"]
    assert submit.used_circuit.qcis == "TRANSPILED_QCIS"
    assert submit.mapping_virtual_to_final is None
    assert submit.initial_layout == "INIT_LAYOUT"
    assert submit.swap_mapping == "COMPILE_STEPS"
    assert single["probability"] == {"0": 0.6, "1": 0.4}


def test_lab_metadata_is_passthrough(monkeypatch):
    """Lab and experiment arguments are preserved as metadata for callers."""
    fp = FakePlatform()
    _patch_platform(monkeypatch, fp)
    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")

    submit, _ = runner.run_circuit(
        FakeCircuit(),
        exp_name="exp-name",
        lab_id=42,
    )
    assert submit.exp_name == "exp-name"
    assert submit.lab_id == 42


def test_run_reverses_result_bitstrings(monkeypatch):
    """Platform results are converted from Q0-right to Q0-left bitstrings."""
    fp = FakePlatform()
    fp.backend._results = [
        FakeExecutionResult(probabilities={"01": 0.75, "10": 0.25}, counts={"01": 3})
    ]
    _patch_platform(monkeypatch, fp)
    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")

    _, single = runner.run_circuit(FakeCircuit(), num_shots=4)

    assert single["probability"] == {"10": 0.75, "01": 0.25}
    assert single["counts"] == {"10": 3}


def test_run_raises_on_empty_result(monkeypatch):
    """Empty platform result list should raise RuntimeError."""
    fp = FakePlatform()
    fp.backend._results = []
    _patch_platform(monkeypatch, fp)
    runner = TianYanRunner(login_key="KEY", machine="tianyan_sw")

    with pytest.raises(RuntimeError):
        runner.run_circuit(FakeCircuit())


def test_print_result_calls_draw(monkeypatch, capsys):
    """print_result should call draw_probability and print key metadata."""
    fp = FakePlatform()
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
