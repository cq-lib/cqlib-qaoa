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

"""Unit tests for QAOA ansatz circuit construction."""

import importlib
import math

import pytest

MODULE_PATH = "cqlib_qaoa.ansatz.qaoa_ansatz"
ansatz = importlib.import_module(MODULE_PATH)


class FakeCircuit:
    """Lightweight stand-in for the real Circuit to record operations."""

    def __init__(self, qubits=None, name=None):
        self.qubits = list(qubits or [])
        self.name = name or "fake"
        self.ops = []
        self._barrier_calls = 0
        self._measured = False
        self.merged = []

    def h(self, q):
        self.ops.append(("H", q))

    def rz(self, q, theta):
        self.ops.append(("RZ", q, theta))

    def rx(self, q, theta):
        self.ops.append(("RX", q, theta))

    def barrier(self, qs):
        self._barrier_calls += 1
        self.ops.append(("BARRIER", tuple(qs)))

    def measure(self, q):
        self._measured = True
        self.ops.append(("M", q))

    def compose(self, other):
        self.merged.append(getattr(other, "name", "unnamed"))
        self.ops.append(("COMPOSE", getattr(other, "name", "unnamed")))


def _patch_minimal_env(monkeypatch):
    """Patch module symbols to use the fake circuit and fake two-qubit rotations."""
    monkeypatch.setattr(ansatz, "Circuit", FakeCircuit)

    def fake_rzz(c, i, j, theta):
        c.ops.append(("RZZ", i, j, theta))

    def fake_rxx(c, i, j, theta):
        c.ops.append(("RXX", i, j, theta))

    def fake_ryy(c, i, j, theta):
        c.ops.append(("RYY", i, j, theta))

    monkeypatch.setattr(ansatz, "rzz_via_cnot", fake_rzz)
    monkeypatch.setattr(ansatz, "rxx_via_cnot", fake_rxx)
    monkeypatch.setattr(ansatz, "ryy_via_cnot", fake_ryy)


def test_build_qaoa_circuit_defaults_and_structure(monkeypatch):
    """Verify default angles, layer ordering, and metadata for X-mixer."""
    _patch_minimal_env(monkeypatch)

    n = 3
    h = {0: 0.5, 2: -0.25}
    J = {(0, 1): 1.0, (1, 2): 2.0}
    circ = ansatz.build_qaoa_circuit(
        n=n, h=h, J=J, reps=2, mixer_operator="x", insert_barriers=True, name="qaoa"
    )

    assert [op for op in circ.ops if op[0] == "H"] == [("H", 0), ("H", 1), ("H", 2)]

    rz_ops = [op for op in circ.ops if op[0] == "RZ"]
    assert len(rz_ops) == 2 * 2 
    rx_ops = [op for op in circ.ops if op[0] == "RX"]
    assert len(rx_ops) == 2 * n

    rzz_ops = [op for op in circ.ops if op[0] == "RZZ"]
    assert len(rzz_ops) == 2 * len(J)

    assert circ._measured is True
    assert [op for op in circ.ops if op[0] == "M"] == [("M", 0), ("M", 1), ("M", 2)]

    assert hasattr(circ, "_qaoa_meta")
    meta = getattr(circ, "_qaoa_meta")
    assert meta["n"] == n and meta["reps"] == 2
    assert meta["mixer"] == "x"
    assert meta["initial_state"] == "|+>^n"
    assert meta["barriers"] is True


def test_build_qaoa_circuit_xy_mixer(monkeypatch):
    """Verify XY mixer applies RXX and RYY with angle 2β on a ring topology."""
    _patch_minimal_env(monkeypatch)

    n = 4
    circ = ansatz.build_qaoa_circuit(
        n=n, h={}, J={}, reps=1, mixer_operator="xy", insert_barriers=False
    )

    rxx = [op for op in circ.ops if op[0] == "RXX"]
    ryy = [op for op in circ.ops if op[0] == "RYY"]
    assert len(rxx) == n and len(ryy) == n
    for (_, i, j, th1), (_, i2, j2, th2) in zip(rxx, ryy):
        assert (i, j) == (i2, j2)
        assert math.isclose(th1, 0.4, rel_tol=1e-12)
        assert math.isclose(th2, 0.4, rel_tol=1e-12)


def test_build_qaoa_circuit_custom_init_callable(monkeypatch):
    """Ensure a callable initial state is invoked and its effects are applied."""
    _patch_minimal_env(monkeypatch)

    applied = {"times": 0}

    def my_init(c, qubits):
        applied["times"] += 1
        for q in qubits:
            c.h(q)

    circ = ansatz.build_qaoa_circuit(
        n=2, h={}, J={}, reps=1, initial_state=my_init, mixer_operator="x"
    )

    assert applied["times"] == 1
    h_ops = [op for op in circ.ops if op[0] == "H"]
    assert len(h_ops) == 2


def test_build_qaoa_circuit_initial_state_is_circuit_monkeypatched(monkeypatch):
    """Ensure Circuit initial_state is composed into the target circuit."""
    _patch_minimal_env(monkeypatch)

    init = FakeCircuit(qubits=[0, 1], name="init_circ")
    circ = ansatz.build_qaoa_circuit(
        n=2, h={}, J={}, reps=1, initial_state=init, mixer_operator="x"
    )
    assert ("COMPOSE", "init_circ") in circ.ops


def test_build_qaoa_circuit_raises_on_invalid_selectors(monkeypatch):
    """Validate error handling for invalid initial_state and mixer_operator."""
    _patch_minimal_env(monkeypatch)

    with pytest.raises(ValueError):
        ansatz.build_qaoa_circuit(n=1, h={}, J={}, reps=1, initial_state="bad")
    with pytest.raises(ValueError):
        ansatz.build_qaoa_circuit(n=1, h={}, J={}, reps=1, mixer_operator="bad")


def test_build_qaoa_circuit_angle_length_assert(monkeypatch):
    """Assert that mismatched gamma/beta lengths raise an error when reps > 1."""
    _patch_minimal_env(monkeypatch)
    with pytest.raises(AssertionError):
        ansatz.build_qaoa_circuit(
            n=2, h={}, J={}, reps=2, gammas=[0.1], betas=[0.2]
        )
