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

"""QCIS emission tests for QAOA circuits."""

import importlib
import re

import pytest
from cqlib import Circuit
from cqlib.ir import qcis

MODULE_PATH = "cqlib_qaoa.ansatz.qaoa_ansatz"
ansatz = importlib.import_module(MODULE_PATH)

# ----------------------------- Normalization utils -----------------------------
def _normalize_floats(s: str, ndigits: int = 6) -> str:
    """Round all floats in a string to a fixed precision."""
    def _round(m):
        return f"{float(m.group(0)):.{ndigits}f}"
    return re.sub(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", _round, s)

def _normalize_qcis(s: str) -> str:
    """Normalize QCIS text."""
    lines = []
    for raw in s.strip().splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        line = re.sub(r"\s+", " ", line.strip())
        lines.append(line)
    s2 = "\n".join(lines)
    s2 = _normalize_floats(s2, ndigits=6)
    return s2

# --------------------------------- Fixtures -----------------------------------
@pytest.fixture
def expected_qcis_text() -> str:
    """This is the expected QCIS."""
    return _normalize_qcis(
        """
        H Q0
        H Q1
        H Q2
        H Q3

        CX Q0 Q1
        RZ Q1 0.8
        CX Q0 Q1

        CX Q0 Q3
        RZ Q3 0.8
        CX Q0 Q3

        CX Q1 Q2
        RZ Q2 0.8
        CX Q1 Q2

        CX Q2 Q3
        RZ Q3 0.8
        CX Q2 Q3

        RX Q0 0.4
        RX Q1 0.4
        RX Q2 0.4
        RX Q3 0.4

        M Q0
        M Q1
        M Q2
        M Q3
        """
    )

# ------------------------------- Helper builders -------------------------------
def _ising_from_paulis_example():
    """
    Example Ising:
      Paulis = ['IIZZ', 'ZIIZ', 'IZZI', 'ZZII'], coeffs = [0.5]*4, offset = -2.0
    This corresponds to ZZ couplings on edges:
      (0,1), (0,3), (1,2), (2,3)  — each with weight 0.5
    """
    h = {}
    J = {
        (0, 1): 0.5,  # IIZZ
        (0, 3): 0.5,  # ZIIZ
        (1, 2): 0.5,  # IZZI
        (2, 3): 0.5,  # ZZII
    }
    return h, J

# ------------------------------------ Tests ------------------------------------
def test_qcis_emit_snapshot_matches_expected(expected_qcis_text):
    """Build circuit from the given Ising, emit QCIS, and compare with expected snapshot."""
    h, J = _ising_from_paulis_example()
    circ = ansatz.build_qaoa_circuit(
        n=4, h=h, J=J, reps=1, mixer_operator="x", insert_barriers=False, name="qaoa_ising_4"
    )

    qcis_text = qcis.dumps(circ)
    got = _normalize_qcis(qcis_text)
    exp = expected_qcis_text

    assert got == exp, f"QCIS mismatch.\n--- got ---\n{got}\n--- exp ---\n{exp}"

def test_qcis_round_trip_is_stable(expected_qcis_text):
    """Test qcis round trip is stable."""
    circ2 = qcis.loads(expected_qcis_text)
    assert isinstance(circ2, Circuit)
    qcis2 = _normalize_qcis(qcis.dumps(circ2))
    assert qcis2 == expected_qcis_text

def test_qcis_style_whitelist_and_measure_tail(expected_qcis_text):
    """Basic style checks: op whitelist and no gates after measurements."""
    ALLOWED = {"H", "RZ", "RX", "CX", "M"}

    def _op(line: str) -> str | None:
        m = re.match(r"^\s*([A-Z]+)\b", line)
        return m.group(1) if m else None

    seen_measure = False
    for line in expected_qcis_text.splitlines():
        op = _op(line)
        if not op:
            continue
        assert op in ALLOWED, f"Illegal op in QCIS: {op}"
        if op == "M":
            seen_measure = True
        if seen_measure:
            assert op == "M", "No gates allowed after the first measurement"
