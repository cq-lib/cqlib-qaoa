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

"""Pytest configuration and cqlib 2.x doubles used when the native extension is absent."""
from __future__ import annotations

import re
import sys
import types

import matplotlib
import pytest


def _cqlib2_available() -> bool:
    """Return True only when the cqlib 2.x Circuit / IR / simulator surface exists."""
    try:
        from cqlib import Circuit
        from cqlib.compile import compile as _compile  # noqa: F401
        from cqlib.ir import qcis  # noqa: F401
        from cqlib.qis.state import Statevector  # noqa: F401

        Circuit([0, 1])
        return True
    except Exception:
        return False


if not _cqlib2_available():
    class Circuit:
        """Minimal Circuit that accepts a qubit list, matching cqlib 2.x."""

        def __init__(self, qubits=None, name=None):
            if qubits is None:
                self.qubits = []
            elif isinstance(qubits, int):
                self.qubits = list(range(int(qubits)))
            else:
                self.qubits = list(qubits)
            self.num_qubits = len(self.qubits)
            self.name = name
            self.operations = []

        def _add(self, name, *args):
            self.operations.append((name, *args))

        def h(self, q):
            self._add("h", q)

        def rz(self, q, theta):
            self._add("rz", q, theta)

        def rx(self, q, theta):
            self._add("rx", q, theta)

        def cx(self, control, target):
            self._add("cx", control, target)

        def barrier(self, qubits):
            self._add("barrier", tuple(qubits))

        def measure(self, q):
            self._add("measure", q)

        def compose(self, other):
            self.operations.extend(getattr(other, "operations", []))
            return self

        def append(self, op):
            self.operations.append(op)

        def extend(self, other):
            return self.compose(other)

    def _qcis_dumps(circuit):
        lines = []
        for operation in getattr(circuit, "operations", []):
            name, *args = operation
            if name == "h":
                lines.append(f"H Q{args[0]}")
            elif name == "rz":
                lines.append(f"RZ Q{args[0]} {args[1]}")
            elif name == "rx":
                lines.append(f"RX Q{args[0]} {args[1]}")
            elif name == "cx":
                lines.append(f"CX Q{args[0]} Q{args[1]}")
            elif name == "measure":
                lines.append(f"M Q{args[0]}")
            elif name == "barrier":
                qubits = args[0] if args and isinstance(args[0], (list, tuple)) else args
                lines.append("BARRIER " + " ".join(f"Q{q}" for q in qubits))
            else:
                raise ValueError(f"unsupported fake QCIS operation {operation!r}")
        return "\n".join(lines) + ("\n" if lines else "")

    _QUBIT = re.compile(r"Q(\d+)", re.IGNORECASE)

    def _qcis_loads(text: str) -> Circuit:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        qubit_ids = []
        parsed = []
        for line in lines:
            parts = line.split()
            op = parts[0].upper()
            qubits = [int(m.group(1)) for m in _QUBIT.finditer(line)]
            qubit_ids.extend(qubits)
            if op == "H":
                parsed.append(("h", qubits[0]))
            elif op == "RZ":
                parsed.append(("rz", qubits[0], float(parts[-1])))
            elif op == "RX":
                parsed.append(("rx", qubits[0], float(parts[-1])))
            elif op == "CX":
                parsed.append(("cx", qubits[0], qubits[1]))
            elif op == "M":
                parsed.append(("measure", qubits[0]))
            elif op == "BARRIER":
                parsed.append(("barrier", tuple(qubits)))
            else:
                raise ValueError(f"unsupported fake QCIS line {line!r}")
        n = (max(qubit_ids) + 1) if qubit_ids else 0
        circuit = Circuit(list(range(n)))
        circuit.operations = parsed
        return circuit

    class _CompileResult:
        def __init__(self, circuit):
            self.circuit = circuit
            self.initial_layout = None
            self.steps = None

    def _compile_circuit(circuit, device=None, **_kwargs):
        return _CompileResult(circuit)

    class _Outcome:
        def __init__(self, bitstr):
            self._bitstr = bitstr

        def to_bitstring(self, num_qubits: int):
            return self._bitstr.zfill(num_qubits)

    class Statevector:
        def __init__(self, num_qubits):
            self.num_qubits = int(num_qubits)

        @staticmethod
        def from_circuit(circuit):
            n = getattr(circuit, "num_qubits", None)
            if n is None:
                n = len(getattr(circuit, "qubits", []))
            return Statevector(int(n or 0))

        def sample_shots(self, shots: int):
            bitstr = "0" * self.num_qubits
            return [_Outcome(bitstr) for _ in range(int(shots))]

    cqlib = types.ModuleType("cqlib")
    cqlib.Circuit = Circuit

    ir = types.ModuleType("cqlib.ir")
    qcis = types.ModuleType("cqlib.ir.qcis")
    qcis.dumps = _qcis_dumps
    qcis.loads = _qcis_loads
    ir.qcis = qcis

    compile_module = types.ModuleType("cqlib.compile")
    compile_module.compile = _compile_circuit

    qis = types.ModuleType("cqlib.qis")
    state = types.ModuleType("cqlib.qis.state")
    state.Statevector = Statevector
    qis.state = state
    qis.Statevector = Statevector

    sys.modules.update(
        {
            "cqlib": cqlib,
            "cqlib.ir": ir,
            "cqlib.ir.qcis": qcis,
            "cqlib.compile": compile_module,
            "cqlib.qis": qis,
            "cqlib.qis.state": state,
        }
    )


try:
    from cqlib_tianyan import TianyanPlatform  # noqa: F401
except ImportError:
    class TianyanPlatform:
        @staticmethod
        def login(**kwargs):
            raise RuntimeError("cqlib_tianyan stub: patch TianyanPlatform in tests")

    tianyan = types.ModuleType("cqlib_tianyan")
    tianyan.TianyanPlatform = TianyanPlatform
    sys.modules["cqlib_tianyan"] = tianyan


@pytest.fixture(scope="session", autouse=True)
def set_matplotlib_headless():
    """Force Matplotlib to use a headless backend for all tests."""
    try:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.show = lambda *args, **kwargs: None
    except Exception as e:
        print(f"Warning: Failed to set Matplotlib backend to Agg. {e}")
