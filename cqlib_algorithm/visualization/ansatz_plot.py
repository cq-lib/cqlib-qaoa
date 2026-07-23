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

"""Utilities to render a QAOA ansatz circuit."""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
from cqlib import Circuit
from cqlib.ir import qcis

from cqlib_algorithm.ansatz.qaoa_ansatz import get_qaoa_metadata


def _fmt_angles(xs: Iterable[float], max_len: int = 3) -> str:
    """Format a sequence of angles for concise display."""
    xs = list(xs)
    if len(xs) <= max_len:
        return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"
    head = ", ".join(f"{x:.3f}" for x in xs[:max_len])
    return f"[{head}, ...] (len={len(xs)})"


def _summary_items(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Convert ansatz metadata into label/value pairs."""
    items: list[tuple[str, str]] = [
        ("Name", str(summary.get("name", "QAOA"))),
        ("Qubits", str(summary.get("n", "?"))),
        ("p", str(summary.get("reps", "?"))),
        ("Mixer", str(summary.get("mixer", "x"))),
        ("Init", str(summary.get("initial_state", "|+>^n"))),
    ]
    if summary.get("betas") is not None:
        items.append(("beta", _fmt_angles(summary["betas"])))
    if summary.get("gammas") is not None:
        items.append(("gamma", _fmt_angles(summary["gammas"])))
    items.append(("Barriers", "True" if summary.get("barriers", False) else "False"))
    return items


def _draw_circuit_text(circ: Circuit) -> str:
    """Render circuit text through the current cqlib QCIS serializer."""
    try:
        return qcis.dumps(circ)
    except Exception:
        return repr(circ)
