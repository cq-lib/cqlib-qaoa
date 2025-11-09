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

"""Utilities to render a QAOA ansatz circuit.

This module wraps the cqlib Matplotlib drawer.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Any, Iterable
import numbers

from cqlib.circuits import Circuit
from cqlib.visualization.circuit.mpl import draw_mpl
from cqlib.visualization.circuit.base import BaseDrawer


def _fmt_angles(xs: Iterable[float], max_len: int = 3) -> str:
    """Format a sequence of angles for concise display.

    Shows up to ``max_len`` values with three decimals; truncates with an
    ellipsis and the total length otherwise.

    Args:
        xs: Iterable of numeric angles.
        max_len: Maximum number of values to display before truncation.

    Returns:
        str: Human-readable angle list summary.
    """
    xs = list(xs)
    if len(xs) <= max_len:
        return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"
    head = ", ".join(f"{x:.3f}" for x in xs[:max_len])
    return f"[{head}, …] (len={len(xs)})"


def _summary_items(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Convert a metadata dict into label pairs for the header row.

    Args:
        summary: Metadata dictionary (e.g., from ``circ._qaoa_meta``) with keys
            like ``name``, ``n``, ``reps``, ``mixer``, ``initial_state``,
            ``betas``, ``gammas``, ``barriers``.

    Returns:
        list[tuple[str, str]]: Ordered (label, value) pairs for display.
    """
    items: list[tuple[str, str]] = []
    name = str(summary.get("name", "QAOA"))
    n = str(summary.get("n", "—"))
    reps = str(summary.get("reps", "—"))
    mixer = str(summary.get("mixer", "x"))
    init_s = summary.get("initial_state")

    if isinstance(init_s, str):
        if "plus" in init_s:
            init_s = "|+>^n"
    else:
        init_s = str(init_s) if init_s is not None else "—"
    betas = summary.get("betas")
    gammas = summary.get("gammas")
    barr = "True" if summary.get("barriers", False) else "False"

    items.append(("Name", name))
    items.append(("Qubits", n))
    items.append(("p", reps))
    items.append(("Mixer", mixer))
    items.append(("Init", init_s))
    if betas is not None:
        items.append(("β", _fmt_angles(betas)))
    if gammas is not None:
        items.append(("γ", _fmt_angles(gammas)))
    items.append(("Barriers", barr))
    return items


def draw_ansatz(
    circ: Circuit,
    *,
    title: str = "QAOA Ansatz",
    filename: str | None = None,
    summary: dict[str, Any] | None = None,
    show: bool = True,
    angle_decimals: int = 1,
):
    """Draw a QAOA circuit and overlay a compact configuration summary.

    If ``summary`` is not provided, the function tries to read ``circ._qaoa_meta``.
    Only string formatting is affected; gate parameters in the circuit are not
    modified.

    Args:
        circ: Circuit to draw.
        title: Figure title.
        filename: Optional path to save the figure (PNG). If ``None``, not saved.
        summary: Optional metadata dict (falls back to ``circ._qaoa_meta``).
        show: Whether to call ``plt.show()``.
        angle_decimals: Decimal places for numeric parameter labels in gate boxes.

    Returns:
        matplotlib.figure.Figure: The created figure.

    Notes:
        - Temporarily patches ``BaseDrawer._str_params`` to adjust numeric label
          formatting, and restores it on exit.
    """
    if summary is None:
        summary = getattr(circ, "_qaoa_meta", None) or {
            "name": getattr(circ, "name", "QAOA"),
            "n": len(getattr(circ, "qubits", [])) or None,
        }

    old_str_params = BaseDrawer._str_params

    def _fmt_params(self, ins):
        ps = getattr(ins, "params", None)
        if not ps:
            return ""

        def f(x):
            return f"{x:.{angle_decimals}f}" if isinstance(x, numbers.Real) else str(x)

        if isinstance(ps, dict):
            return ", ".join(f"{k}={f(v)}" for k, v in ps.items())
        if isinstance(ps, (list, tuple)):
            return ", ".join(f(v) for v in ps)
        return f(ps)

    BaseDrawer._str_params = _fmt_params

    try:
        with plt.rc_context(
            {
                "font.family": "DejaVu Sans",
                "axes.titleweight": "bold",
            }
        ):
            fig = draw_mpl(circ, title=title, filename=None, style="default")

            if summary:
                items = _summary_items(summary)
                k = max(len(items), 1)
                y_label, y_value = 0.045, 0.03
                for i, (label, value) in enumerate(items):
                    x = (i + 0.5) / k
                    fig.text(
                        x,
                        y_label,
                        label,
                        ha="center",
                        va="bottom",
                        color="#5B5B5B",
                        fontsize=9,
                        fontweight="regular",
                    )
                    fig.text(
                        x,
                        y_value,
                        value,
                        ha="center",
                        va="top",
                        color="#1A1A1A",
                        fontsize=10,
                        fontweight="bold",
                    )

            if filename:
                fig.savefig(filename, dpi=240, bbox_inches="tight")
            if show:
                plt.show()
        return fig
    finally:
        BaseDrawer._str_params = old_str_params
