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

"""QUBO."""

from dataclasses import dataclass
import numpy as np


@dataclass
class QUBO:
    """Quadratic Unconstrained Binary Optimization model.

    A QUBO is defined as: ``x^T Q x + c^T x + offset``, where ``x ∈ {0,1}^n``.

    Attributes:
        Q: Quadratic coefficient matrix with shape ``(n, n)``. It will be
            symmetrized on initialization as ``0.5 * (Q + Q.T)``.
        c: Linear coefficient vector with shape ``(n,)``.
        offset: Constant term.
    """

    Q: np.ndarray
    c: np.ndarray
    offset: float = 0.0

    def __post_init__(self):
        """Normalize internal arrays and symmetrize ``Q``."""
        self.Q = np.array(self.Q, dtype=float)
        self.c = np.array(self.c, dtype=float)
        self.Q = 0.5 * (self.Q + self.Q.T)

    @property
    def n(self) -> int:
        """Number of variables (dimension of the QUBO)."""
        return int(self.Q.shape[0])

    def __str__(self) -> str:
        """Return a string representation."""
        return self.to_string()

    def __repr__(self) -> str:
        """Return a string representation."""
        return self.to_string()

    def to_string(self, precision: int = 12, zero_tol: float = 1e-12) -> str:
        """Print as a single-line objective docplex pretty output."""
        terms = []

        for i in range(self.n):
            for j in range(i + 1, self.n):
                coef = self.Q[i, j] + self.Q[j, i]
                if abs(coef) > zero_tol:
                    terms.append((coef, f"x_{i}*x_{j}"))

        for i in range(self.n):
            lin_coef = self.Q[i, i] + self.c[i]
            if abs(lin_coef) > zero_tol:
                terms.append((lin_coef, f"x_{i}"))

        const = float(self.offset)

        def fmt_coef(v: float) -> str:
            v = 0.0 if abs(v) <= zero_tol else v
            if abs(v - round(v)) <= 10 ** (-precision):
                return str(int(round(v)))
            return f"{v:.{max(0, min(precision, 12))}g}"

        quad = [(c, s) for (c, s) in terms if "*x_" in s]
        lin = [(c, s) for (c, s) in terms if "*x_" not in s]
        ordered = quad + lin

        expr_parts = []
        for k, (coef, sym) in enumerate(ordered):
            sign = " + " if coef >= 0 else " - "
            mag = fmt_coef(abs(coef))
            piece = ("" if k == 0 and coef >= 0 else sign) + f"{mag}*{sym}"
            expr_parts.append(piece)

        if abs(const) > zero_tol or not expr_parts:
            sign = " + " if const >= 0 else " - "
            mag = fmt_coef(abs(const))
            piece = ("" if not expr_parts and const >= 0 else sign) + mag
            expr_parts.append(piece)

        direction = "Maximize" if self.sense == "max" else "Minimize"
        expr = "".join(expr_parts).lstrip()

        return f"========== [ QUBO ] ==========\n{direction}: {expr}\n"
