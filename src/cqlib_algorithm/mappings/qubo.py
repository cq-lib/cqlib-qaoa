# This code is part of cqlib-algorithm.
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

    def to_string(self, precision: int = 3) -> str:
        """Format the QUBO as aligned numeric arrays.

        Args:
            precision: Number of decimal places for printing coefficients.

        Returns:
            str: Multi-line human-readable representation of ``Q``, ``c`` and ``offset``.
        """
        rows = []
        for row in self.Q:
            row_str = ", ".join(f"{x:.{precision}f}" for x in row)
            rows.append(f"[{row_str}]")
        Q_str = "[\n " + "\n ".join(rows) + "\n]"

        c_str = "[" + ", ".join(f"{x:.{precision}f}" for x in self.c) + "]"
        off_str = f"{self.offset:.{precision}f}"

        return (
            f"========== [ QUBO ] ==========\n"
            f"Q = {Q_str}\n"
            f"c = {c_str}\n"
            f"offset = {off_str}\n"
        )
