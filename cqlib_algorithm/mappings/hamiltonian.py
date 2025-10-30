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

"""Ising Hamiltonian."""

from dataclasses import dataclass

Edge = tuple[int, int]


@dataclass
class IsingHamiltonian:
    """Ising Hamiltonian H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset.

    Attributes:
        n: Number of qubits.
        h: Local field terms, mapping qubit index i → h_i.
        J: Pairwise ZZ couplings, mapping (i, j) with i < j → J_ij.
        offset: Constant energy shift.
    """

    n: int
    h: dict[int, float]
    J: dict[Edge, float]
    offset: float = 0.0

    def __str__(self) -> str:
        """Return a concise Pauli-string representation."""
        return self.to_pauli_string()

    def __repr__(self) -> str:
        """Return a concise Pauli-string representation."""
        return self.to_pauli_string()

    def to_pauli_string(self, precision: int = 3) -> str:
        """Format the Hamiltonian as Pauli strings with coefficients.

        The Pauli string is printed in big-endian convention (leftmost character
        corresponds to the highest qubit index ``n-1``). For a term acting on
        qubit ``i``, the character at position ``n-1-i`` is set to ``'Z'``; the
        rest remain ``'I'``.

        Args:
            precision: Number of decimal places for coefficients.

        Returns:
            str: Multi-line human-readable representation.
        """
        paulis: list[str] = []
        coeffs: list[str] = []

        # Single-qubit Z terms
        for i, v in sorted(self.h.items()):
            if abs(v) < 1e-12:
                continue
            s = ["I"] * self.n
            s[self.n - 1 - i] = "Z"
            paulis.append("".join(s))
            coeffs.append(f"{v:.{precision}f}")

        # Two-qubit ZZ terms
        for (i, j), v in sorted(self.J.items()):
            if abs(v) < 1e-12:
                continue
            s = ["I"] * self.n
            s[self.n - 1 - i] = "Z"
            s[self.n - 1 - j] = "Z"
            paulis.append("".join(s))
            coeffs.append(f"{v:.{precision}f}")

        # Offset
        if abs(self.offset) > 1e-12:
            offset = self.offset

        pauli_str = ", ".join([f"'{p}'" for p in paulis])
        coeff_str = ", ".join(coeffs)
        return (
            f"========== [ Ising Hamiltonian ] ==========\n"
            f"Paulis = [{pauli_str}],\n"
            f"coeffs = [{coeff_str}]\n"
            f"offset = {offset:.{precision}f}\n"
        )
