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

"""Required circuit module structure."""

import math

from cqlib.circuits import Circuit


def rzz_via_cnot(circ: Circuit, i: int, j: int, theta: float):
    """Implement e^{-i (theta/2) Z_i Z_j}
    CNOT(i→j) · RZ_j(theta) · CNOT(i→j)
    """
    circ.cx(i, j)
    circ.rz(j, theta)
    circ.cx(i, j)


def rxx_via_cnot(circ: Circuit, i: int, j: int, theta: float) -> None:
    """Implement RXX(theta) = exp(-i * theta/2 * X_i X_j)
    (H⊗H) · RZZ(theta) · (H⊗H)
    """
    circ.h(i)
    circ.h(j)
    rzz_via_cnot(circ, i, j, theta)
    circ.h(i)
    circ.h(j)


def ryy_via_cnot(circ: Circuit, i: int, j: int, theta: float) -> None:
    """Implement RYY(theta) = exp(-i * theta/2 * Y_i Y_j)
    RYY(theta) = (Rx(π/2)⊗Rx(π/2)) · RZZ(theta) · (Rx(-π/2)⊗Rx(-π/2))
    """
    half_pi = math.pi / 2.0
    circ.rx(i, +half_pi)
    circ.rx(j, +half_pi)
    rzz_via_cnot(circ, i, j, theta)
    circ.rx(i, -half_pi)
    circ.rx(j, -half_pi)
