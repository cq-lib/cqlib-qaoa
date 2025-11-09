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

"""Optimizer protocol, result container, and common type aliases."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol, Any

Objective = Callable[[list[float]], float]
Callback = Callable[[list[float], float, int, int], None]


@dataclass
class OptimResult:
    """Standardized optimizer result.

    Attributes:
        theta_opt: Best (minimizing) parameters found.
        fun: Best objective value corresponding to ``theta_opt``.
        nfev: Total number of objective function evaluations.
        nit: Number of iterations performed.
        converged: Whether the optimizer reported convergence.
        message: Human-readable termination message/reason.
        history: Optimizer-specific per-iteration records (optional).
    """

    theta_opt: list[float]
    fun: float
    nfev: int
    nit: int
    converged: bool
    message: str
    history: list[dict[str, Any]]


class Optimizer(Protocol):
    """Protocol for optimizers used throughout the library.

    Implementations should minimize a scalar objective over a parameter vector.
    """

    def minimize(
        self,
        fun: Objective,
        x0: list[float],
        *,
        callback: Callback | None = None,
    ) -> OptimResult: ...
