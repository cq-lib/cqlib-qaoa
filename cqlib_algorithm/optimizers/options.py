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

"""Options container for optimizer configuration."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizerOptions:
    """Optimizer selection and hyperparameters.

    Attributes:
        name: Optimizer name (case-insensitive). Supported values include
            ``"spsa"``, ``"cobyla"``, and ``"nelder_mead"``.
        options: Arbitrary hyperparameters and backend-specific settings passed
            to the optimizer implementation (e.g., ``maxiter``, ``rhobeg``,
            ``tol``, ``bounds``, etc.).
    """

    name: str
    options: dict[str, Any] = field(default_factory=dict)
