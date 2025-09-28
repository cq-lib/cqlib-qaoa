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

"""Optimizers public API.

Exports:
    Optimizer: Abstract optimizer interface.
    OptimResult: Result container returned by optimizers.
    Objective: Callable protocol type for objective functions.
    OptimizerOptions: Configurable options for creating optimizers.
    OptimizerFactory: Factory for constructing optimizers from options.
"""

from .base import Optimizer, OptimResult, Objective
from .options import OptimizerOptions
from .factory import OptimizerFactory

__all__ = [
    "Optimizer",
    "OptimResult",
    "Objective",
    "OptimizerOptions",
    "OptimizerFactory",
]
