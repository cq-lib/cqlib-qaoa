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

"""Optimizers Module.

This package provides classical optimizers used within hybrid quantum-classical
algorithms.

"""

from .base import Optimizer, OptimResult
from .options import OptimizerOptions
from .factory import OptimizerFactory
from .spsa import SPSA
from .cobyla import COBYLA
from .nelder_mead import NelderMead

__all__ = [
    "Optimizer",
    "OptimResult",
    "OptimizerOptions",
    "OptimizerFactory",
    "SPSA",
    "COBYLA",
    "NelderMead",
]
