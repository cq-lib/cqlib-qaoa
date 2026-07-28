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

"""Results Module.

This package provides solution decoding and result analysis utilities.

"""

from .maxcut_decoder import plot_maxcut_solution
from .tsp_decoder import plot_tsp_solution
from .vrp_decoder import plot_vrp_solution

__all__ = [
    "plot_maxcut_solution",
    "plot_tsp_solution",
    "plot_vrp_solution",
]
