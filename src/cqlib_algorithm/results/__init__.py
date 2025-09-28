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

"""Results utilities public API.

Exports problem decoding.
"""

from cqlib_algorithm.results.utils import parse_probability, topk_items
from cqlib_algorithm.results.maxcut_decoder import (
    best_bitstring_from_probability,
    decode_from_platform_result,
    plot_maxcut_solution,
)
__all__ = [
    "parse_probability", 
    "topk_items", 
    "best_bitstring_from_probability", 
    "decode_from_platform_result", 
    "plot_maxcut_solution",
]
