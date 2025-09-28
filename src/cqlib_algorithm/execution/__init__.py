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

"""Execution backends public API.

Exports:
    TianYanRunner: Cloud/platform runner.
    SubmitResult: Submission metadata/result wrapper.
    LocalRunner: Local simulation runner.
"""

from cqlib_algorithm.execution.platform_runner import TianYanRunner, SubmitResult
from cqlib_algorithm.execution.local_runner import LocalRunner
__all__ = ["TianYanRunner", 
           "SubmitResult",
           "LocalRunner",]
