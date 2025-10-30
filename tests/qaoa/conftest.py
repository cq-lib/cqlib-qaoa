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

"""Pytest configuration."""

import pytest
import matplotlib


@pytest.fixture(scope="session", autouse=True)
def set_matplotlib_headless():
    """Force Matplotlib to use a headless backend for all tests."""
    try:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.show = lambda *args, **kwargs: None

    except Exception as e:
        print(f"Warning: Failed to set Matplotlib backend to Agg. {e}")
