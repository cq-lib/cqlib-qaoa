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

"""Matplotlib helper to visualize optimizer history."""

import matplotlib.pyplot as plt


def draw_history(history, title="Optimization History"):
    """Plot objective values over iterations.

    Args:
      history: Iterable of dicts with at least keys ``'iter'`` (int) and
        ``'fun'`` (float), e.g., records collected during optimization.
      title: Figure title.

    Returns:
      None. Displays the plot via ``plt.show()``.
    """
    iters = [h["iter"] for h in history]
    funs = [h["fun"] for h in history]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )

    plt.figure(figsize=(6, 4))
    plt.plot(iters, funs, marker="o", markersize=3, linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value (Energy)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
