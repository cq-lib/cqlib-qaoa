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

"""Bar chart visualization for measurement probability distributions."""

import json
import matplotlib.pyplot as plt


def draw_probability(
    prob_dict, title: str = "Measurement Probability", topk: int | None = None
):
    """Plot a bar chart of bitstring measurement probabilities.

    Args:
        prob_dict: Probability mapping ``{bitstring: probability}`` or a JSON
            string encoding the same structure.
        title: Figure title.
        topk: If provided, only the top-k most probable bitstrings are shown.

    Returns:
        None. Displays the plot via ``plt.show()``.
    """
    if isinstance(prob_dict, str):
        prob_dict = json.loads(prob_dict)

    items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    if topk:
        items = items[:topk]

    states, probs = zip(*items)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(states, probs, color="C0")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Probability", fontsize=12)
    plt.xlabel("Qubit string", fontsize=12)
    plt.xticks(rotation=60)

    for bar, p in zip(bars, probs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{p:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()
