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

"""Utility helpers for parsing probability maps and ranking items."""

from __future__ import annotations
import json


def parse_probability(prob: dict[str, float] | str) -> dict[str, float]:
    """Normalize a probability map possibly provided as JSON.

    Args:
        prob: Either a dict mapping bitstrings to probabilities or a JSON-encoded
            string of such a dict.

    Returns:
        dict[str, float]: Parsed probability mapping.
    """
    if isinstance(prob, str):
        return json.loads(prob)
    return prob


def topk_items(d: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Return the top-k items by value in descending order.

    Args:
        d: Mapping from keys (e.g., bitstrings) to scores/probabilities.
        k: Number of top entries to return.

    Returns:
        list[tuple[str, float]]: The ``k`` highest (key, value) pairs sorted by value.
    """
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]
