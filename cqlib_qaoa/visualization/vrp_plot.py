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

"""Vrp instance visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx


def _classical_mds_2d(D: np.ndarray) -> np.ndarray:
    """Embed a symmetric distance matrix into 2D via classical MDS.

    Args:
        D: Square symmetric distance matrix ``(n, n)``.

    Returns:
        np.ndarray: 2D coordinates ``(n, 2)`` preserving distances in least-squares sense.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    d = np.maximum(w[:2], 0.0)
    X = V[:, :2] * np.sqrt(d[np.newaxis, :])
    return X


def plot_vrp(
    X: np.ndarray, 
    routes=None, 
    depot: int = 0, 
    title: str = "VRP instance"
):
    """Plot a VRP instance and (optionally) per-vehicle routes.

    Accepts either coordinates ``(n, 2)`` or a symmetric distance matrix
    ``(n, n)``. The depot is highlighted in gold; customers in orange.
    If ``routes`` are provided, each vehicle path is shown with arrows and
    distinct colors as a closed loop depot → route → depot.

    Args:
      X: Coordinates ``(n, 2)`` or symmetric distance matrix ``(n, n)``.
      routes: Per-vehicle customer sequences (labels ``1..n-1``), excluding
        the depot, e.g., ``[[1, 2], [3, 4]]``.
      depot: Depot node index (default ``0``).
      title: Figure title.

    Raises:
      ValueError: If ``X`` is not ``(n, 2)`` or ``(n, n)``.

    Returns:
      None. Displays the plot via ``plt.show()``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n = X.shape[0]

    if X.shape[1] == 2:
        coords = X
        D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    elif X.shape[0] == X.shape[1]:
        D = X
        coords = _classical_mds_2d(D)
    else:
        raise ValueError("X must be (n,2) coordinates or an (n,n) distance matrix.")

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(D[i, j]))

    pos = {i: coords[i] for i in range(n)}

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    ):
        plt.figure(figsize=(6, 6))

        node_colors = ["gold" if i == depot else "C1" for i in range(n)]
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)

        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)

        edge_labels = {(u, v): f"{d['weight']:.0f}" for (u, v, d) in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="0.2")

        if routes:
            palette = ["C3", "C2", "C4", "C5", "C6", "C7"]
            ax = plt.gca()
            for k, r in enumerate(routes):
                color = palette[k % len(palette)]
                path = [depot] + list(r) + [depot]
                for u, v in zip(path[:-1], path[1:]):
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    dx, dy = x2 - x1, y2 - y1
                    shrink = 0.03
                    x1s = x1 + dx * shrink
                    y1s = y1 + dy * shrink
                    x2s = x2 - dx * shrink
                    y2s = y2 - dy * shrink
                    arrow = FancyArrowPatch(
                        (x1s, y1s),
                        (x2s, y2s),
                        arrowstyle="-|>",
                        color=color,
                        mutation_scale=28,
                        lw=2.2,
                    )
                    ax.add_patch(arrow)

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
