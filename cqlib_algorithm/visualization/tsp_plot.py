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

"""Tsp instance visualization."""

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


def plot_tsp(
    X: np.ndarray,
    tour=None,
    title: str = "TSP instance",
    figsize=(5, 5),
    show_weights: bool = True,
):
    """Plot a TSP instance and (optionally) a directed tour.

    The input can be either city coordinates ``(n, 2)`` or a distance matrix
    ``(n, n)``. If a tour is provided, arrows highlight the route (closed loop).

    Args:
        X: Coordinates ``(n, 2)`` or symmetric distance matrix ``(n, n)``.
        tour: Optional city order as a list of indices; the plot shows a cycle.
        title: Figure title.
        figsize: Matplotlib figure size.
        show_weights: If True, draw edge weights (rounded) on the base graph.

    Raises:
        ValueError: If ``X`` is neither ``(n, 2)`` nor square ``(n, n)``.

    Returns:
        None. Displays the plot via ``plt.show()``.
    """
    X = np.asarray(X, dtype=float)
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
        plt.figure(figsize=figsize)

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="C1")
        nx.draw_networkx_labels(G, pos)

        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)

        if tour:
            path = list(tour) + [tour[0]]
            ax = plt.gca()
            for u, v in zip(path[:-1], path[1:]):
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                dx, dy = x2 - x1, y2 - y1
                shrink_ratio = 0.03
                x1s = x1 + dx * shrink_ratio
                y1s = y1 + dy * shrink_ratio
                x2s = x2 - dx * shrink_ratio
                y2s = y2 - dy * shrink_ratio

                arrow = FancyArrowPatch(
                    (x1s, y1s),
                    (x2s, y2s),
                    arrowstyle="-|>",
                    color="C3",
                    mutation_scale=25,
                    lw=2,
                )
                ax.add_patch(arrow)

        if show_weights:
            edge_labels = {
                (u, v): f"{d['weight']:.0f}" for (u, v, d) in G.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_color="0.2"
            )

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
