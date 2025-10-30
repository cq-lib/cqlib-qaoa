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

"""Max-Cut instance visualization."""

from typing import Iterable
import matplotlib.pyplot as plt
import networkx as nx

Edge = tuple[int, int]


def plot_maxcut(
    n: int,
    weights: dict[Edge, float],
    partition: Iterable[int] | None = None,
    pos: dict[int, tuple[float, float]] | None = None,
    title: str = "Max-Cut instance",
    figsize=(5, 5),
    show_weights: bool = True,
):
    """Plot a Max-Cut instance and (optionally) highlight a partition cut.

    Nodes are labeled ``0..n-1``. If a partition is provided, nodes in the set
    are colored differently and cut edges are emphasized.

    Args:
      n: Number of vertices.
      weights: Edge weights mapping with keys as unordered pairs ``(u, v)``.
      partition: Indices of vertices in subset A (others are treated as subset B).
      pos: Optional fixed positions for nodes; if ``None``, spring layout is used.
      title: Plot title.
      figsize: Matplotlib figure size.
      show_weights: If True, draw edge weight labels.

    Returns:
      None. Displays the plot via ``plt.show()``.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (u, v), w in weights.items():
        a, b = (u, v) if u < v else (v, u)
        G.add_edge(a, b, weight=float(w))

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    ):
        plt.figure(figsize=figsize)
        node_color = None
        if partition is not None:
            A = set(partition)
            node_color = ["C0" if i in A else "C1" for i in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_color)
        nx.draw_networkx_labels(G, pos)

        cut_edges = set()
        if partition is not None:
            A = set(partition)
            for u, v in G.edges():
                if (u in A) ^ (v in A):
                    cut_edges.add((u, v))
        ec_default = [(u, v) for (u, v) in G.edges() if (u, v) not in cut_edges]
        nx.draw_networkx_edges(G, pos, edgelist=ec_default, width=2, alpha=0.6)
        if cut_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=3, edge_color="C3")

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
