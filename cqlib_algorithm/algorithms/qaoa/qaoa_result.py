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

"""QAOA results."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import matplotlib.pyplot as plt
import math
import numpy as np

from cqlib_algorithm.visualization.probability_plot import draw_probability
from cqlib_algorithm.visualization.history_plot import draw_history
from cqlib_algorithm.results.maxcut_decoder import plot_maxcut_solution
from cqlib_algorithm.results.tsp_decoder import plot_tsp_solution
from cqlib_algorithm.results.vrp_decoder import plot_vrp_solution
from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian


@dataclass
class QAOAResult:
    """Aggregated outputs of a QAOA optimization and its evaluations.

    This dataclass stores the optimized parameters, objective value, execution
    snapshots, and raw measurement results. It also provides helper utilities
    to print summaries and visualize histories/solutions for common problems.

    Attributes:
        theta_opt: Optimized flat parameter vector ``[gammas..., betas...]``.
        fun: Best objective value.
        history: Optional per-iteration records from the optimizer.
        runner_name: Runner identifier (e.g., ``"LocalRunner"`` / ``"TianYanRunner"``).
        qaoa_cfg: QAOA configuration snapshot.
        opt_cfg: Optimizer configuration snapshot.
        submit_info: Runner-specific submission info object.
        result_raw: Raw execution result dict.
        ising: Ising Hamiltonian.
    """

    theta_opt: list[float]
    fun: float
    history: list[dict[str, Any]] | None = None
    runner_name: str = "LocalRunner"
    qaoa_cfg: dict[str, Any] = field(default_factory=dict)
    opt_cfg: dict[str, Any] = field(default_factory=dict)
    submit_info: Any = None
    result_raw: dict[str, Any] | None = None
    ising: IsingHamiltonian | None = None

    def print_result(self, *, topk: int = 20, show: bool = True) -> None:
        """Print a human-readable summary of the submission and measurement results.

        Depending on the runner name, this formats a subset of fields from
        ``submit_info`` and dumps the keys of ``result_raw``.

        Args:
            topk: Top-K entries to show in probability plots (if enabled).
            show: Whether to show plots immediately (currently disabled here).

        Raises:
            ValueError: If ``result_raw`` is missing.
        """
        if not self.result_raw:
            raise ValueError("Missing `result_raw`; cannot print measurement results.")

        result = self.result_raw
        si = self.submit_info

        print("\n========== [ Experiment Information ] ==========")
        if "Local" in self.runner_name:
            # LocalRunner
            print(f"Task ID  :", si.query_id)
            print(f"Shots  :", si.num_shots)
        elif "TianYan" in self.runner_name:
            # TianYanRunner
            print(f"Lab ID  :", getattr(si, "lab_id", ""))
            print(f"Task ID  :", getattr(si, "query_id", ""))
            print(f"Machine  :", getattr(si, "machine", ""))
            print(f"Shots  :", getattr(si, "num_shots", ""))
            print(f"Mapping  :", getattr(si, "mapping_virtual_to_final", {}))
        else:
            print(f"Task ID  :", getattr(si, "query_id", "unknown"))
            print(f"Shots  :", getattr(si, "num_shots", 0))

        lines = []
        lines.append("\n========== [ Optimize Configs ] ==========")
        lines.append(f"runner      : {self.runner_name}")
        if self.qaoa_cfg:
            lines.append(f"qaoa_cfg    : {self.qaoa_cfg}")
        if self.opt_cfg:
            lines.append(f"opt_cfg    : {self.opt_cfg}")
        print("\n".join(lines))

        print("\n========== [ Measurement Results ] ==========")
        for k, v in result.items():
            print(k, ":", v)

    # ====== History visualization =====
    def plot_history(self, *, title: str = "Optimization History", show: bool = True):
        """Plot the optimization history (objective vs. iteration).

        Args:
            title: Figure title.
            show: Whether to call ``plt.show()`` after plotting.
        """
        draw_history(self.history, title=title)
        if show:
            plt.show()

    # ====== Probability distribution visualization ======
    def plot_probability(
        self,
        *,
        title: str = "QAOA Result Probability",
        topk: int = 20,
        show: bool = True,
    ):
        """Plot a bar chart of the measured probability distribution.

        This method expects that ``self.result_raw`` contains a ``"probability"``
        field with a mapping ``bitstring -> probability``.

        Args:
            title: Figure title.
            topk: Number of top probability entries to display.
            show: Whether to call ``plt.show()`` after plotting.

        Raises:
            ValueError: If ``result_raw`` is missing or does not contain ``"probability"``.
        """
        if not self.result_raw or "probability" not in self.result_raw:
            raise ValueError(
                "No available probability distribution in `result_raw['probability']`."
            )
        probs = self.result_raw["probability"]
        draw_probability(probs, title=title, topk=topk)
        if show:
            plt.show()

    # ====== MaxCut solution visualization ======
    def plot_maxcut_solution(
        self,
        *,
        title: str = "MaxCut Solution (QAOA)",
        n: int,
        weights: dict[tuple[int, int], float],
        choose_ones: bool = True,
        show: bool = True,
    ):
        """Decode and plot a MaxCut solution from the raw result.

        Args:
            title: Figure title.
            n: Number of nodes in the graph.
            weights: Edge weights keyed by ``(i, j)``.
            choose_ones: Whether to interpret bit value ``1`` as one partition.
            show: Whether to call ``plt.show()`` inside the decoder (if supported).

        Returns:
            Any: Problem-specific artifact returned by the decoder.

        Raises:
            ValueError: If ``result_raw`` is missing.
            ValueError: If the optimized parameter length is not even (cannot split into 2p).
        """
        if not self.result_raw:
            raise ValueError("Missing `result_raw`; cannot decode MaxCut solution.")

        lines = []
        lines.append("\n========== [ Optimization Results ] ==========")
        theta = list(self.theta_opt)
        theta_num = len(theta)
        if theta_num % 2 != 0:
            raise ValueError(f"theta length {theta_num} is not even; expected 2p.")
        cfg_num = theta_num // 2
        gammas = theta[:cfg_num]
        betas = theta[cfg_num:]
        gammas = np.fmod(gammas, 2 * math.pi)
        betas = np.fmod(betas, math.pi)

        def fmt(xs, prec=3):
            return "[" + ", ".join(f"{x:.{prec}f}" for x in xs) + "]"

        lines.append(f"Best gammas : {fmt(gammas)}")
        lines.append(f"Best betas  : {fmt(betas)}")
        lines.append(f"Best energy : {self.fun:.3f}")
        print("\n".join(lines))

        part = plot_maxcut_solution(
            n=n,
            weights=weights,
            result=self.result_raw,
            choose_ones=choose_ones,
            title=title,
            show=show,
            ising=self.ising,
        )
        return part

    # ====== TSP solution visualization ======
    def plot_tsp_solution(
        self,
        *,
        distance_matrix,
        title: str = "TSP Solution (QAOA)",
        show: bool = True,
    ):
        """Decode and plot a TSP tour from the raw result.

        Args:
            distance_matrix: Either a square distance matrix (n, n) or city coordinates (n, 2).
            title: Figure title.
            show: Whether to call ``plt.show()`` inside the decoder (if supported).

        Returns:
            Any: Problem-specific artifact (decoded tour) returned by the decoder.

        Raises:
            ValueError: If ``result_raw`` is missing.
            ValueError: If the optimized parameter length is not even (cannot split into 2p).
        """
        if not self.result_raw:
            raise ValueError("Missing `result_raw`; cannot decode TSP solution.")

        lines = []
        lines.append("\n========== [ Optimization Results ] ==========")
        theta = list(self.theta_opt)
        theta_num = len(theta)
        if theta_num % 2 != 0:
            raise ValueError(f"theta length {theta_num} is not even; expected 2p.")
        cfg_num = theta_num // 2
        gammas = theta[:cfg_num]
        betas = theta[cfg_num:]
        gammas = np.fmod(gammas, 2 * math.pi)
        betas = np.fmod(betas, math.pi)

        def fmt(xs, prec=3):
            return "[" + ", ".join(f"{x:.{prec}f}" for x in xs) + "]"

        lines.append(f"Best gammas : {fmt(gammas)}")
        lines.append(f"Best betas  : {fmt(betas)}")
        lines.append(f"Best energy : {self.fun:.3f}")
        print("\n".join(lines))

        tour = plot_tsp_solution(
            distance_matrix=distance_matrix,
            result=self.result_raw,
            title=title,
            show=show,
            ising=self.ising,
        )
        return tour

    # ====== VRP solution visualization ======
    def plot_vrp_solution(
        self,
        *,
        distance,
        n: int,
        vehicle_count: int,
        depot: int = 0,
        title: str = "VRP Solution (QAOA)",
        show: bool = True,
    ):
        """Decode and plot a VRP solution from the raw result.

        Args:
            distance: Either a square distance matrix (n, n) or coordinates (n, 2).
            n: Number of customers (or locations).
            vehicle_count: Number of vehicles.
            depot: Depot index (default 0).
            title: Figure title.
            show: Whether to call ``plt.show()`` inside the decoder (if supported).

        Returns:
            Any: Problem-specific artifact (routes) returned by the decoder.

        Raises:
            ValueError: If ``result_raw`` is missing.
            ValueError: If the optimized parameter length is not even (cannot split into 2p).
        """
        if not self.result_raw:
            raise ValueError("Missing `result_raw`; cannot decode VRP solution.")

        lines = []
        lines.append("\n========== [ Optimization Results ] ==========")
        theta = list(self.theta_opt)
        theta_num = len(theta)
        if theta_num % 2 != 0:
            raise ValueError(f"theta length {theta_num} is not even; expected 2p.")
        cfg_num = theta_num // 2
        gammas = theta[:cfg_num]
        betas = theta[cfg_num:]
        gammas = np.fmod(gammas, 2 * math.pi)
        betas = np.fmod(betas, math.pi)

        def fmt(xs, prec=3):
            return "[" + ", ".join(f"{x:.{prec}f}" for x in xs) + "]"

        lines.append(f"Best gammas : {fmt(gammas)}")
        lines.append(f"Best betas  : {fmt(betas)}")
        lines.append(f"Best energy : {self.fun:.3f}")
        print("\n".join(lines))

        routes = plot_vrp_solution(
            distance=distance,
            result=self.result_raw,
            n=n,
            vehicle_count=vehicle_count,
            depot=depot,
            title=title,
            show=show,
            ising=self.ising,
        )
        return routes

    # ====== Serialization ======
    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a plain dictionary."""
        d = asdict(self)
        return d
