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

"""LocalRunner.

This module provides a local runner that executes a circuit using a
statevector simulator and returns probability distributions over bitstrings.
It also defines a :class:`SubmitResult` dataclass used to pass submission
metadata alongside results.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from datetime import datetime

from cqlib.simulator.statevector_simulator import StatevectorSimulator
from cqlib_algorithm.visualization.probability_plot import draw_probability


@dataclass
class SubmitResult:
    """Submission metadata for a circuit execution.

    Attributes:
        query_id: Unique identifier for the run (timestamp-based here).
        num_shots: Number of measurement shots used for sampling.
    """

    query_id: str
    num_shots: int


class LocalRunner:
    """Local (simulator-based) execution backend."""

    def _infer_num_qubits(self, circ) -> int:
        """Infer number of qubits from a circuit object.

        Args:
            circ: Circuit object.

        Returns:
            int: Number of qubits if detectable; otherwise 0.
        """
        if hasattr(circ, "num_qubits"):
            return int(circ.num_qubits)
        if hasattr(circ, "qubits"):
            return len(circ.qubits)
        return 0

    def _identity_mapping(self, n: int) -> dict[int, int]:
        """Return an identity virtual-to-physical qubit mapping.

        Args:
            n: Number of qubits.

        Returns:
            dict[int, int]: Mapping ``i -> i`` for ``i in [0, n)``.
        """
        return {i: i for i in range(n)}

    def run_circuit(
        self,
        circ: Any,
        *,
        num_shots: int = 1000,
    ) -> tuple[SubmitResult, dict]:
        """Execute a circuit locally and return normalized probabilities.

        Args:
            circ: Circuit to execute. Must be compatible with
                :class:`StatevectorSimulator`.
            num_shots: Number of measurement shots for sampling.

        Returns:
            tuple[SubmitResult, dict]: A pair of
                - :class:`SubmitResult` with submission metadata.
                - Result dict containing:
                    * ``"probability"``: ``{bitstring: probability}`` sorted by
                      decreasing probability.

        Notes:
            The local statevector sampler typically yields bitstrings with Q0 on
            the **right**. To be consistent with many external tools that expect
            the leftmost bit as Q0, we explicitly reverse each bitstring here.
        """
        simulator = StatevectorSimulator(circ)
        counts = simulator.sample(shots=num_shots)
        total = sum(counts.values()) or 1
        probs = {b: v / total for b, v in counts.items()}
        probs_le = {s[::-1]: p for s, p in probs.items()}
        probs_sorted = dict(sorted(probs_le.items(), key=lambda kv: (-kv[1], kv[0])))

        result = {"probability": probs_sorted}

        query_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        submit_info = SubmitResult(query_id=query_id, num_shots=num_shots)
        return submit_info, result

    def print_result(self, submit_info, result, topk):
        """Pretty-print submission info and plot probability bars.

        Args:
            submit_info: Submission metadata returned by :meth:`run_circuit`.
            result: Result dict containing a ``"probability"`` field.
            topk: Number of top probability entries to visualize in the bar chart.
        """
        print("\n========== [ Experiment Information ] ==========")
        print(f"Task ID  :", submit_info.query_id)
        print(f"Shots  :", submit_info.num_shots)

        print("\n========== [ Measurement Results ] ==========")
        for k, v in result.items():
            print(k, ":", v)

        result_topk = topk
        probs = result["probability"]
        draw_probability(
            probs, title=f"TaskID: {submit_info.query_id}", topk=result_topk
        )
