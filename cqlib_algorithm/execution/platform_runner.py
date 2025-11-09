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

"""TianYanRunner.

This module provides a wrapper around the China Telecom “TianYan” Quantum 
Computing Cloud Platform for submitting quantum circuits, handling (optional) 
topology-aware transpilation, and retrieving execution results.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from datetime import datetime

from cqlib import TianYanPlatform
from cqlib.circuits import Circuit
from cqlib.mapping import transpile_qcis
from cqlib_algorithm.visualization.probability_plot import draw_probability


@dataclass
class SubmitResult:
    """Submission metadata for a Tianyan platform run.

    Attributes:
        query_id: Unique job identifier returned by the platform.
        lab_id: Experiment collection identifier (lab) on the platform.
        exp_name: Experiment name used during submission.
        machine: Backend machine.
        num_shots: Number of measurement shots.
        mapping_virtual_to_final: Optional virtual→physical qubit mapping after transpilation.
        initial_layout: Optional initial layout chosen by the transpiler/platform.
        swap_mapping: Optional SWAP mapping information produced during transpilation.
        used_circuit: The circuit that was actually submitted (post-transpile if any).
    """

    query_id: str
    lab_id: int
    exp_name: str
    machine: str
    num_shots: int
    mapping_virtual_to_final: dict[int, int] | None = None
    initial_layout: Any | None = None
    swap_mapping: Any | None = None
    used_circuit: Circuit | None = None


class TianYanRunner:
    """Runner wrapper for the TianYan cloud/platform.

    Responsibilities:
      - Authenticate and select a machine.
      - Transpile QCIS for real hardware (when required).
      - Submit jobs and poll results.

    Args:
        login_key: API key/token for logging in to the platform.
        machine: Backend machine.

    Raises:
        ValueError: If ``login_key`` is not provided.
    """

    def __init__(self, login_key: str | None = None, machine: str = "tianyan_sw"):
        self.login_key = login_key
        if not self.login_key:
            raise ValueError("Missing login key: please provide `login_key`.")
        self.platform = TianYanPlatform(login_key=self.login_key)
        self.platform.set_machine(machine)
        self.machine = machine

    # ============== Circuit execution API ==============
    def run_circuit(
        self,
        circuit: Circuit,
        *,
        num_shots: int = 1000,
        exp_name: str | None = None,
        lab_id: int | None = None,
        create_lab_if_missing: bool = True,
        need_transpile: bool | None = None,
    ) -> tuple[SubmitResult, dict]:
        """Submit a circuit and return (submission info, single-experiment result).

        For simulators, the method directly submits ``circuit.qcis``.
        For real hardware, the method will transpile to satisfy device topology.

        Args:
            circuit: Circuit to execute (must expose ``.qcis``).
            num_shots: Number of measurement shots.
            exp_name: Optional experiment name. If not provided, an auto name is used.
            lab_id: Existing lab/experiment-collection ID to attach the job to.
            create_lab_if_missing: Whether to auto-create a lab if ``lab_id`` is not provided.
            need_transpile: Override transpilation behavior. If ``None``, it is inferred
                from the backend name (real hardware requires transpilation).

        Returns:
            tuple[SubmitResult, dict[str, Any]]: Submission metadata and a single result dict
            as returned by the platform (e.g., includes ``"probability"``).

        Raises:
            ValueError: If ``lab_id`` is not provided and auto-creation is disabled.
            RuntimeError: If the platform returns no result within the wait window.
        """
        if need_transpile is None:
            simulators = {"tianyan_sw", "tianyan_sim", "tianyan_tn", "tianyan_tnn"}
            low = self.machine.lower()
            need_transpile = low not in simulators

        # 1) Transpile for real hardware
        used_circuit = circuit
        initial_layout = swap_mapping = None
        mapping_virtual_to_final = None
        if need_transpile:
            used_circuit, initial_layout, swap_mapping, mapping_virtual_to_final = (
                transpile_qcis(circuit.qcis, self.platform)
            )

        # 2) Ensure lab exists
        if lab_id is None:
            if create_lab_if_missing:
                lab_id = self.platform.create_lab(
                    name=f"lab.{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    remark="auto_created",
                )
            else:
                raise ValueError("`lab_id` not provided and auto-creation is disabled.")

        # 3) Submit job
        qcis_text = used_circuit.qcis if need_transpile else circuit.qcis
        query_id = self.platform.submit_job(
            circuit=qcis_text,
            exp_name=f"exp.{datetime.now().strftime('%Y%m%d%H%M%S')}",
            lab_id=lab_id,
            num_shots=num_shots,
        )

        submit_info = SubmitResult(
            query_id=query_id,
            lab_id=lab_id,
            exp_name=exp_name,
            machine=self.machine,
            mapping_virtual_to_final=mapping_virtual_to_final,
            initial_layout=initial_layout,
            swap_mapping=swap_mapping,
            used_circuit=used_circuit,
            num_shots=num_shots,
        )

        # 4) Query results
        result = self.platform.query_experiment(
            query_id=query_id, max_wait_time=120, sleep_time=5
        )
        if not result:
            raise RuntimeError("Platform returned no results.")
        single = result[0]

        return submit_info, single

    def print_result(self, submit_info, result, topk):
        """Pretty-print submission info and (optionally) plot probability bars.

        Args:
            submit_info: Submission metadata returned by :meth:`run_circuit`.
            result: Single result dictionary returned by the platform.
            topk: Number of top probability entries for the bar chart.

        Notes:
            Expects ``result["probability"]`` to be a mapping ``bitstring -> probability``.
        """
        print("\n========== [ Experiment Information ] ==========")
        print(f"Lab ID  :", submit_info.lab_id)
        print(f"Task ID  :", submit_info.query_id)
        print(f"Machine  :", submit_info.machine)
        print(f"Shots  :", submit_info.num_shots)
        print(f"Mapping  :", submit_info.mapping_virtual_to_final)

        print("\n========== [ Measurement Results ] ==========")
        for k, v in result.items():
            print(k, ":", v)

        # Probability bar chart
        result_topk = topk
        probs = result["probability"]
        draw_probability(
            probs, title=f"TaskID: {submit_info.query_id}", topk=result_topk
        )
