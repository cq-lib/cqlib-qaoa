# This code is part of cqlib.
#
# Copyright (C) 2025-2026 China Telecom Quantum Group.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TianYanRunner based on the current cqlib_tianyan binding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cqlib import Circuit
from cqlib.compile import compile as compile_circuit
from cqlib.ir import qcis
from cqlib_tianyan import TianyanPlatform

from cqlib_algorithm.visualization.probability_plot import draw_probability


@dataclass
class SubmitResult:
    """Submission metadata for a TianYan platform run."""

    query_id: str
    lab_id: int | None
    exp_name: str | None
    machine: str
    num_shots: int
    mapping_virtual_to_final: dict[int, int] | None = None
    initial_layout: Any | None = None
    swap_mapping: Any | None = None
    used_circuit: Circuit | str | None = None
    task_ids: list[str] | None = None
    task_handle: Any | None = None


class TianYanRunner:
    """Runner wrapper for the TianYan cloud platform.

    The current platform client is provided by ``cqlib_tianyan``. It accepts
    QCIS strings and returns ``TaskHandle`` / ``ExecutionResult`` objects.
    """

    def __init__(
        self,
        login_key: str | None = None,
        machine: str = "tianyan_sw",
        *,
        domain: str | None = None,
        save_credentials: bool = True,
        auto_refresh: bool = True,
        credentials_path: str | None = None,
    ):
        self.login_key = login_key
        if not self.login_key:
            raise ValueError("Missing login key: please provide `login_key`.")

        login_kwargs: dict[str, Any] = {
            "api_key": self.login_key,
            "save_credentials": save_credentials,
            "auto_refresh": auto_refresh,
        }
        if domain is not None:
            login_kwargs["domain"] = domain
        if credentials_path is not None:
            login_kwargs["credentials_path"] = credentials_path

        self.platform = TianyanPlatform.login(**login_kwargs)
        self.machine = machine
        self.backend = self.platform.get_backend(machine)

    def _circuit_to_qcis(self, circuit: Circuit | str) -> str:
        """Serialize a circuit-like object to QCIS text."""
        if isinstance(circuit, str):
            return circuit
        if hasattr(circuit, "qcis"):
            return str(getattr(circuit, "qcis"))
        return qcis.dumps(circuit)

    def _maybe_compile(
        self, circuit: Circuit | str, need_transpile: bool
    ) -> tuple[Circuit | str, Any | None]:
        """Optionally compile a circuit for the selected backend device."""
        if not need_transpile or isinstance(circuit, str):
            return circuit, None

        device = self.backend.device_config()
        compiled = compile_circuit(circuit, device=device)
        return compiled.circuit, compiled

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
        """Normalize cqlib_tianyan results to the package's result shape."""
        probs = getattr(result, "probabilities", None)
        counts = getattr(result, "counts", {}) or {}
        shots = int(getattr(result, "shots", 0) or 0)

        if probs is None:
            denom = shots or sum(counts.values()) or 1
            probs = {bitstr: count / denom for bitstr, count in counts.items()}

        # cqlib Outcome strings place Q0 on the right; this package evaluates
        # Ising bitstrings with Q0 on the left.
        probs_q0_left = {str(bitstr)[::-1]: float(prob) for bitstr, prob in probs.items()}
        probs_q0_left = dict(
            sorted(probs_q0_left.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        counts_q0_left = {str(bitstr)[::-1]: int(count) for bitstr, count in counts.items()}

        normalized: dict[str, Any] = {
            "probability": probs_q0_left,
            "counts": counts_q0_left,
        }
        for attr in ("task_id", "shots", "num_qubits"):
            if hasattr(result, attr):
                normalized[attr] = getattr(result, attr)
        if hasattr(result, "status"):
            normalized["status"] = str(getattr(result, "status"))
        return normalized

    def run_circuit(
        self,
        circuit: Circuit,
        *,
        num_shots: int = 1000,
        exp_name: str | None = None,
        lab_id: int | None = None,
        need_transpile: bool | None = None,
    ) -> tuple[SubmitResult, dict]:
        """Submit a circuit and return submission metadata plus one result dict."""
        if not isinstance(num_shots, int) or isinstance(num_shots, bool) or num_shots <= 0:
            raise ValueError("num_shots must be a positive integer.")

        if need_transpile is None:
            simulators = {"tianyan_sw", "tianyan_sim", "tianyan_tn", "tianyan_tnn"}
            need_transpile = self.machine.lower() not in simulators

        used_circuit, compile_result = self._maybe_compile(circuit, need_transpile)
        qcis_text = self._circuit_to_qcis(used_circuit)

        task = self.backend.run([qcis_text], shots=num_shots)
        task_ids = list(task.task_ids)
        query_id = task_ids[0] if task_ids else ""

        submit_info = SubmitResult(
            query_id=query_id,
            lab_id=lab_id,
            exp_name=exp_name or f"exp.{datetime.now().strftime('%Y%m%d%H%M%S')}",
            machine=self.machine,
            num_shots=num_shots,
            mapping_virtual_to_final=None,
            initial_layout=getattr(compile_result, "initial_layout", None),
            swap_mapping=getattr(compile_result, "steps", None),
            used_circuit=used_circuit,
            task_ids=task_ids,
            task_handle=task,
        )

        results = task.wait(timeout_secs=120.0, poll_interval_secs=5.0)
        if not results:
            raise RuntimeError("Platform returned no results.")
        single = self._result_to_dict(results[0])

        return submit_info, single

    def print_result(self, submit_info, result, topk):
        """Pretty-print submission info and plot probability bars."""
        print("\n========== [ Experiment Information ] ==========")
        print(f"Lab ID  :", submit_info.lab_id)
        print(f"Task ID  :", submit_info.query_id)
        print(f"Machine  :", submit_info.machine)
        print(f"Shots  :", submit_info.num_shots)
        print(f"Mapping  :", submit_info.mapping_virtual_to_final)

        print("\n========== [ Measurement Results ] ==========")
        for k, v in result.items():
            print(k, ":", v)

        probs = result["probability"]
        draw_probability(probs, title=f"TaskID: {submit_info.query_id}", topk=topk)
