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

"""End-to-end unit test for QAOA TSP integration."""

import importlib
import pytest

MOD_TSP = "cqlib_qaoa.problems.tsp"
MOD_CONVERT = "cqlib_qaoa.mappings.convert"
MOD_VIS_TSP = "cqlib_qaoa.visualization.tsp_plot"
MOD_EXEC = "cqlib_qaoa.execution"
MOD_QAOA = "cqlib_qaoa.algorithms.qaoa"
MOD_OPT_OPTS = "cqlib_qaoa.optimizers.options"

m_tsp = importlib.import_module(MOD_TSP)
m_convert = importlib.import_module(MOD_CONVERT)
m_vis = importlib.import_module(MOD_VIS_TSP)
m_exec = importlib.import_module(MOD_EXEC)
m_qaoa = importlib.import_module(MOD_QAOA)
m_optopts = importlib.import_module(MOD_OPT_OPTS)

TSP = m_tsp.TSP
tsp_to_qubo = m_convert.tsp_to_qubo
qubo_to_ising = m_convert.qubo_to_ising
plot_tsp = m_vis.plot_tsp
LocalRunner = m_exec.LocalRunner
QAOASolver = m_qaoa.QAOASolver
QAOAConfig = m_qaoa.QAOAConfig
OptimizerOptions = m_optopts.OptimizerOptions


def _patch_runner_deterministic(monkeypatch, bit_probs):
    """Patch LocalRunner.run_circuit to return a fixed probability distribution."""
    lr_mod = importlib.import_module("cqlib_qaoa.execution.local_runner")
    SubmitResult = lr_mod.SubmitResult

    def fake_run_circuit(self, circ, *, num_shots=1000):
        submit = SubmitResult(query_id="20250101000000", num_shots=num_shots)
        result = {"probability": dict(bit_probs)}
        return submit, result

    monkeypatch.setattr(lr_mod.LocalRunner, "run_circuit", fake_run_circuit)


def test_example_tsp_end2end_fast(monkeypatch):
    """Run an end-to-end TSP→QUBO→Ising→QAOA pipeline with deterministic results.

    Steps:
      1. Build a 4-city TSP instance.
      2. Convert to QUBO with penalty parameter A.
      3. Convert QUBO to Ising Hamiltonian.
      4. Configure a fast SPSA optimizer (maxiter=1).
      5. Patch LocalRunner to return a fixed 16-bit solution bitstring.
      6. Suppress all plotting/show calls for headless CI.
      7. Execute QAOA solver and validate basic result attributes.
      8. Ensure plotting/printing helpers do not raise.
    """
    import numpy as np

    distance_matrix = np.array(
        [
            [0, 26, 30, 19],
            [26, 0, 4, 17],
            [30, 4, 0, 11],
            [19, 17, 11, 0],
        ],
        dtype=float,
    )
    tsp = TSP(n=4, distance_matrix=distance_matrix)

    monkeypatch.setattr(m_vis, "plot_tsp", lambda *a, **k: None)

    qubo = tsp_to_qubo(tsp, A=428.5)
    assert hasattr(qubo, "Q") and hasattr(qubo, "c")

    ising = qubo_to_ising(qubo)
    assert ising.n == 16 

    opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 1, "a": 0.2, "c": 0.2})

    fixed_bitstring = "1000010000100001"
    _patch_runner_deterministic(monkeypatch, {fixed_bitstring: 1.0})

    def _noop(*a, **k):
        pass

    monkeypatch.setattr(m_exec, "draw_probability", _noop, raising=False)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", _noop, raising=True)

    solver = QAOASolver(
        ising,
        runner=LocalRunner(),
        qaoa_cfg=QAOAConfig(reps=1, mixer="x"),
        opt_cfg=opt_cfg,
    )

    res = solver.run()

    assert hasattr(res, "theta_opt")
    theta = res.theta_opt
    assert isinstance(theta, (list, tuple)) and len(theta) > 0

    if hasattr(res, "print_result"):
        res.print_result()
    if hasattr(res, "plot_history"):
        res.plot_history(title="Optimization History")
    if hasattr(res, "plot_probability"):
        res.plot_probability(title="QAOA Probability (best θ)", topk=10)
    if hasattr(res, "plot_tsp_solution"):
        res.plot_tsp_solution(distance_matrix=tsp.distance_matrix, title="TSP Solution (QAOA)")

    final_prob = None
    if hasattr(res, "probability"):
        final_prob = res.probability
    elif hasattr(res, "result_raw") and isinstance(res.result_raw, dict) and "probability" in res.result_raw:
        final_prob = res.result_raw["probability"]

    if final_prob:
        assert fixed_bitstring in final_prob
