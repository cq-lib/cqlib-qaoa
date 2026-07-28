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

"""End-to-end unit test for QAOA MaxCut integration."""

import importlib
from types import SimpleNamespace
import pytest

MOD_MAXCUT = "cqlib_qaoa.problems.maxcut"
MOD_CONVERT = "cqlib_qaoa.mappings.convert"
MOD_VIS_MAXCUT = "cqlib_qaoa.visualization.maxcut_plot"
MOD_EXEC = "cqlib_qaoa.execution"
MOD_QAOA = "cqlib_qaoa.algorithms.qaoa"
MOD_OPT_OPTS = "cqlib_qaoa.optimizers.options"

m_maxcut = importlib.import_module(MOD_MAXCUT)
m_convert = importlib.import_module(MOD_CONVERT)
m_vis = importlib.import_module(MOD_VIS_MAXCUT)
m_exec = importlib.import_module(MOD_EXEC)
m_qaoa = importlib.import_module(MOD_QAOA)
m_optopts = importlib.import_module(MOD_OPT_OPTS)

MaxCut = m_maxcut.MaxCut
maxcut_to_qubo = m_convert.maxcut_to_qubo
qubo_to_ising = m_convert.qubo_to_ising
plot_maxcut = m_vis.plot_maxcut
LocalRunner = m_exec.LocalRunner
QAOASolver = m_qaoa.QAOASolver
QAOAConfig = m_qaoa.QAOAConfig
OptimizerOptions = m_optopts.OptimizerOptions


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def _patch_runner_deterministic(monkeypatch, bit_probs):
    """Patch LocalRunner.run_circuit to return a fixed probability map."""
    SubmitResult = SimpleNamespace

    def fake_run_circuit(self, circ, *, num_shots=1000):
        submit = SubmitResult(query_id="20250101000000", num_shots=num_shots)
        result = {"probability": dict(bit_probs)}
        return submit, result

    monkeypatch.setattr(LocalRunner, "run_circuit", fake_run_circuit)


# ---------------------------------------------------------------------
# Test: QAOA MaxCut full integration
# ---------------------------------------------------------------------
def test_example_maxcut_end2end_fast(monkeypatch):
    """Run a full MaxCut→QUBO→Ising→QAOA pipeline in a deterministic mode.

    Steps:
      1. Create a 4-node MaxCut instance.
      2. Convert to QUBO, then to Ising form.
      3. Configure a fast SPSA optimizer (maxiter=1).
      4. Patch LocalRunner to return a fixed probability distribution
         biased toward alternating cuts ('0101' or '1010').
      5. Disable all Matplotlib GUI calls.
      6. Execute the QAOA solver.
      7. Validate result object structure and visual interface safety.
    """
    weights = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}
    mc = MaxCut(n=4, weights=weights)

    monkeypatch.setattr(m_vis, "plot_maxcut", lambda *args, **kwargs: None)

    qubo = maxcut_to_qubo(mc)
    assert hasattr(qubo, "Q") and hasattr(qubo, "c")

    ising = qubo_to_ising(qubo)
    assert ising.n == 4

    opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 1, "a": 0.2, "c": 0.2})

    _patch_runner_deterministic(monkeypatch, bit_probs={"0101": 0.8, "1010": 0.2})

    def _noop(*a, **k):
        """No-op placeholder for GUI calls (e.g., plt.show)."""
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
    assert hasattr(res, "result_raw") or hasattr(res, "best_result") or hasattr(res, "probability")

    if hasattr(res, "print_result"):
        res.print_result()
    if hasattr(res, "plot_history"):
        res.plot_history(title="Optimization History")
    if hasattr(res, "plot_probability"):
        res.plot_probability(title="QAOA Probability (best θ)", topk=5)
    if hasattr(res, "plot_maxcut_solution"):
        res.plot_maxcut_solution(n=mc.n, weights=mc.weights, title="MaxCut Solution (QAOA)")

    final_prob = None
    if hasattr(res, "probability"):
        final_prob = res.probability
    elif hasattr(res, "best_result") and isinstance(res.best_result, dict) and "probability" in res.best_result:
        final_prob = res.best_result["probability"]

    if final_prob:
        keys = set(final_prob.keys())
        assert any(k in keys for k in ("0101", "1010"))
