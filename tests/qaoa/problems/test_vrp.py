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

"""End-to-end unit test for QAOA VRP integration."""

import importlib
import numpy as np
import pytest

m_qaoa = importlib.import_module("cqlib_algorithm.algorithms.qaoa")
QAOASolver = m_qaoa.QAOASolver
QAOAConfig = m_qaoa.QAOAConfig

m_exec_pkg = importlib.import_module("cqlib_algorithm.execution")
LocalRunner = m_exec_pkg.LocalRunner

m_vis = importlib.import_module("cqlib_algorithm.visualization.vrp_plot")
m_conv = importlib.import_module("cqlib_algorithm.mappings.convert")
vrp_to_qubo = m_conv.vrp_to_qubo
qubo_to_ising = m_conv.qubo_to_ising

m_problem = importlib.import_module("cqlib_algorithm.problems.vrp")
VRP = m_problem.VRP

dec_mod = importlib.import_module("cqlib_algorithm.results.vrp_decoder")


# ---------------------------------------------------------------------
# Utility helpers for constructing bitstrings and deterministic runs.
# ---------------------------------------------------------------------
def _eid(n: int, i: int, j: int) -> int:
    """Linearized edge index mapping: eid(i, j) = i*(n-1) + (j - 1 if j > i else j)."""
    assert i != j
    return i * (n - 1) + (j - 1 if j > i else j)


def _make_arc_bitstring(n: int, arcs: list[tuple[int, int]]) -> str:
    """Generate an n*(n−1)-length bitstring for a given set of directed arcs."""
    N = n * (n - 1)
    bits = ["0"] * N
    for (i, j) in arcs:
        bits[_eid(n, i, j)] = "1"
    return "".join(bits)


def _patch_runner_deterministic(monkeypatch, bit_probs: dict[str, float], shots: int = 1000):
    """Patch LocalRunner.run_circuit to return deterministic probability outputs."""
    lr_mod = importlib.import_module("cqlib_algorithm.execution.local_runner")
    SubmitResult = lr_mod.SubmitResult

    def _fake_run(self, circ, *, num_shots=shots):
        submit = SubmitResult(query_id="20250101000000", num_shots=num_shots)
        total = sum(bit_probs.values()) or 1.0
        probs = {k: float(v) / total for k, v in bit_probs.items()}
        return submit, {"probability": probs}

    monkeypatch.setattr(lr_mod.LocalRunner, "run_circuit", _fake_run, raising=True)


# ---------------------------------------------------------------------
# Main end-to-end integration test.
# ---------------------------------------------------------------------
def test_example_vrp_end2end_fast(monkeypatch):
    """Run a fast VRP→QUBO→Ising→QAOA end-to-end test.

    Steps:
      1. Construct a 5-node, 2-vehicle VRP instance.
      2. Convert to QUBO, then to Ising Hamiltonian.
      3. Patch LocalRunner to return a fixed bitstring probability.
      4. Run QAOA with minimal SPSA iterations (maxiter=1).
      5. Verify result structure and basic fields.
    """
    distance = np.array(
        [
            [0, 2, 6, 7, 3],
            [2, 0, 4, 3, 5], 
            [6, 4, 0, 6, 2],
            [7, 3, 6, 0, 4], 
            [3, 5, 2, 4, 0], 
        ],
        dtype=float,
    )
    demand = np.array([0, 1, 1, 1, 1], dtype=float)
    vrp = VRP(n=5, distance=distance, demand=demand, vehicle_count=2)

    monkeypatch.setattr(m_vis, "plot_vrp", lambda *a, **k: None)

    qubo = vrp_to_qubo(vrp, A_assign=700, A_pos=700)
    assert hasattr(qubo, "Q") and hasattr(qubo, "c")

    ising = qubo_to_ising(qubo)
    assert ising.n == vrp.n * (vrp.n - 1) 

    from cqlib_algorithm.optimizers.options import OptimizerOptions

    opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 1, "a": 0.2, "c": 0.2})

    arcs = [(0, 1), (1, 0), (0, 2), (2, 3), (3, 0)]
    fixed_bitstring = _make_arc_bitstring(vrp.n, arcs)
    _patch_runner_deterministic(monkeypatch, {fixed_bitstring: 1.0})

    monkeypatch.setattr(dec_mod, "plot_vrp", lambda *a, **k: None, raising=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=True)

    solver = QAOASolver(
        ising,
        runner=LocalRunner(),
        qaoa_cfg=QAOAConfig(reps=1, mixer="x"),
        opt_cfg=opt_cfg,
    )
    res = solver.run()

    assert hasattr(res, "theta_opt")
    assert isinstance(res.result_raw, dict)
    assert "probability" in res.result_raw
