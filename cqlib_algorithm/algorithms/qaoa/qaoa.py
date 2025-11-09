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

"""QAOA configs and solver."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math

from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian
from cqlib_algorithm.ansatz.qaoa_ansatz import build_qaoa_circuit
from cqlib_algorithm.execution import LocalRunner, TianYanRunner
from cqlib_algorithm.algorithms.qaoa.qaoa_evaluator import QAOAEvaluator
from cqlib_algorithm.algorithms.qaoa.qaoa_minimize import QAOAMinimizer
from cqlib_algorithm.visualization.ansatz_plot import draw_ansatz
from cqlib_algorithm.optimizers.factory import OptimizerFactory
from cqlib_algorithm.optimizers.options import OptimizerOptions
from cqlib_algorithm.algorithms.qaoa.qaoa_result import QAOAResult


@dataclass
class QAOAConfig:
    """QAOA configs.

    Attributes:
        reps: QAOA depth (number of alternating operator layers).
        mixer: Mixer operator identifier passed to circuit builder (e.g., "x", "xy").
        shots: Number of measurement shots for sampling.
        name: Algorithm name.
        wrap_angles: Whether to wrap (gamma, beta) into canonical ranges.
        need_transpile: Optional transpilation flag (real machine in TianYanRunner).
        verbose: Print per-iteration logs during optimization.
    """

    reps: int = 1
    mixer: str = "x"
    shots: int = 1000
    name: str = "QAOA"
    wrap_angles: bool = True
    need_transpile: bool | None = None
    verbose: bool = True


class QAOASolver:
    """End-to-end driver that optimizes QAOA parameters and optionally re-evaluates at optimum."""

    def __init__(
        self,
        ising: IsingHamiltonian,
        *,
        runner=None,
        qaoa_cfg: QAOAConfig | None = None,
        opt_cfg: OptimizerOptions | None = None,
    ):
        """Initialize the solver with problem, backend, and configs.

        Args:
            ising: Target Ising Hamiltonian.
            runner: Execution backend (e.g., ``"LocalRunner"`` / ``"TianYanRunner"``).
            qaoa_cfg: QAOA configuration.
            opt_cfg: Optimizer configuration.
        """
        self.ising = ising
        self.runner = runner or LocalRunner()
        self.qaoa_cfg = qaoa_cfg or QAOAConfig()
        self.opt_cfg = opt_cfg or OptimizerOptions(
            name="spsa", options={"maxiter": 100, "a": 0.2, "c": 0.2}
        )

        self.evaluator = QAOAEvaluator(
            ising=self.ising,
            runner=self.runner,
            reps=self.qaoa_cfg.reps,
            mixer=self.qaoa_cfg.mixer,
            shots=self.qaoa_cfg.shots,
            name=self.qaoa_cfg.name,
        )
        self.minimizer = QAOAMinimizer(
            ising=self.ising,
            evaluator=self.evaluator,
            wrap_angles=self.qaoa_cfg.wrap_angles,
        )

    @staticmethod
    def default_theta(p: int, gamma: float = 0.8, beta: float = 0.2) -> list[float]:
        """Build a flat initial parameter vector ``[gammas..., betas...]`` of length ``2p``."""
        return [gamma] * p + [beta] * p

    def _build_circuit_for_theta(self, theta: list[float]):
        """Construct a QAOA circuit for a given parameter vector.

        The evaluator's current configuration is used (depth, mixer, naming, etc.).
        The convention is ``theta = [gammas..., betas...]`` with length ``2 * reps``.

        Args:
            theta: Flat parameter vector.

        Returns:
            Any: A quantum circuit object as returned by ``build_qaoa_circuit``.

        Raises:
            AssertionError: If ``len(theta) != 2 * reps``.
        """
        evalr = self.evaluator
        ising = evalr.ising
        reps = evalr.reps
        mixer = getattr(evalr, "mixer", "x")
        name = getattr(evalr, "name", "Maxcut_ansatz")
        insert_barriers = getattr(evalr, "insert_barriers", False)

        assert len(theta) == 2 * reps, "theta must be of length 2*reps"
        gammas = list(theta[:reps])
        betas = list(theta[reps:])

        if getattr(self, "wrap_angles", False):
            gammas = [t % (2 * math.pi) for t in gammas]
            betas = [t % math.pi for t in betas]

        circ = build_qaoa_circuit(
            n=ising.n,
            h=ising.h,
            J=ising.J,
            reps=reps,
            betas=betas,
            gammas=gammas,
            mixer_operator=mixer,
            insert_barriers=insert_barriers,
            name=name,
        )
        draw_ansatz(circ, title="QAOA Ansatz")
        return circ

    # ------- Run entrypoint -------
    def run(
        self,
        initial_theta: list[float] | None = None,
        *,
        need_transpile: bool | None = None,
        verbose: bool | None = None,
        post_eval: bool = True,
    ) -> dict[str, Any]:
        """Optimize parameters and optionally re-sample at the optimum.

        Args:
            initial_theta: Optional initial parameter vector; if None, uses ``default_theta(p)``.
            need_transpile: Optional transpile toggle (real machine in TianYanRunner).
            verbose: Per-iteration logging toggle.
            post_eval: If True, run an additional circuit execution at the found optimum.

        Returns:
            QAOAResult: Optimization outputs.
        """
        p = self.qaoa_cfg.reps
        theta0 = initial_theta or self.default_theta(p)

        optimizer = OptimizerFactory.create(self.opt_cfg)

        res = self.minimizer.minimize(
            theta0,
            optimizer=optimizer,
            need_transpile=self.qaoa_cfg.need_transpile
            if need_transpile is None
            else need_transpile,
            verbose=self.qaoa_cfg.verbose if verbose is None else verbose,
        )

        theta_opt = res.get("theta_opt", None)
        submit_info = None
        result_raw = None

        if post_eval and theta_opt is not None:
            try:
                circ_best = self._build_circuit_for_theta(theta_opt)
                submit_info, result_raw = self.runner.run_circuit(
                    circ_best,
                    num_shots=self.qaoa_cfg.shots,
                )
            except Exception as e:
                result_raw = {"post_eval_error": f"{type(e).__name__}: {e}"}

        return QAOAResult(
            theta_opt=theta_opt or [],
            fun=res.get("fun", float("nan")),
            history=res.get("history", None),
            runner_name=type(self.runner).__name__,
            qaoa_cfg=asdict(self.qaoa_cfg),
            opt_cfg=asdict(self.opt_cfg),
            submit_info=submit_info,
            result_raw=result_raw,
            ising=self.ising,
        )
