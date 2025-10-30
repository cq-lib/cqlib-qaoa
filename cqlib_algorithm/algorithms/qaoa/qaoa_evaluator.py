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

"""QAOA evaluator."""

from __future__ import annotations

from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian
from cqlib_algorithm.execution import LocalRunner, TianYanRunner
from cqlib_algorithm.ansatz.qaoa_ansatz import build_qaoa_circuit
from cqlib_algorithm.execution.objective import expectation_from_probability


class QAOAEvaluator:
    """Evaluate a QAOA parameter vector on an Ising Hamiltonian.

    Given an :class:`IsingHamiltonian` and a runner, this class builds a QAOA
    circuit using the provided depths and mixer, executes it, and computes the
    expectation value of the cost Hamiltonian from the returned probability
    distribution.

    """

    def __init__(
        self,
        ising: IsingHamiltonian,
        runner: LocalRunner,
        *,
        reps: int,
        mixer: str = "x",
        name: str = "QAOA_eval",
        shots: int = 2000,
        insert_barriers: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            ising: Ising Hamiltonian.
            runner: Execution backend (e.g., ``"LocalRunner"`` / ``"TianYanRunner"``).
            reps: Number of QAOA layers (p).
            mixer: Mixer type identifier (e.g., "x", "xy").
            name: Circuit/experiment name.
            shots: Number of measurement shots.
            insert_barriers: Whether to insert visual barriers between layers.
        """
        self.ising = ising
        self.runner = runner
        self.reps = reps
        self.mixer = mixer
        self.name = name
        self.shots = shots
        self.insert_barriers = insert_barriers

    def _unflatten(self, theta: list[float]) -> tuple[list[float], list[float]]:
        """Split a flat parameter vector into (gammas, betas).

        The convention used here is ``theta = [gammas..., betas...]``.

        Args:
            theta: Flat parameter vector of length ``2 * reps``.

        Returns:
            tuple[list[float], list[float]]: Two lists containing ``gammas`` and
            ``betas`` respectively, each of length ``reps``.

        Raises:
            AssertionError: If ``len(theta) != 2 * reps``.
        """
        assert len(theta) == 2 * self.reps, (
            "theta must be length 2*reps (gammas then betas)"
        )
        gammas = list(theta[: self.reps])
        betas = list(theta[self.reps :])
        return gammas, betas

    def evaluate(
        self,
        theta: list[float],
        *,
        lab_id: int | None = None,
        need_transpile: bool | None = None,
    ) -> tuple[float, dict[str, float] | str, dict, dict]:
        """Build, run, and score a QAOA circuit for the given parameters.

        Args:
            theta: Flat parameter vector ``[gammas..., betas...]`` with length ``2 * reps``.
            lab_id: Optional lab identifier (used by :class:`TianYanRunner` only).
            need_transpile: Optional transpile toggle (used by :class:`TianYanRunner` only).

        Returns:
            tuple:
                - expval (float): Expected energy of the Ising Hamiltonian.
                - prob (dict[str, float] | str): Probability distribution over bitstrings.
                - submit_info (dict): Submission information derived from the runner.
                - result (dict): Raw execution result as returned by the runner.

        Raises:
            AssertionError: If the parameter vector length is inconsistent with ``reps``.
        """
        gammas, betas = self._unflatten(theta)
        circ = build_qaoa_circuit(
            n=self.ising.n,
            h=self.ising.h,
            J=self.ising.J,
            reps=self.reps,
            gammas=gammas,
            betas=betas,
            mixer_operator=self.mixer,
            insert_barriers=self.insert_barriers,
            name=self.name,
        )
        if isinstance(self.runner, LocalRunner):
            submit_info, result = self.runner.run_circuit(
                circ,
                num_shots=self.shots,
            )
        elif isinstance(self.runner, TianYanRunner):
            submit_info, result = self.runner.run_circuit(
                circ,
                num_shots=self.shots,
                lab_id=lab_id,
                need_transpile=need_transpile,
            )
        prob = result.get("probability", {})
        expval = expectation_from_probability(self.ising, prob)
        return expval, prob, submit_info.__dict__, result
