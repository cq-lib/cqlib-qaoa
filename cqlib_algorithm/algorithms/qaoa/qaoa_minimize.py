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

"""QAOA objective wrapper and minimization moduel."""

from __future__ import annotations
from typing import Any

from cqlib_algorithm.algorithms.qaoa.qaoa_evaluator import QAOAEvaluator
from cqlib_algorithm.optimizers.base import Optimizer


class QAOAMinimizer:
    """Wraps a :class:`QAOAEvaluator` as a scalar objective for optimizers.

    This class adapts the evaluator's flexible return formats to a scalar loss
    usable by generic optimizers and caches the latest raw result for downstream
    analysis/visualization.

    """

    def __init__(self, ising, evaluator: QAOAEvaluator, wrap_angles: bool = True):
        """Initialize the minimizer.

        Args:
            ising: Ising Hamiltonian.
            evaluator: QAOA evaluator.
            wrap_angles: Whether to wrap angles (reserved; currently not applied).
            _last_raw_result: Last raw result dict extracted from evaluator output.
        """
        self.ising = ising
        self.evaluator = evaluator
        self.wrap_angles = wrap_angles
        self._last_raw_result: dict[str, Any] | None = None

    def _objective_scalar(
        self, theta: list[float], need_transpile: bool | None
    ) -> float:
        """Evaluate parameters and return a scalar energy.

        Supports multiple evaluator output formats:
          * ``float`` / ``int`` — interpreted directly as energy.
          * ``dict`` — must contain key ``"energy"``; entire dict is cached.
          * ``tuple``/``list`` — first element is energy; the last ``dict`` in the
            tail (if any) is cached as raw result.

        Args:
            theta: Flat parameter vector.
            need_transpile: Optional transpile toggle (real machine in TianYanRunner).

        Returns:
            Energy as a float.

        Raises:
            TypeError: If the evaluator output type or structure is unsupported.
        """
        out = self.evaluator.evaluate(theta, need_transpile=need_transpile)

        # 1) Scalar output
        if isinstance(out, (int, float)):
            self._last_raw_result = None
            return float(out)

        # 2) dict output
        if isinstance(out, dict):
            energy = out.get("energy", None)
            if energy is None:
                raise TypeError(
                    "Evaluator returned dict without required key 'energy'."
                )
            self._last_raw_result = out
            return float(energy)

        # 3) tuple/list output
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            energy = out[0]
            raw = None
            for obj in reversed(out[1:]):
                if isinstance(obj, dict):
                    raw = obj
                    break
            self._last_raw_result = raw
            return float(energy)

        # Unsupported output
        raise TypeError(
            f"Unsupported evaluator output type: {type(out)}. "
            "Expected float | dict | (energy, ...)."
        )

    def minimize(
        self,
        theta0: list[float],
        *,
        optimizer: Optimizer,
        need_transpile: bool | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the provided optimizer on the QAOA objective.

        Args:
            theta0: Initial parameter vector.
            optimizer: Optimizer instance with a ``minimize`` method.
            need_transpile: Optional transpile toggle (real machine in TianYanRunner).
            verbose: Whether to print per-iteration logs.

        Returns:
            dict: Optimization summary containing:
                - ``theta_opt``: Optimized parameters.
                - ``fun``: Best objective value.
                - ``nit``: Number of iterations.
                - ``nfev``: Number of function evaluations.
                - ``history``: Optimizer-specific history/logs.
                - ``result_raw``: Last cached raw evaluator result (dict or None).
        """

        def obj(theta: list[float]) -> float:
            return self._objective_scalar(theta, need_transpile)

        def cb(theta, fval, it, nfev):
            if verbose:
                theta_str = "[" + ", ".join(f"{x:.3f}" for x in theta) + "]"
                print(f"[iter={it:3d}] f={fval:.3f} theta={theta_str}")

        res = optimizer.minimize(obj, theta0, callback=cb)

        return {
            "theta_opt": res.theta_opt,
            "fun": res.fun,
            "nit": res.nit,
            "nfev": res.nfev,
            "history": res.history,
            "result_raw": self._last_raw_result,
        }
