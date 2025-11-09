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

"""SPSA optimizer (Simultaneous Perturbation Stochastic Approximation)."""

from __future__ import annotations
import random

from cqlib_algorithm.optimizers.base import Optimizer, OptimResult, Objective, Callback
from cqlib_algorithm.optimizers.options import OptimizerOptions


class SPSA(Optimizer):
    """Stochastic, gradient-free optimizer using simultaneous perturbations.

    Supported options (via :class:`OptimizerOptions`):
        - ``maxiter`` (int): Number of iterations.
        - ``a`` (float): Learning-rate base.
        - ``c`` (float): Perturbation base.
        - ``seed`` (int | None): RNG seed for reproducibility.
    """

    def __init__(self, cfg: OptimizerOptions):
        """Initialize SPSA with user-provided hyperparameters.

        Args:
            cfg: Optimizer options container.
        """
        self.cfg = cfg
        opts = cfg.options
        self.maxiter = int(opts.get("maxiter", 50))
        self.a = float(opts.get("a", 0.2))
        self.c = float(opts.get("c", 0.2))
        self.seed = opts.get("seed", None)
        if self.seed is not None:
            random.seed(self.seed)

    def _perturb(self, x: list[float], ck: float):
        """Generate ± perturbations along a Rademacher vector.

        Args:
            x: Current parameter vector.
            ck: Perturbation scale at iteration k.

        Returns:
            tuple:
                - plus (list[float]): ``x + ck * delta``
                - minus (list[float]): ``x - ck * delta``
                - delta (list[int]): Rademacher entries in ``{+1, -1}``.
        """
        delta = [1 if random.random() < 0.5 else -1 for _ in x]
        plus = [xi + ck * di for xi, di in zip(x, delta)]
        minus = [xi - ck * di for xi, di in zip(x, delta)]
        return plus, minus, delta

    def minimize(
        self,
        fun: Objective,
        x0: list[float],
        *,
        callback: Callback | None = None,
    ) -> OptimResult:
        """Run SPSA to minimize a scalar objective.

        Args:
            fun: Objective function ``f(theta) -> float``.
            x0: Initial parameter vector.
            callback: Optional per-iteration callback invoked as
                ``callback(theta, fval, iter, nfev)``.

        Returns:
            OptimResult: Final parameters, value, evaluation counts, iterations,
            a convergence flag (True on loop completion), message, and history.
        """
        x = list(x0)
        history = []
        f_cur = fun(x)
        nfev = 1

        for k in range(1, self.maxiter + 1):
            ak = self.a / (k**0.602)
            ck = self.c / (k**0.101)

            x_plus, x_minus, delta = self._perturb(x, ck)
            f_plus = fun(x_plus)
            nfev += 1
            f_minus = fun(x_minus)
            nfev += 1
            gk = [(f_plus - f_minus) / (2.0 * ck * di) for di in delta]
            x = [xi - ak * gi for xi, gi in zip(x, gk)]
            f_cur = fun(x)
            nfev += 1

            history.append({"iter": k, "x": list(x), "fun": f_cur})
            if callback:
                callback(list(x), f_cur, k, nfev)

        return OptimResult(
            theta_opt=x,
            fun=f_cur,
            nfev=nfev,
            nit=self.maxiter,
            converged=True,
            message="SPSA finished",
            history=history,
        )
