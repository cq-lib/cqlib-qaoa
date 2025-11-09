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

"""Pure-Python Nelder–Mead optimizer with per-iteration callback/history."""

from __future__ import annotations
import math

from cqlib_algorithm.optimizers.base import Optimizer, OptimResult, Objective, Callback
from cqlib_algorithm.optimizers.options import OptimizerOptions


class NelderMead(Optimizer):
    """Derivative-free Nelder–Mead simplex optimizer.

    Supported options (via :class:`OptimizerOptions`):
        - ``maxiter`` (int): Maximum number of iterations / evaluations budget.
        - ``initial_step`` (float | list[float]): Initial simplex scale. A scalar
          applies the same step to all dimensions; a list specifies per-dimension steps.
        - ``alpha`` (float): Reflection coefficient (default 1.0).
        - ``gamma`` (float): Expansion coefficient (default 2.0).
        - ``rho`` (float): Contraction coefficient (default 0.5).
        - ``sigma`` (float): Shrink coefficient (default 0.5).
        - ``ftol`` (float): Function spread tolerance. If ``max(f_i) - min(f_i) < ftol``,
          the function is considered stable.
        - ``xtol`` (float): Simplex size tolerance. If the simplex diameter is below
          this threshold, the shape/scale is considered stable.
    """

    def __init__(self, cfg: OptimizerOptions):
        """Initialize the optimizer with user-configured options.

        Args:
            cfg: Optimizer options container.
        """
        self.cfg = cfg
        opts = cfg.options
        self.maxiter = int(opts.get("maxiter", 200))
        self.initial_step = opts.get("initial_step", 0.05)
        self.alpha = float(opts.get("alpha", 1.0))
        self.gamma = float(opts.get("gamma", 2.0))
        self.rho = float(opts.get("rho", 0.5))
        self.sigma = float(opts.get("sigma", 0.5))
        self.ftol = float(opts.get("ftol", 1e-12))
        self.xtol = float(opts.get("xtol", 1e-6))

    # --- helpers ---
    def _build_initial_simplex(self, x0: list[float]) -> list[list[float]]:
        """Create an initial (n+1)-vertex simplex around ``x0``.

        If ``initial_step`` is a scalar, the same step is applied to each coordinate.
        If it is a list, per-dimension steps are used.

        Args:
            x0: Initial point.

        Returns:
            list[list[float]]: The initial simplex vertices.
        """
        n = len(x0)
        simplex = [list(x0)]
        if isinstance(self.initial_step, (float, int)):
            steps = [float(self.initial_step)] * n
        else:
            steps = [float(s) for s in self.initial_step]
            assert len(steps) == n, "initial_step list length must equal dimension"

        for i in range(n):
            v = list(x0)
            v[i] += steps[i] if steps[i] != 0.0 else 0.05
            simplex.append(v)
        return simplex

    def _centroid(self, verts: list[list[float]], exclude_idx: int) -> list[float]:
        """Compute the centroid of all vertices except the one at ``exclude_idx``."""
        n = len(verts[0])
        m = len(verts) - 1
        c = [0.0] * n
        for j, v in enumerate(verts):
            if j == exclude_idx:
                continue
            for i in range(n):
                c[i] += v[i]
        return [ci / m for ci in c]

    def _lin_comb(self, a: list[float], b: list[float], t: float) -> list[float]:
        """Return the linear combination ``a + t*(b - a)``."""
        return [ai + t * (bi - ai) for ai, bi in zip(a, b)]

    def _simplex_size(self, verts):
        best = verts[0]
        def dist(u, v):
            s = 0.0
            for ui, vi in zip(u, v):
                if getattr(self, "wrap_angles", False):
                    d = (abs(ui - vi)) % (2*math.pi)
                else:
                    d = abs(ui - vi)
                s += d*d
            return math.sqrt(s)
        return max(dist(best, v) for v in verts[1:])

    def minimize(
        self,
        fun: Objective,
        x0: list[float],
        *,
        callback: Callback | None = None,
    ) -> OptimResult:
        """Minimize the objective starting from ``x0`` using Nelder–Mead.

        Args:
            fun: Objective function ``f(theta) -> float``.
            x0: Initial parameter vector.
            callback: Optional per-iteration callback invoked as
                ``callback(theta_best, f_best, it, nfev)``.

        Returns:
            OptimResult: Optimization outcome with the best parameters, value,
            total function evaluations, iterations, a convergence flag
            (set to True when the loop exits, including tolerance/budget),
            a message, and the per-iteration history (best point and value).
        """
        n = len(x0)
        opts = self.cfg.options
        if "gamma" not in opts:
             self.gamma = 1.0 + 2.0 / n
        if "rho" not in opts:
             self.rho = 0.75 - 1.0 / (2.0 * n)
        if "sigma" not in opts:
             self.sigma = 1.0 - 1.0 / n

        simplex = self._build_initial_simplex(list(x0))
        fvals = [fun(v) for v in simplex]
        nfev = len(fvals)
        history = []
        it = 0

        while it < self.maxiter:
            it += 1
            order = sorted(range(len(simplex)), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            best_x, best_f = simplex[0], fvals[0]
            worst_x, worst_f = simplex[-1], fvals[-1]
            second_worst_f = fvals[-2]

            history.append({"iter": it, "x": list(best_x), "fun": best_f})
            if callback:
                callback(list(best_x), best_f, it, nfev)

            fspread = max(abs(fi - best_f) for fi in fvals)
            size = self._simplex_size(simplex)
            if fspread < self.ftol and size < self.xtol:
                break

            c = self._centroid(simplex, exclude_idx=len(simplex) - 1)

            # 1) Reflection: xr = c + alpha*(c - x_worst)
            xr = self._lin_comb(c, worst_x, -self.alpha)
            fr = fun(xr)
            nfev += 1

            if fr < best_f:
                # 2) Expansion: xe = c + gamma*(xr - c)
                xe = self._lin_comb(c, xr, self.gamma)
                fe = fun(xe)
                nfev += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
                continue

            if fr < second_worst_f:
                simplex[-1], fvals[-1] = xr, fr
                continue

            # 3) Contraction
            if fr < worst_f:
                xco = self._lin_comb(c, xr, self.rho)
                fco = fun(xco)
                nfev += 1
                if fco <= fr:
                    simplex[-1], fvals[-1] = xco, fco
                    continue
            else:
                xci = self._lin_comb(c, worst_x, -self.rho)
                fci = fun(xci)
                nfev += 1
                if fci < worst_f:
                    simplex[-1], fvals[-1] = xci, fci
                    continue

            # 4) Shrink towards the best to avoid stagnation
            best = simplex[0]
            new_simplex = [best]
            new_fvals = [fvals[0]]
            for i in range(1, len(simplex)):
                xs = [
                    best[j] + self.sigma * (simplex[i][j] - best[j])
                    for j in range(len(best))
                ]
                fs = fun(xs)
                nfev += 1
                new_simplex.append(xs)
                new_fvals.append(fs)
            simplex, fvals = new_simplex, new_fvals

        return OptimResult(
            theta_opt=list(simplex[0]),
            fun=fvals[0],
            nfev=nfev,
            nit=it,
            converged=True,
            message="Nelder-Mead finished",
            history=history,
        )
