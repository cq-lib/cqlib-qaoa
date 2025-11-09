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

"""SciPy-backed COBYLA optimizer wrapper with per-iteration history."""

from __future__ import annotations
from typing import Sequence, Any
import numpy as np

try:
    from scipy.optimize import minimize
except Exception as e:
    raise ImportError("SciPy is required for COBYLA backend. pip install scipy") from e

from cqlib_algorithm.optimizers.base import Optimizer, OptimResult, Objective, Callback
from cqlib_algorithm.optimizers.options import OptimizerOptions


def _to_list(x: Sequence[float]) -> list[float]:
    """Convert a numeric sequence (e.g., ndarray) to a Python list.

    Args:
        x: Numeric sequence.

    Returns:
        list[float]: Copy as a standard Python list.
    """
    return list(x)


class COBYLA(Optimizer):
    """COBYLA optimizer powered by :func:`scipy.optimize.minimize`.

    Supported options (via :class:`OptimizerOptions`):
        - ``maxiter`` (int): Maximum number of iterations/function evaluations.
        - ``rhobeg`` (float): Initial trust-region radius / step scale.
        - ``rhoend`` / ``tol`` (float): Stopping tolerance (mapped to SciPy ``tol``).
        - ``catol`` (float): Constraint feasibility tolerance.
        - ``f_target`` (float): Optional early-stop target objective (if supported).
        - ``disp`` (int): Console verbosity.
        - ``bounds`` (list[tuple[lb, ub]] | None): Variable bounds; converted to
          inequality constraints internally (since COBYLA has no native bounds).
        - ``constraints``: SciPy-style constraints
          (``LinearConstraint`` / ``NonlinearConstraint`` / ``dict`` or list thereof).
    """

    def __init__(self, cfg: OptimizerOptions):
        """Initialize the COBYLA optimizer.

        Args:
            cfg: Optimizer options container.
        """
        self.cfg = cfg
        opts = cfg.options
        self.maxiter: int = int(opts.get("maxiter", 1000))
        self.rhobeg: float = float(opts.get("rhobeg", 1.0))
        self.tol: float | None = (
            float(opts["rhoend"])
            if "rhoend" in opts
            else (float(opts["tol"]) if "tol" in opts else 0.0)
        )
        self.catol: float | None = float(opts.get("catol", 2e-4))
        self.f_target: float | None = (
            float(opts.get("f_target", None)) if "f_target" in opts else None
        )
        self.disp: int = int(opts.get("disp", 0))
        self.bounds = opts.get("bounds", None)
        self.cfg_constraints = opts.get("constraints", None)

    def _bounds_to_constraints(self, n: int):
        """Convert bound tuples into COBYLA-compatible inequality constraints.

        For each variable i:
            - Lower bound lb -> constraint: x[i] - lb >= 0
            - Upper bound ub -> constraint: ub - x[i] >= 0

        Args:
            n: Number of variables.

        Returns:
            list[dict]: list of SciPy-style inequality constraints.
        """
        if not self.bounds:
            return []
        cons: list[dict[str, Any]] = []
        for i, (lb, ub) in enumerate(self.bounds):
            if lb is not None and not np.isneginf(lb):
                cons.append({"type": "ineq", "fun": lambda x, i=i, lb=lb: x[i] - lb})
            if ub is not None and not np.isposinf(ub):
                cons.append({"type": "ineq", "fun": lambda x, i=i, ub=ub: ub - x[i]})
        return cons

    def minimize(
        self,
        fun: Objective,
        x0: list[float],
        *,
        constraints: Any = None,
        callback: Callback | None = None,
    ) -> OptimResult:
        """Minimize an objective with COBYLA.

        Args:
            fun: Objective function ``f(theta) -> float``.
            x0: Initial parameter vector.
            constraints: Optional SciPy-style constraints to merge with configured ones.
            callback: Optional per-iteration callback; called as
                ``callback(theta, fval, iter, nfev)``. Here ``nfev`` is set to ``-1``.

        Returns:
            OptimResult: Summary of the optimization run, including per-iteration history.
        """
        x0 = np.asarray(x0, dtype=float)
        history: list[dict[str, Any]] = []

        def _obj(x: np.ndarray) -> float:
            return float(fun(_to_list(x)))
        
        seen_any = False
        def _cb(xk: np.ndarray):
            nonlocal seen_any
            if not seen_any:
                seen_any = True
                return
            fk = _obj(xk)
            rec = {"iter": len(history) + 1, "x": _to_list(xk), "fun": fk}
            history.append(rec)
            if callback is not None:
                callback(_to_list(xk), fk, rec["iter"], -1)

        cons_list: list = []
        if self.cfg_constraints is not None:
            if isinstance(self.cfg_constraints, (list, tuple)):
                cons_list.extend(self.cfg_constraints)
            else:
                cons_list.append(self.cfg_constraints)
        cons_list.extend(self._bounds_to_constraints(len(x0)))

        if constraints is not None:
            if isinstance(constraints, (list, tuple)):
                cons_list.extend(constraints)
            else:
                cons_list.append(constraints)

        options = {
            "rhobeg": self.rhobeg,
            "maxiter": self.maxiter,
            "disp": self.disp,
            "catol": self.catol,
        }
        if self.tol is not None:
            options["tol"] = self.tol
        if self.f_target is not None:
            options["f_target"] = self.f_target

        res = minimize(
            _obj,
            x0,
            method="COBYLA",
            constraints=cons_list if cons_list else (),
            callback=_cb,
            options=options,
        )

        msg = f"SciPy COBYLA finished: {res.message}"
        result = OptimResult(
            theta_opt=_to_list(res.x),
            fun=float(res.fun),
            nfev=int(getattr(res, "nfev", -1)),
            nit=int(getattr(res, "nit", -1)),
            converged=bool(res.success),
            message=msg,
            history=history,
        )
        return result
