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

"""Unit tests for SPSA, COBYLA, and Nelder–Mead optimizers."""

import importlib
from types import SimpleNamespace
import numpy as np
import math
import pytest


# ---------------------------------------------------------------------
# Import targets
# ---------------------------------------------------------------------
MODULE_SPSA = "cqlib_algorithm.optimizers.spsa"
MODULE_COBY = "cqlib_algorithm.optimizers.cobyla"
MODULE_NM = "cqlib_algorithm.optimizers.nelder_mead"
MODULE_OPTS = "cqlib_algorithm.optimizers.options"

spsa_mod = importlib.import_module(MODULE_SPSA)
coby_mod = importlib.import_module(MODULE_COBY)
nm_mod = importlib.import_module(MODULE_NM)
opts_mod = importlib.import_module(MODULE_OPTS)

SPSA = spsa_mod.SPSA
COBYLA = coby_mod.COBYLA
NelderMead = nm_mod.NelderMead
OptimizerOptions = opts_mod.OptimizerOptions


# ---------------------------------------------------------------------
# Shared objective functions and callback recorders
# ---------------------------------------------------------------------
def quad_fun(x):
    """Quadratic objective with minimum at x = 1."""
    return float(sum((xi - 1.0) ** 2 for xi in x))


class CallbackRecorder:
    """Collects callback invocations as immutable snapshots."""

    def __init__(self):
        self.calls = []

    def __call__(self, theta, fval, it, nfev):
        self.calls.append((list(theta), float(fval), int(it), int(nfev)))


# ---------------------------------------------------------------------
# SPSA
# ---------------------------------------------------------------------
def test_spsa_perturb_shape_and_values():
    """_perturb should produce delta in {−1, +1} and symmetric plus/minus vectors."""
    cfg = OptimizerOptions(name="spsa", options={"seed": 123})
    opt = SPSA(cfg)
    x0 = [0.0, 0.0, 0.0]
    plus, minus, delta = opt._perturb(x0, ck=0.5)

    assert len(plus) == len(minus) == len(delta) == len(x0)
    assert set(delta).issubset({-1, 1})
    for p, m, d in zip(plus, minus, delta):
        assert p == pytest.approx(0.5 * d, abs=1e-15)
        assert m == pytest.approx(-0.5 * d, abs=1e-15)


def test_spsa_evaluation_count_and_history_and_callback():
    """SPSA should match expected nfev, nit, history length, and callback cadence."""
    maxiter = 12
    cfg = OptimizerOptions(name="spsa", options={"seed": 1, "maxiter": maxiter, "a": 0.25, "c": 0.25})
    opt = SPSA(cfg)
    x0 = [0.0, 0.0, 0.0]

    cb = CallbackRecorder()
    res = opt.minimize(quad_fun, x0, callback=cb)

    assert res.nfev == 1 + 3 * maxiter
    assert res.nit == maxiter
    assert res.converged is True
    assert isinstance(res.history, list) and len(res.history) == maxiter
    assert len(cb.calls) == maxiter
    for k, (_theta, _f, it, nfev) in enumerate(cb.calls, start=1):
        assert it == k
        assert nfev >= 1 + 3 * k - 2


def test_spsa_runs_and_returns_valid_result():
    """SPSA minimize should run, converge, and return a complete result structure."""
    cfg = OptimizerOptions(name="spsa", options={"seed": 2, "maxiter": 8, "a": 0.2, "c": 0.2})
    opt = SPSA(cfg)
    x0 = [0.0, 0.0]
    res = opt.minimize(quad_fun, x0)

    assert isinstance(res.theta_opt, list) and len(res.theta_opt) == len(x0)
    assert isinstance(res.fun, float)
    assert res.nfev > 0 and res.nit == 8
    assert res.converged is True
    assert res.fun >= 0.0


# ---------------------------------------------------------------------
# COBYLA
# ---------------------------------------------------------------------
def quad_shifted(x):
    """Quadratic objective with minimum at x = 1."""
    return float(sum((xi - 1.0) ** 2 for xi in x))


class CBRecorder:
    """Collects callback invocations for COBYLA tests."""

    def __init__(self):
        self.calls = []

    def __call__(self, theta, fval, it, nfev):
        self.calls.append((list(theta), float(fval), int(it), int(nfev)))


def test_cobyla_options_and_bounds_to_constraints_and_merge(monkeypatch):
    """COBYLA should wire options, convert bounds to constraints, and merge externals."""
    captured = {}

    def fake_minimize(fun, x0, method, constraints, callback, options):
        captured["method"] = method
        captured["options"] = dict(options)
        if isinstance(constraints, tuple):
            constraints = list(constraints)
        captured["constraints"] = list(constraints)
        captured["has_callback"] = callback is not None
        return SimpleNamespace(x=np.asarray(x0, float), fun=float(fun(x0)), success=True,
                               message="OK", nfev=12, nit=3)

    monkeypatch.setattr(coby_mod, "minimize", fake_minimize)

    cfg = OptimizerOptions(
        name="cobyla",
        options={
            "maxiter": 123,
            "rhobeg": 1.5,
            "rhoend": 1e-4,
            "tol": 1e-2, 
            "catol": 3e-4,
            "f_target": 0.0,
            "disp": 1,
            "bounds": [(-0.5, 0.25), (None, 0.1), (-np.inf, np.inf)],
            "constraints": {"type": "ineq", "fun": lambda x: x[0] + 10}, 
        },
    )
    opt = COBYLA(cfg)

    ext_con = {"type": "ineq", "fun": lambda x: 100 - x[1]}

    res = opt.minimize(quad_shifted, x0=[0.0, 0.0, 0.0], constraints=ext_con, callback=CBRecorder())

    assert captured["method"] == "COBYLA"
    opts = captured["options"]
    assert opts["rhobeg"] == 1.5
    assert opts["maxiter"] == 123
    assert opts["catol"] == 3e-4
    assert opts["disp"] == 1
    assert opts["f_target"] == 0.0
    assert np.isclose(opts["tol"], 1e-4)
    assert len(captured["constraints"]) == 5
    assert captured["has_callback"] is True
    assert res.converged is True
    assert res.nit == 3
    assert res.nfev == 12
    assert isinstance(res.history, list)


def test_cobyla_merge_external_constraints(monkeypatch):
    """COBYLA should merge bounds-derived, config, and external constraints."""
    captured = {}

    def fake_minimize(fun, x0, method, constraints, callback, options):
        if isinstance(constraints, tuple):
            constraints = list(constraints)
        captured["constraints_count"] = len(constraints)
        return SimpleNamespace(
            x=np.asarray([0.0, 0.0]),
            fun=float(fun(np.asarray([0.0, 0.0]))),
            success=True,
            message="OK",
            nfev=10,
            nit=5,
        )

    monkeypatch.setattr(coby_mod, "minimize", fake_minimize)

    cfg = OptimizerOptions(
        name="cobyla",
        options={
            "bounds": [(0.0, 1.0), (None, 2.0)],
            "constraints": [{"type": "ineq", "fun": lambda x: x[0] + 1}], 
        },
    )
    opt = COBYLA(cfg)

    ext_cons = [
        {"type": "ineq", "fun": lambda x: 2 - x[1]},
        {"type": "ineq", "fun": lambda x: 100 - (x[0] + x[1])},
    ]
    _ = opt.minimize(quad_shifted, [0.0, 0.0], constraints=ext_cons)

    assert captured["constraints_count"] == 6


# ---------------------------------------------------------------------
# Nelder–Mead
# ---------------------------------------------------------------------
def test_build_initial_simplex_scalar_and_list():
    """Initial simplex should honor scalar and per-dimension initial_step."""
    cfg = OptimizerOptions(name="nelder_mead", options={"initial_step": 0.1})
    opt = NelderMead(cfg)
    x0 = [0.0, 0.0, 0.0]
    simp = opt._build_initial_simplex(x0)
    assert len(simp) == 4 
    assert simp[1] == [0.1, 0.0, 0.0]
    assert simp[2] == [0.0, 0.1, 0.0]
    assert simp[3] == [0.0, 0.0, 0.1]

    cfg2 = OptimizerOptions(name="nelder_mead", options={"initial_step": [0.2, 0.0, 0.05]})
    opt2 = NelderMead(cfg2)
    simp2 = opt2._build_initial_simplex(x0)
    assert simp2[1] == [0.2, 0.0, 0.0]
    assert simp2[2] == [0.0, 0.05, 0.0]
    assert simp2[3] == [0.0, 0.0, 0.05]


def test_centroid_lincomb_simplex_size_properties():
    """Verify centroid (excluding one vertex), linear combination, and simplex size."""
    cfg = OptimizerOptions(name="nelder_mead", options={})
    opt = NelderMead(cfg)
    verts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
    ]
    c = opt._centroid(verts, exclude_idx=2)
    assert c == [0.5, 0.0]

    a = [1.0, 1.0]
    b = [3.0, 5.0]
    t = 0.5
    ab = opt._lin_comb(a, b, t)
    assert ab == [2.0, 3.0]

    size = opt._simplex_size(verts)
    assert size >= 0.0
    assert math.isclose(size, 2.0, rel_tol=1e-12, abs_tol=1e-12)


def test_minimize_on_quadratic_converges_and_records_history():
    """Nelder–Mead should converge on convex quadratic and record history/callbacks."""
    cfg = OptimizerOptions(
        name="nelder_mead",
        options={
            "maxiter": 200,
            "initial_step": 0.2,
            "ftol": 1e-9,
            "xtol": 1e-9,
        },
    )
    opt = NelderMead(cfg)
    x0 = [2.5, -1.0, 0.5] 

    cb = CBRecorder()
    res = opt.minimize(quad_shifted, x0, callback=cb)

    assert isinstance(res.theta_opt, list) and len(res.theta_opt) == len(x0)
    assert isinstance(res.fun, float)
    assert res.nfev > 0
    assert 0 < res.nit <= 200
    assert res.converged is True
    assert isinstance(res.history, list) and len(res.history) == len(cb.calls)
    assert len(res.history) >= 1
    assert res.fun >= 0.0
    assert res.history[-1]["fun"] <= res.history[0]["fun"] + 1e-9

    th_last, f_last, it_last, nfev_last = cb.calls[-1]
    assert isinstance(th_last, list) and isinstance(f_last, float)
    assert isinstance(it_last, int) and isinstance(nfev_last, int)


def test_stops_early_with_large_tolerances():
    """Large tolerances should trigger early stopping in Nelder–Mead."""
    cfg = OptimizerOptions(
        name="nelder_mead",
        options={
            "maxiter": 200,
            "initial_step": 1e-6,
            "ftol": 1e-3, 
            "xtol": 1e-3, 
        },
    )
    opt = NelderMead(cfg)
    x0 = [0.0, 0.0]

    cb = CBRecorder()
    res = opt.minimize(quad_shifted, x0, callback=cb)

    assert 1 <= res.nit < 20
    assert len(res.history) == len(cb.calls) == res.nit
    assert res.converged is True
