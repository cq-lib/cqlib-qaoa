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

"""Unit tests for OptimizerFactory."""

import importlib
from types import SimpleNamespace
import pytest

MODULE_F = "cqlib_algorithm.optimizers.factory"
fac_mod = importlib.import_module(MODULE_F)
OptimizerFactory = fac_mod.OptimizerFactory


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
class _Recorder:
    """Callable test double that records constructed instances."""

    def __init__(self):
        self.instances = []

    def __call__(self, cfg):
        """Return a namespaced object and record the construction."""
        obj = SimpleNamespace(kind=self.__class__.__name__, cfg=cfg)
        self.instances.append(obj)
        return obj


class FakeSPSA(_Recorder):
    """Fake SPSA constructor recorder."""


class FakeCOBYLA(_Recorder):
    """Fake COBYLA constructor recorder."""


class FakeNelderMead(_Recorder):
    """Fake Nelder–Mead constructor recorder."""


def _patch_optimizers(monkeypatch):
    """Patch factory targets with fakes that record invocations."""
    spsa = FakeSPSA()
    cobyla = FakeCOBYLA()
    nm = FakeNelderMead()
    monkeypatch.setattr(fac_mod, "SPSA", spsa)
    monkeypatch.setattr(fac_mod, "COBYLA", cobyla)
    monkeypatch.setattr(fac_mod, "NelderMead", nm)
    return spsa, cobyla, nm


def _mk_cfg(name: str, **opts):
    """Create a minimal config-like object for the factory."""
    return SimpleNamespace(name=name, **opts)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_create_spsa_returns_spsa_instance(monkeypatch):
    """Factory should return an SPSA instance."""
    spsa, cobyla, nm = _patch_optimizers(monkeypatch)
    cfg = _mk_cfg("spsa", maxiter=123, seed=7)

    opt = OptimizerFactory.create(cfg)
    assert getattr(opt, "kind") == "FakeSPSA"
    assert opt.cfg is cfg
    assert len(spsa.instances) == 1
    assert len(cobyla.instances) == 0
    assert len(nm.instances) == 0


def test_create_cobyla_returns_cobyla_instance(monkeypatch):
    """Factory should return a COBYLA instance."""
    spsa, cobyla, nm = _patch_optimizers(monkeypatch)
    cfg = _mk_cfg("cobyla", maxiter=50, tol=1e-6)

    opt = OptimizerFactory.create(cfg)
    assert getattr(opt, "kind") == "FakeCOBYLA"
    assert opt.cfg is cfg
    assert len(cobyla.instances) == 1
    assert len(spsa.instances) == 0
    assert len(nm.instances) == 0


@pytest.mark.parametrize("name_variant", ["nelder_mead", "nelder-mead", "NELDER_MEAD"])
def test_create_nelder_mead_variants(monkeypatch, name_variant):
    """Factory should accept common name variants for Nelder–Mead."""
    spsa, cobyla, nm = _patch_optimizers(monkeypatch)
    cfg = _mk_cfg(name_variant, maxiter=99)

    opt = OptimizerFactory.create(cfg)
    assert getattr(opt, "kind") == "FakeNelderMead"
    assert opt.cfg is cfg
    assert len(nm.instances) == 1
    assert len(spsa.instances) == 0
    assert len(cobyla.instances) == 0


def test_name_is_case_insensitive(monkeypatch):
    """Factory should treat optimizer names case-insensitively."""
    spsa, cobyla, nm = _patch_optimizers(monkeypatch)
    cfg = _mk_cfg("SpSa") 

    opt = OptimizerFactory.create(cfg)
    assert getattr(opt, "kind") == "FakeSPSA"
    assert opt.cfg is cfg


def test_unknown_optimizer_raises_value_error(monkeypatch):
    """Unknown optimizer name should raise ValueError."""
    _patch_optimizers(monkeypatch)
    cfg = _mk_cfg("unknown_opt")
    with pytest.raises(ValueError):
        OptimizerFactory.create(cfg)
