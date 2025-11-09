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

"""Factory for constructing optimizers from :class:`OptimizerOptions`."""

from __future__ import annotations

from cqlib_algorithm.optimizers.options import OptimizerOptions
from cqlib_algorithm.optimizers.spsa import SPSA
from cqlib_algorithm.optimizers.cobyla import COBYLA
from cqlib_algorithm.optimizers.nelder_mead import NelderMead


class OptimizerFactory:
    """Create optimizer instances based on a configured name."""

    @staticmethod
    def create(opt_cfg: OptimizerOptions):
        """Instantiate an optimizer according to ``opt_cfg.name``.

        Supported names (case-insensitive):
            - ``"spsa"``
            - ``"cobyla"``
            - ``"nelder_mead"`` or ``"nelder-mead"``

        Args:
            opt_cfg: Optimizer configuration containing the name and options.

        Returns:
            Optimizer: An optimizer instance matching ``opt_cfg.name``.

        Raises:
            ValueError: If the optimizer name is unknown.
        """
        name = opt_cfg.name.lower()
        if name == "spsa":
            return SPSA(opt_cfg)
        elif name == "cobyla":
            return COBYLA(opt_cfg)
        elif name in {"nelder_mead", "nelder-mead"}:
            return NelderMead(opt_cfg)
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
