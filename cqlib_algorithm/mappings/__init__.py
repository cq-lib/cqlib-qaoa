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

"""Mappings Module.

This package defines the data structures and transformations
for problem encoding and Hamiltonian construction in quantum optimization algorithms.

"""

from .qubo import QUBO
from .hamiltonian import IsingHamiltonian
from .convert import maxcut_to_qubo, tsp_to_qubo, vrp_to_qubo, qubo_to_ising

__all__ = [
    "QUBO",
    "IsingHamiltonian",
    "maxcut_to_qubo",
    "tsp_to_qubo",
    "vrp_to_qubo",
    "qubo_to_ising",
]

