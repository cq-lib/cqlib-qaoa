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

"""Unit tests for IsingHamiltonian string conversion and formatting."""

import importlib
import re
import pytest

MODULE_H = "cqlib_qaoa.mappings.hamiltonian"
ham_mod = importlib.import_module(MODULE_H)
IsingHamiltonian = ham_mod.IsingHamiltonian


def test_to_pauli_string_big_endian_and_precision():
    """Verify Pauli-string formatting, ordering, and precision output."""

    h = {0: 1.0, 1: 1e-14, 2: -0.5}
    J = {(0, 2): 2.0, (1, 2): 1.0, (0, 1): 1e-14}
    offset = 1.5
    H = IsingHamiltonian(n=3, h=h, J=J, offset=offset)

    s = H.to_pauli_string(precision=3)

    assert "========== [ Ising Hamiltonian ] ==========" in s
    assert re.search(r"offset\s=\s1\.500", s)

    assert "'IIZ'" in s 
    assert "'ZII'" in s 
    assert "'ZIZ'" in s
    assert "'ZZI'" in s 

    assert "1.000" in s    
    assert "-0.500" in s  
    assert "2.000" in s    
    assert "1.000" in s    

    assert str(H) == s
    assert repr(H) == s
