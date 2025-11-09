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

"""Expectation utilities for Ising Hamiltonians."""

from __future__ import annotations
import json

from cqlib_algorithm.mappings.hamiltonian import IsingHamiltonian


def _spin_from_bitstring(bitstr: str):
    """Convert a bitstring into a list of spins (+1 for '0', -1 for '1').

    Args:
        bitstr: Bitstring.

    Returns:
        list[int]: Spin configuration where ``+1`` maps to bit ``'0'`` and
        ``-1`` maps to bit ``'1'``.
    """
    return [1 if b == "0" else -1 for b in bitstr]


def energy_of_bitstring(ising: IsingHamiltonian, bitstr: str) -> float:
    """Compute the Ising energy of a given bitstring.

    Uses the mapping ``'0' -> +1`` and ``'1' -> -1`` for spins. The Hamiltonian
    is assumed to be of the form ``H = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j``.

    Args:
        ising: Target Ising Hamiltonian with fields ``h`` and ``J``.
        bitstr: Bitstring representing a configuration.

    Returns:
        float: Energy of the configuration under the Ising model.
    """
    s = _spin_from_bitstring(bitstr)
    e = 0.0
    for i, hi in ising.h.items():
        e += hi * s[i]
    for (i, j), Jij in ising.J.items():
        e += Jij * s[i] * s[j]
    return e


def parse_probability(prob: dict[str, float] | str) -> dict[str, float]:
    """Normalize probability input to a dictionary.

    Args:
        prob: Either a mapping ``bitstring -> probability`` or a JSON string
            that decodes to such a mapping.

    Returns:
        dict[str, float]: Parsed probability dictionary.

    Raises:
        TypeError: If ``prob`` is neither a ``dict`` nor a JSON ``str``.
        json.JSONDecodeError: If ``prob`` is a string but not valid JSON.
    """
    if isinstance(prob, dict):
        return prob
    if isinstance(prob, str):
        return json.loads(prob)
    raise TypeError("`prob` must be a dict or a JSON string.")


def expectation_from_probability(
    ising: IsingHamiltonian, prob: dict[str, float] | str
) -> float:
    """Compute expected Ising energy from a probability distribution.

    Args:
        ising: Target Ising Hamiltonian.
        prob: Probability distribution over bitstrings, either as a dict
            ``{bitstring: probability}`` or a JSON string representing such a dict.

    Returns:
        float: Expected energy ``E = sum_b E(b) * P(b)`` where ``E(b)`` is the
        energy of bitstring ``b``.
    """
    p = parse_probability(prob)
    ex = 0.0
    for bitstr, pr in p.items():
        ex += energy_of_bitstring(ising, bitstr) * float(pr)
    return ex
