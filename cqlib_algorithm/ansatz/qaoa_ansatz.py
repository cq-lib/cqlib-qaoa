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

"""QAOA ansatz construction.

This module provides a builder for QAOA circuits using an Ising cost Hamiltonian
``(h, J)`` together with X/XY mixers or custom mixer/initial-state callbacks.
It relies on a minimal :class:`Circuit` interface from ``cqlib.circuits``.
"""

from typing import Callable

from cqlib.circuits import Circuit
from cqlib_algorithm.transpiler.builders import rzz_via_cnot, rxx_via_cnot, ryy_via_cnot

Edge = tuple[int, int]
MixerFn = Callable[[Circuit, list[int], float], None]
InitFn = Callable[[Circuit, list[int]], None]


def prepare_plus_state(circ: Circuit, qubits: list[int]):
    """Prepare the |+>^n state on the specified qubits.

    Args:
        circ: Target circuit to append gates to.
        qubits: Qubit indices to prepare in |+>.

    Notes:
        This applies a single Hadamard to each qubit.
    """
    for q in qubits:
        circ.h(q)


def build_cost_layer(
    circ: Circuit, h: dict[int, float], J: dict[Edge, float], gamma: float
):
    """Apply the Ising cost layer U_C(gamma).

    Implements single-qubit Z rotations for linear terms and two-qubit ZZ
    rotations for pairwise couplings.

    Args:
        circ: Target circuit.
        h: Linear coefficients of the Ising Hamiltonian; keys are qubit indices.
        J: Pairwise ZZ coefficients; keys are (i, j) tuples.
        gamma: Phase separation angle.

    Notes:
        - For numerical stability, coefficients with absolute value <= 1e-12 are skipped.
        - Rotations use the convention RZ(2 * gamma * h_i) and RZZ(2 * gamma * J_ij).
    """
    for i, hi in h.items():
        if abs(hi) > 1e-12:
            circ.rz(i, 2.0 * gamma * hi)
    for (i, j), w in J.items():
        if abs(w) > 1e-12:
            rzz_via_cnot(circ, i, j, 2.0 * gamma * w)


def mixer_x(circ: Circuit, qubits: list[int], beta: float):
    """Apply a global X mixer layer U_M(beta).

    Args:
        circ: Target circuit.
        qubits: Qubit indices to act on.
        beta: Mixer angle.
    """
    for q in qubits:
        circ.rx(q, 2.0 * beta)


def _ring_edges(qubits: list[int]) -> list[Edge]:
    """Generate ring connectivity over the given qubits.

    Args:
        qubits: Ordered list of qubit indices.

    Returns:
        list[Edge]: Edges connecting (q_k, q_{k+1}) with wrap-around.
    """
    if len(qubits) < 2:
        return []
    return [(qubits[k], qubits[(k + 1) % len(qubits)]) for k in range(len(qubits))]


def mixer_xy(circ: Circuit, qubits: list[int], beta: float) -> None:
    """Apply an XY mixer over a ring: for each edge, RXX(2*beta) then RYY(2*beta).

    The internal connectivity is fixed to a ring:
        (0,1), (1,2), ..., (n-2,n-1), (n-1,0).

    Args:
        circ: Target circuit.
        qubits: Qubit indices in ring order.
        beta: Mixer angle.

    Notes:
        Decomposition uses builders ``rxx_via_cnot`` and ``ryy_via_cnot``.
    """
    theta = 2.0 * beta
    for i, j in _ring_edges(qubits):
        rxx_via_cnot(circ, i, j, theta)
        ryy_via_cnot(circ, i, j, theta)


def build_qaoa_circuit(
    n: int,
    h: dict[int, float],
    J: dict[Edge, float],
    # *,
    reps: int = 1,
    betas: list[float] | None = None,
    gammas: list[float] | None = None,
    mixer_operator: str | MixerFn | None = None,
    initial_state: str | InitFn | Circuit | None = None,
    insert_barriers: bool = False,
    name: str = "QAOA_ansatz",
) -> Circuit:
    """Construct a generic QAOA circuit ``|psi0> -> [ U_C(gamma_k) · U_M(beta_k) ]^reps``.

    Args:
        n: Number of qubits.
        h: Linear Ising coefficients (Z terms) keyed by qubit index.
        J: Pairwise Ising coefficients (ZZ terms) keyed by (i, j).
        reps: Depth p; number of alternating cost/mixer layers (>= 1).
        betas: Mixer angles of length ``reps``. If omitted, defaults to 0.2.
        gammas: Cost angles of length ``reps``. If omitted, defaults to 0.8.
        mixer_operator: Mixer selector:
            - ``None`` or ``"x"``: global X mixer (default).
            - ``"xy"``: ring-structured XY mixer.
            - ``Callable[circ, qubits, beta]``: custom mixer function.
        initial_state: Initial state selector:
            - ``None`` or ``"plus"``: prepare |+>^n.
            - ``Callable[circ, qubits]``: custom initializer.
            - ``Circuit``: prepend an existing circuit (best-effort merge).
        insert_barriers: Insert barriers between cost/mixer layers for readability.
        name: Circuit name.

    Returns:
        Circuit: The constructed QAOA circuit with terminal measurements.

    Raises:
        AssertionError: If ``reps < 1`` or if angle list lengths do not equal ``reps``.
        ValueError: If ``initial_state`` or ``mixer_operator`` selectors are invalid.
        RuntimeError: If a provided initial-state circuit cannot be merged.
    """
    assert reps >= 1, "reps (p) must be >= 1"

    # Parameter preparation
    if gammas is None:
        gammas = [0.8] * reps
    if betas is None:
        betas = [0.2] * reps
    assert len(gammas) == reps and len(betas) == reps, (
        "betas/gammas must be length `reps`"
    )

    # Initial state selection
    init_fn: InitFn
    if initial_state is None or initial_state == "plus":
        init_fn = prepare_plus_state
        init_info = "|+>^n"
    elif isinstance(initial_state, Circuit):
        init_circ = initial_state

        def _append_existing(c: Circuit, qubits: list[int]) -> None:
            for method in ("compose", "extend", "append_circuit"):
                if hasattr(c, method):
                    try:
                        getattr(c, method)(init_circ)
                        return
                    except Exception:
                        pass
            if hasattr(init_circ, "operations"):
                for op in init_circ.operations:
                    try:
                        c.append(op)
                    except Exception:
                        raise RuntimeError(
                            "Failed to merge `initial_state` circuit into main circuit."
                        )
            else:
                raise RuntimeError(
                    "Initial_state is a Circuit but lacks a usable merge/iteration API."
                )

        init_fn = _append_existing
        init_info = f"custom Circuit(name={getattr(initial_state, 'name', 'unnamed')})"
    elif callable(initial_state):
        init_fn = initial_state
        init_info = (
            f"callable init_fn {getattr(initial_state, '__name__', str(initial_state))}"
        )
    else:
        raise ValueError("`initial_state` must be None/'plus'/Circuit/Callable")

    # Mixer selection
    mixer_fn: MixerFn
    if mixer_operator is None or mixer_operator == "x":
        mixer_fn = mixer_x
        mix_info = "X-mixer"
    elif mixer_operator == "xy":
        mixer_fn = mixer_xy
        mix_info = "XY-mixer"
    elif callable(mixer_operator):
        mixer_fn = mixer_operator
        mix_info = (
            f"custom mixer {getattr(mixer_operator, '__name__', str(mixer_operator))}"
        )
    else:
        raise ValueError(
            "`mixer_operator` must be None/'x'/'xy'/Callable(circ, qubits, beta)"
        )

    # Metadata for downstream visualization
    _qaoa_meta = {
        "name": name,
        "n": n,
        "reps": reps,
        "betas": betas,
        "gammas": gammas,
        "mixer": (
            "x"
            if (mixer_operator is None or mixer_operator == "x")
            else "xy"
            if mixer_operator == "xy"
            else getattr(mixer_operator, "__name__", str(mixer_operator))
        ),
        "initial_state": (
            "|+>^n"
            if (initial_state is None or initial_state == "plus")
            else f"custom Circuit(name={getattr(initial_state, 'name', 'unnamed')})"
            if isinstance(initial_state, Circuit)
            else f"callable init_fn {getattr(initial_state, '__name__', str(initial_state))}"
        ),
        "barriers": insert_barriers,
    }

    # Circuit preparation
    qubits = list(range(n))
    try:
        circ = Circuit(qubits=qubits, name=name)
    except Exception:
        circ = Circuit(qubits=qubits)

    # Initial state preparation
    init_fn(circ, qubits)

    # Attach metadata
    try:
        setattr(circ, "_qaoa_meta", _qaoa_meta)
    except Exception:
        pass

    # Layered structure: [Cost(gamma_k) -> Mixer(beta_k)] for k in 0..p-1
    for k in range(reps):
        build_cost_layer(circ, h, J, gammas[k])
        if insert_barriers:
            try:
                circ.barrier(*qubits)
            except Exception:
                pass
        mixer_fn(circ, qubits, betas[k])
        if insert_barriers and k < reps - 1:
            try:
                circ.barrier(*qubits)
            except Exception:
                pass

    # Terminal measurements
    circ.measure_all()

    return circ
