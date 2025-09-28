# This code is part of cqlib-algorithm.
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

"""Example: Converting the Tsp problem model into a Hamiltonian and generating a quantum circuit."""

import numpy as np

from cqlib_algorithm.problems.tsp import TSP
from cqlib_algorithm.mappings.convert import tsp_to_qubo, qubo_to_ising
from cqlib_algorithm.ansatz.qaoa_ansatz.qaoa_ansatz import build_qaoa_circuit
from cqlib_algorithm.visualization.tsp_plot import plot_tsp
from cqlib_algorithm.visualization.ansatz_plot import draw_ansatz


def main():
    # 1) Tsp instance
    distance_matrix = np.array([
        [0, 48, 91],
        [48, 0, 63],
        [91, 63, 0]
    ])
    tsp = TSP(n=3, distance_matrix=distance_matrix)

    # 2) Visualize the instance
    plot_tsp(tsp.distance_matrix, tour=[], title="TSP Problem")

    # 3) Tsp -> QUBO
    qubo = tsp_to_qubo(tsp, A=606.5)
    print(qubo)

    # 4) QUBO -> Ising
    ising = qubo_to_ising(qubo)
    print(ising)

    # 5) Build QAOA circuit
    betas  = [1]
    gammas = [1]
    reps = 1
    mixer_operator = "x"
    circ = build_qaoa_circuit(ising.n, ising.h, ising.J, reps, betas, gammas, mixer_operator, name="TSP_ansatz")
    print(circ.qcis)

    # 6) Draw circuit
    draw_ansatz(circ, title="TSP Ansatz")


if __name__ == "__main__":
    main()
