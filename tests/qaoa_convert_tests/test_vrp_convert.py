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

"""Example: Converting the Vrp problem model into a Hamiltonian and generating a quantum circuit."""

import numpy as np

from cqlib_algorithm.problems.vrp import VRP
from cqlib_algorithm.mappings.convert import vrp_to_qubo, qubo_to_ising
from cqlib_algorithm.ansatz.qaoa_ansatz.qaoa_ansatz import build_qaoa_circuit
from cqlib_algorithm.visualization.vrp_plot import plot_vrp
from cqlib_algorithm.visualization.ansatz_plot import draw_ansatz


def main():
    # 1) Vrp instance
    distance = np.array([
        [0,  2,  6,  7,  3],   # depot 0
        [2,  0,  4,  3,  5],   # customer 1
        [6,  4,  0,  6,  2],   # customer 2
        [7,  3,  6,  0,  4],   # customer 3
        [3,  5,  2,  4,  0],   # customer 4
    ], dtype=float)
    demand = np.array([0, 1, 1, 1, 1], dtype=float) 
    vrp = VRP(
        n=5,
        distance=distance,
        demand=demand,
        vehicle_count=2,
        capacity=2,              
        positions_per_vehicle=None
    )

    # 2) Visualize the instance
    plot_vrp(vrp.distance, routes=[], depot=0, title="VRP Problem")

    # 3) Tsp -> QUBO
    qubo = vrp_to_qubo(vrp, A_assign=100.0, A_pos=100.0)
    print(qubo)

    # 4) QUBO -> Ising
    ising = qubo_to_ising(qubo) 
    print(ising)

    # 5) Build QAOA circuit
    betas  = [0.4]
    gammas = [0.6]
    reps = 1
    mixer_operator = "x"
    circ = build_qaoa_circuit(ising.n, ising.h, ising.J, reps, betas, gammas, mixer_operator, name="VRP_ansatz")
    print(circ.qcis)

    # 6) Draw circuit
    draw_ansatz(circ, title="VRP Ansatz")

if __name__ == "__main__":
    main()
