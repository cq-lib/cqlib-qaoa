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

"""Example: Converting the Maxcut problem model into a Hamiltonian and generating a quantum circuit."""

from cqlib_algorithm.problems.maxcut import MaxCut
from cqlib_algorithm.mappings.convert import maxcut_to_qubo, qubo_to_ising
from cqlib_algorithm.ansatz.qaoa_ansatz.qaoa_ansatz import build_qaoa_circuit
from cqlib_algorithm.visualization.maxcut_plot import plot_maxcut
from cqlib_algorithm.visualization.ansatz_plot import draw_ansatz

def main():
    # 1) MaxCut instance
    weights = {(0,1):1, (1,2):1, (2,3):1, (3,0):1}
    mc = MaxCut(n=4, weights=weights)

    # 2) Visualize the instance
    plot_maxcut(n=mc.n, weights=mc.weights, partition={}, title="MaxCut Problem")

    # 3) Maxcut -> QUBO
    qubo  = maxcut_to_qubo(mc)
    print(qubo)

    # 4) QUBO -> Ising 
    ising = qubo_to_ising(qubo)
    print(ising)

    # 5) Build QAOA circuit
    gammas = [0.8]
    betas  = [0.2]
    reps = 1
    mixer_operator = "x"
    circ = build_qaoa_circuit(ising.n, ising.h, ising.J, reps, betas, gammas, mixer_operator, name="Maxcut_ansatz")
    print(circ.qcis)
 
    # 6) Draw circuit
    draw_ansatz(circ, title="Maxcut Ansatz")

if __name__ == "__main__":
    main()
