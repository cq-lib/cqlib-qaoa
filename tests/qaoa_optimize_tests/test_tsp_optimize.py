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

"""Example: Using QAOA to solve a Tsp instance."""

import numpy as np

from cqlib_algorithm.problems.tsp import TSP
from cqlib_algorithm.mappings.convert import tsp_to_qubo, qubo_to_ising
from cqlib_algorithm.visualization.tsp_plot import plot_tsp
from cqlib_algorithm.execution import LocalRunner, TianYanRunner
from cqlib_algorithm.algorithms.qaoa import QAOASolver, QAOAConfig
from cqlib_algorithm.optimizers.options import OptimizerOptions


def main():
    # 1) Tsp instance
    distance_matrix = np.array([
         [ 0., 48., 91., 33.],
         [48.,  0., 63., 71.],
         [91., 63.,  0., 92.],
         [33., 71., 92.,  0.]
    ])
    tsp = TSP(n=4, distance_matrix=distance_matrix)

    # 2) Visualize the instance
    plot_tsp(tsp.distance_matrix, tour=[], title="TSP Problem")

    # 3) Tsp -> QUBO
    qubo = tsp_to_qubo(tsp, A=1592.5)
    print(qubo)

    # 4) QUBO -> Ising
    ising = qubo_to_ising(qubo)
    print(ising)

    # 5) Select optimizer
    # SPSA
    #opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 50, "a": 0.2, "c": 0.2})
    # COBYLA
    opt_cfg = OptimizerOptions(name="cobyla",options={"maxiter": 50,"rhobeg": 1.0,"rhoend": 1e-3})
    # Nelder-Mead
    #opt_cfg = OptimizerOptions(name="nelder_mead",options={"maxiter": 50,"initial_step": 0.05,"alpha": 1.0,"gamma": 2.0,"rho": 0.5,"sigma": 0.5,"ftol": 1e-6,"xtol": 1e-6})

    # 6) Solving with QAOA
    solver = QAOASolver(ising, 
                        runner=LocalRunner(), 
                        qaoa_cfg=QAOAConfig(reps=3, mixer="x"), 
                        opt_cfg=opt_cfg)

    res = solver.run()

    # 7) Print optimization results
    # Print measurement results
    res.print_result()
    
    # Print convergence curve
    res.plot_history(title="Optimization History")

    # Print probability distribution
    res.plot_probability(title="QAOA Probability (best θ)", topk=20)

    # Print optimization solution
    res.plot_tsp_solution(distance_matrix = tsp.distance_matrix, title="TSP Solution(QAOA)")

if __name__ == "__main__":
    main()
