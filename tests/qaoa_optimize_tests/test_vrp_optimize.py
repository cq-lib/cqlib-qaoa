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

"""Example: Using QAOA to solve a Vrp instance."""

import numpy as np

from cqlib_algorithm.problems.vrp import VRP
from cqlib_algorithm.mappings.convert import vrp_to_qubo, qubo_to_ising
from cqlib_algorithm.visualization.vrp_plot import plot_vrp
from cqlib_algorithm.execution import LocalRunner, TianYanRunner
from cqlib_algorithm.algorithms.qaoa import QAOASolver, QAOAConfig
from cqlib_algorithm.optimizers.options import OptimizerOptions


def main():
    # 1) Vrp instance
    distance = np.array(
        [
            [0, 2, 6, 7, 3],  # depot 0
            [2, 0, 4, 3, 5],  # customer 1
            [6, 4, 0, 6, 2],  # customer 2
            [7, 3, 6, 0, 4],  # customer 3
            [3, 5, 2, 4, 0],  # customer 4
        ],
        dtype=float,
    )
    demand = np.array([0, 1, 1, 1, 1], dtype=float)
    vrp = VRP(
        n=5,
        distance=distance,
        demand=demand,
        vehicle_count=2,
        capacity=2,
    )

    # 2) Visualize the instance
    plot_vrp(vrp.distance, routes=[], depot=0, title="VRP Problem")

    # 3) Tsp -> QUBO
    qubo = vrp_to_qubo(vrp, A_assign=700, A_pos=700)
    print(qubo)

    # 4) QUBO -> Ising
    ising = qubo_to_ising(qubo)
    print(ising)

    # 5) Select optimizer
    # SPSA
    # opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 50, "a": 0.2, "c": 0.2, "seed": 123})
    # COBYLA
    # opt_cfg = OptimizerOptions(name="cobyla",options={"maxiter": 50,"rhobeg": 1.0,"rhoend": 1e-3})
    # Nelder-Mead
    opt_cfg = OptimizerOptions(
        name="nelder_mead",
        options={
            "maxiter": 5,
            "initial_step": 0.05,
            "alpha": 1.0,
            "gamma": 2.0,
            "rho": 0.5,
            "sigma": 0.5,
            "ftol": 1e-6,
            "xtol": 1e-6,
        },
    )

    # 6) Solving with QAOA
    solver = QAOASolver(
        ising,
        runner=LocalRunner(),
        qaoa_cfg=QAOAConfig(reps=2, mixer="x"),
        opt_cfg=opt_cfg,
    )

    res = solver.run()

    # 7) Print optimization results
    # Print measurement results
    res.print_result()

    # Print convergence curve
    res.plot_history(title="Optimization History")

    # Print probability distribution
    res.plot_probability(title="QAOA Probability (best θ)", topk=20)

    # Print optimization solution
    res.plot_vrp_solution(
        distance=vrp.distance,
        n=vrp.n,
        vehicle_count=vrp.vehicle_count,
        depot=0,
        title="VRP Solution (QAOA)",
    )


if __name__ == "__main__":
    main()
