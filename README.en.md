# Cqlib-QAOA
See the Chinese version at [Chinese version](README.md)。

This project is a QAOA quantum algorithm package based on **[Cqlib](https://github.com/cq-lib)**, created by the development team of China Telecom Quantum Group. It includes functions such as creating new optimization tasks, generating Hamiltonians, creating quantum circuits, executing quantum circuits, measuring and sampling, and returning optimization results. 

---

## Installation

Requirements: Python 3.10 or later.

Install `cqlib-qaoa` using pip:

```bash
pip install cqlib-qaoa
```

## Package Structure

- **algorithms**: Core algorithm module. Includes QAOA main loop, energy evaluation, and optimization results.
- **ansatz**: Quantum circuit builders. Includes QAOA-specific circuit generation.
- **execution**: Execution backends. Includes LocalRunner (statevector simulator) and TianYanRunner (**[China Telecom “TianYan” quantum cloud platform](https://qc.zdxlz.com)**).
- **mappings**: Problem mappings. Includes QUBO conversion and cost-Hamiltonian generation.
- **optimizers**: Optimizers: `SPSA` / `COBYLA` / `Nelder-Mead`.
- **problems**: Problem definitions for `MaxCut` / `TSP` / `VRP`.
- **results**: Result decoding for `MaxCut` / `TSP` / `VRP`.
- **transpiler**: Circuit decomposition utilities for two-qubit rotations such as RZZ, RXX, and RYY.
- **visualization**: Visualization of optimization history, probability distributions, and problem graphs.

---

## Example: Solve MaxCut with QAOA

```python
from cqlib_qaoa.problems import MaxCut
from cqlib_qaoa.mappings import maxcut_to_qubo, qubo_to_ising
from cqlib_qaoa.visualization import plot_maxcut
from cqlib_qaoa.execution import LocalRunner
from cqlib_qaoa.algorithms import QAOASolver, QAOAConfig
from cqlib_qaoa.optimizers import OptimizerOptions

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

    # 5) Select optimizer
    # SPSA
    opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 50, "a": 0.2, "c": 0.2})

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
    res.plot_maxcut_solution(n=mc.n, weights=mc.weights, title="MaxCut Solution(QAOA)")

if __name__ == "__main__":
    main()

```

To run on China Telecom's TianYan quantum cloud platform, replace `LocalRunner()` with:

```python
from cqlib_qaoa.execution import TianYanRunner

runner = TianYanRunner(login_key="YOUR_KEY", machine="tianyan_sw")
```

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---
