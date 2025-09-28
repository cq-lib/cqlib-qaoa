# Cqlib Algorithm

本项目是由中电信量子集团开发团队打造的基于 **[Cqlib](https://gitee.com/cq-lib/cqlib)** 的量子算法包，包含新建优化任务，生成哈密顿量，创建量子电路，执行量子电路，测量采样，返回优化结果等功能，目前已实现 QAOA，后续将持续集成 VQE 等量子算法。

---

## 安装说明

使用 pip 下载 cqlib-algorithm 包:

```bash
pip install cqlib-algorithm
```

## 结构说明

- **algorithms**：算法主体模块，内置QAOA算法主体、能量值评估、优化结果。
- **ansatz**：量子线路模块，内置QAOA专用线路生成。
- **execution**：算法执行后端模块，内置`LocalRunner`后端(statevector模拟器)、`TianYanRunner`后端( **[中电信“天衍”量子计算云平台](https://qc.zdxlz.com)** )。
- **mappings**：问题模型转化模块，内置QUBO转化、代价哈密顿量生成。
- **optimizers**：优化器模块，内置 `SPSA` / `COBYLA` / `Nelder-Mead`优化器。
- **problems**：问题定义模块，内置`MaxCut` / `TSP` / `VRP`三类典型优化问题模型。
- **results**：问题解码模块，内置`MaxCut` / `TSP` / `VRP`三类典型问题的优化结果解码过程。
- **visualization**：可视化模块，内置量子线路可视化、迭代历史曲线可视化、概率分布可视化、问题模型可视化。

---

## 应用示例：使用 QAOA 求解 MaxCut 问题。

```python
from cqlib_algorithm.problems.maxcut import MaxCut
from cqlib_algorithm.mappings.convert import maxcut_to_qubo, qubo_to_ising
from cqlib_algorithm.visualization.maxcut_plot import plot_maxcut
from cqlib_algorithm.execution import LocalRunner, TianYanRunner
from cqlib_algorithm.algorithms.qaoa import QAOASolver, QAOAConfig
from cqlib_algorithm.optimizers.options import OptimizerOptions

def main():
    # ---- 定义 Maxcut 问题 ----
    weights = {(0,1):1, (1,2):1, (2,3):1, (3,0):1}
    mc = MaxCut(n=4, weights=weights)

    # ---- 绘制模型图 ----
    plot_maxcut(n=mc.n, weights=mc.weights, partition={}, title="MaxCut Problem")

    # ---- Maxcut -> QUBO ----
    qubo  = maxcut_to_qubo(mc)
    print(qubo)

    # ---- QUBO -> Ising ----
    ising = qubo_to_ising(qubo)
    print(ising)

    # ---- 选择优化器 ----
    # SPSA
    opt_cfg = OptimizerOptions(name="spsa", options={"maxiter": 100, "a": 0.2, "c": 0.2})

    # ---- QAOA求解 ----
    solver = QAOASolver(ising, 
                        runner=LocalRunner(), 
                        qaoa_cfg=QAOAConfig(reps=5, mixer="x"), 
                        opt_cfg=opt_cfg)

    res = solver.run()

    # ---- 优化结果 ----
    # 1) 结果打印
    res.print_result()
    
    # 2) 收敛曲线可视化
    res.plot_history(title="Optimization History")

    # 3) 概率分布可视化
    res.plot_probability(title="QAOA Probability (best θ)", topk=20)

    # 4) 解决方案可视化
    res.plot_maxcut_solution(n=mc.n, weights=mc.weights, title="MaxCut Solution(QAOA)")

if __name__ == "__main__":
    main()
```

---

## License

Apache License 2.0，详见 [LICENSE](LICENSE)。

---

## Contributing

欢迎贡献！如需改进功能或修复缺陷，请提交 Issue 或发起 Pull Request。

---
