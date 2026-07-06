# Project Custom Rules: DyRF-BO Proximity UQ

## 1. Pragmatic UQ Research TDD Workflow
Always implement any new UQ algorithms, estimators, or mathematical helpers using the following Test-Driven Development workflow:

1. **API Contract**: Define the function/class signature, parameters, and mathematical invariants in docstrings.
2. **Red Phase (Parity Test First)**: Before implementing the logic, write a unit test in `tests/` that checks correctness against manual calculations, mock models, or slower reference packages. Verify that it fails.
3. **Green Phase**: Implement the minimal functional logic to make the test pass.
4. **Refactor & Optimize**: Vectorize operations, implement CPU/GPU (NumPy/CuPy) backends, and run optimizations using the test to guarantee zero regressions.
5. **E2E Smoke Verification**: Ensure that the code integrates with the main orchestrator (`Uncertainty_Quantification.py`) and passes `bash run_tests.sh`.

## 2. Autonomous Subagent Activation
Proactively evaluate incoming tasks and automatically spawn the appropriate specialized subagent (`self` clone with injected prompt) to execute the work:
* **HPC Performance Auditor**: Activate automatically whenever writing or refactoring code that runs on the cluster, involves CUDA/CuPy operations, handles GMM Monte Carlo sampling, or requires performance/memory/latency optimization of large-scale loops.
* **Scientific Verification Auditor (TDD)**: Activate automatically whenever implementing new UQ baselines, mathematical estimators, evaluation metrics, or testing suites. The subagent must run under a strict Test-Driven Development (TDD) loop and verify floating-point tolerances and CPU-GPU parity.
