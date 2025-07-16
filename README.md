# IncrementalExecution: How Many Shots Are Enough for a Quantum Circuit?

## 🧠 Overview

**IncrementalExecution** is a Python framework for dynamically determining the number of execution shots required by a quantum circuit to estimate its output distribution with sufficient accuracy. It introduces an **online, black-box optimization strategy** that avoids costly over-execution by detecting the **point of diminishing returns**, where additional shots no longer yield statistically significant information.


## 🚀 Key Features

- **Black-box execution model**: No prior assumptions about the quantum circuit or noise model.
- **Online shot management**: Dynamically stop execution once empirical distribution stabilizes.
- **Modular policy design**:
  - Customizable stopping criteria
  - Stability enforcement
  - Adaptive shot allocation
- **Pluggable interface** for easy integration with hybrid quantum-classical loops (e.g., VQA, QML).
- **Open-source benchmark** with 1.7M+ experiments across 12,000+ configurations on 140 circuit-backend pairs.

## 📦 Installation

```bash
git clone https://github.com/GBisi/quantum-incremental-execution.git
cd quantum-incremental-execution
pip install -r requirements.txt
```

## 🔧 Usage

### Minimal Example

```python
from incremental_execution import IncrementalExecution

# Define your own policy components or use built-ins
incremental = IncrementalExecution(
    stopping_criterion=stop_criterion_fn,
    stability_criterion=stability_fn,
    allocate_shots=allocation_fn,
    default_shots=100,
    max_shots=20000
)

result = incremental.run(runner=my_runner)
```

### With Decorators

```python
@incremental
def my_runner(shots):
    return run_quantum_circuit(shots=shots)

result = my_runner()
```

## 🧪 Benchmark Dataset

The repository includes:
- Execution traces of 140 quantum circuit-backend pairs
- Parameter sweeps over 12,474 configurations
- Analysis scripts for evaluating total variation distance (TVD) and shot savings

📂 See the results directory for full experimental data and analysis.

## 🧭 When Should You Use This?

- To optimize execution cost on NISQ devices
- To balance fidelity vs. shot count in real-time
- In hybrid algorithms where circuit output distributions evolve gradually (e.g., VQE, QML)
- As a general-purpose replacement for fixed-shot quantum circuit execution

## 📄 Citation

If you use this framework in your research, please cite:

> Giuseppe Bisicchia, Alessandro Bocci, Ernesto Pimentel, and Antonio Brogi.  
> *How Many Shots Are Enough for a Quantum Circuit?*  