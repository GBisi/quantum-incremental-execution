from src.inc_exc.incremental_execution import IncrementalExecution
from src.inc_exc.stopping_criteria import *
import random
import math
import numpy as np
from collections import Counter
from typing import Dict, List
from itertools import product
from src.shotter import Shotter

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QFT, GroverOperator
from mqt.bench import get_benchmark

def generate_random_circuit(num_qubits: int, depth: int, measure: bool = True, seed: int = None):
    """
    Generate a random quantum circuit with the specified number of qubits and depth.
    
    Parameters:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Depth (number of layers) of the circuit.
        measure (bool): If True, append measurement operations to all qubits.
        seed (int): Optional seed for reproducibility.
        
    Returns:
        QuantumCircuit: A randomly generated quantum circuit.
    """
    circuit = random_circuit(num_qubits, depth, measure=measure, seed=seed)
    return circuit

def generate_qft_circuit(n):
    qft = QFT(n, approximation_degree=1)
    qft_circuit = QuantumCircuit(qft.num_qubits)
    qft_circuit.append(qft, range(qft.num_qubits))
    qft_circuit.measure_all()
    return qft_circuit

def generate_grover_circuit(n):
    
    oracle = QuantumCircuit(n)
    for i in range(n):
        if random.choice([True, False]):
            oracle.x(i)
    
    grover = GroverOperator(oracle)
    grover_circuit = QuantumCircuit(n)
    grover_circuit.append(grover, range(n))

    grover_circuit.measure_all()

    return grover_circuit

def generate_qc(n, s, seed):
    if n == "qft":
        return generate_qft_circuit(s)
    elif n == "grover":
        return generate_grover_circuit(s)
    elif n.startswith("random-"):
        depth = int(n.split("-")[1])
        return generate_random_circuit(num_qubits=s, depth=depth, seed=seed)
    elif n.startswith("benchmark-"):
        benchmark_name = n.split("-")[1]
        return get_benchmark(benchmark_name=benchmark_name, level="alg", circuit_size=s)
    else:
        raise ValueError(f"Unknown circuit type: {n}")

shotter = Shotter()
def runner(*, shots: int, backend: str = "fake_sherbrooke", circuit: Any) -> Dict[str, int]:
    return shotter.run(circuit=circuit, shots=shots, backends={"local":[backend]})[0]
    

def total_shots_from_result(result: Dict[str, int]) -> int:
    """Compute the total number of shots from the cumulative frequency result."""
    return sum(result.values())

def main():
    seed = 42
    random.seed(seed)  # For reproducibility


    # Constants
    DEFAULT_SHOTS = 500
    MAX_SHOTS = 100_000
    STABILITY = 10
    THRESHOLD = 0.01
    
    backend = "fake_sherbrooke"
    circuit = "random-30"
    qubits = 5
    
    # Generate a random quantum circuit
    qc = generate_qc(circuit, qubits, seed)
    
    true_dist = shotter.run(circuit=qc, shots=MAX_SHOTS, backends={"local": ["aer_simulator"]})[0]

    # --- Test 1 ---
    incremental_exec1 = IncrementalExecution(
        stopping_criterion=lambda history, cumulative, last: delta_distance(
            history, cumulative, last, threshold=THRESHOLD, offset=1),
        stability_criterion=constant_stability_criterion(STABILITY),
        next_shots=constant_next_shots(DEFAULT_SHOTS),
        default_shots=DEFAULT_SHOTS,
        max_shots=MAX_SHOTS,
        verbose=True,
    )
    result1 = incremental_exec1.run(
        runner=lambda *args, shots, **kwargs: runner(*args, shots=shots, backend=backend, circuit=qc),
    )
    shots1 = total_shots_from_result(result1)
    error1 = total_variation_distance(result1, true_dist)
    
    print(f"Test 1: Shots = {shots1}, Error = {error1:.4f}")

if __name__ == "__main__":
    main()