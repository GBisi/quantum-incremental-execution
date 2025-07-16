from tqdm import tqdm
from src.shotter import Shotter
import json
import os
import time
import hashlib
import random

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

circuit_fun = generate_qc

max_shots = 20_000
shots_blocks = [50]
dir = "results"
git = True

circuits_names = ["benchmark-qpeexact", "grover", "qft", "random-60"]
circuits_sizes = [2,4,5,6,8,10,12,14,15,16]
circuits_seeds = [42]

backends = [{"local": ["fake_fez"]},{"local": ["fake_marrakesh"]}] 
# backends = [{"local": ["fake_sherbrooke"]},{"local": ["fake_torino"]},{"local": ["fake_kyiv"]}] 

policy = "multiplier"
wait_time = 5
multiprocess = False

# backends = [{"local": ["fake_sherbrooke", "fake_torino", "fake_kyiv"]}]

# backends = [{"local": ["aer_simulator_noisy-100", "aer_simulator_noisy-95", "aer_simulator_noisy-90",
#              "aer_simulator_noisy-85", "aer_simulator_noisy-80", "aer_simulator_noisy-75", "aer_simulator_noisy-70",
#              "aer_simulator_noisy-65", "aer_simulator_noisy-60", "aer_simulator_noisy-55", "aer_simulator_noisy-50",
#              "aer_simulator_noisy-45", "aer_simulator_noisy-40", "aer_simulator_noisy-35", "aer_simulator_noisy-30",
#              "aer_simulator_noisy-25", "aer_simulator_noisy-20", "aer_simulator_noisy-15", "aer_simulator_noisy-10",
#              "aer_simulator_noisy-5"]}]

# backends = [{"local": ["aer_simulator_noisy-100", "aer_simulator_noisy-80", "aer_simulator_noisy-60",   "aer_simulator_noisy-50", "aer_simulator_noisy-40", "aer_simulator_noisy-20", "aer_simulator_noisy-10"]}]


circuits = [(n,s,seed) for seed in circuits_seeds for s in circuits_sizes for n in circuits_names]

shotter = Shotter(raise_exc=False)

try:
    os.mkdir(dir)
except FileExistsError:
    pass

if __name__ == "__main__":

    for shots_block in tqdm(shots_blocks, desc="Shots Block", unit="block"):
        for circuit in tqdm(circuits, desc="Circuits", unit="circuit"):
            print(f"*** Ground Truth for {circuit} ***")
            random.seed(circuit[2])
            qc = circuit_fun(circuit[0], circuit[1], circuit[2])
            while True:
                try:
                    gt = shotter.run(
                        circuit=qc.copy(),
                        shots=max_shots,
                        backends={"local":["aer_simulator"]},
                        policy = policy,
                        blob = None,
                        multiprocess=False,
                        single_thread=True,
                        seed = circuit[2],
                    )[0]
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Retrying... in {wait_time} seconds")
                    time.sleep(wait_time)
                    shotter = Shotter(raise_exc=False)
                    continue
            
            for backend in tqdm(backends, desc="Backends", unit="backend"):
                results = {"metadata": {}, "data": [], "individual": []}
                iterations = max_shots // shots_block
                print("\n")
                print(f"*** Running {shots_block} shots for {circuit} on {backend} ***")
                for i in tqdm(range(iterations), desc="Iterations", unit="iteration"):
                    while True:
                        try:
                            res = shotter.run(
                                circuit=qc.copy(),
                                shots=shots_block,
                                backends=backend,
                                policy = policy,
                                blob = None,
                                multiprocess = multiprocess,
                                single_thread=not multiprocess,
                                seed = circuit[2],
                            )
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                            print(f"Retrying... in {wait_time} seconds")
                            time.sleep(wait_time)
                            shotter = Shotter(raise_exc=False)
                            continue
                    results["data"].append(res[0])
                    results["individual"].append(res[1])
                    
                _shots_performed = []
                agg = 0
                for r in results["data"]:
                    agg += sum(r.values())
                    _shots_performed.append(agg)
                    
                results["metadata"]["shots"] = {
                    "shots_block": shots_block,
                    "total_shots": max_shots,
                    "shots_performed": _shots_performed
                }
                results["metadata"]["iterations"] = iterations
                results["metadata"]["circuit"] = str(circuit[0])
                results["metadata"]["circuit_size"] = circuit[1]
                results["metadata"]["backends"] = backend
                results["metadata"]["ground_truth"] = gt
                results["metadata"]["policy"] = policy
                results["metadata"]["seed"] = circuit[2]
                        
                # Save results to a JSON file
                filename = f"{circuit}-{shots_block}-{max_shots}-{hashlib.md5(str(backend).encode()).hexdigest()}"
                with open(f"{dir}/{filename}.json", "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Results saved to results/{filename}.json")
                print("\n")
                time.sleep(1)
    
                # git push
                if git:
                    os.system("git pull")
                    os.system("git add .")
                    os.system("git commit -m 'update results'")
                    os.system("git push")