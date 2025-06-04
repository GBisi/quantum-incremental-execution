from incremental_execution import IncrementalExecution
from stopping_criteria import *
import random
import math
import numpy as np
from collections import Counter
from typing import Dict, List
from itertools import product

# --- Code under test ---
# [Assume the full code from the previous message is already defined here, including:
#   - the IncrementalExecution class,
#   - the utility functions normalised_frequency, merge_distributions, subtract_distributions,
#   - the distance metrics: total_variation_distance, kl_divergence, chi_square_distance,
#   - the stopping criteria: delta_distance, quorum, delta_distance_moving_average,
#   - the next_shots and stability criteria functions (constant_stability_criterion, constant_next_shots).]

# Runner: produce random bitstrings of fixed length L (using bias to simulate unknown distribution).
def random_bitstring_runner(*, shots: int, bitstring_length: int = 10) -> Dict[str, int]:
    """
    Generate `shots` random bitstrings of length `bitstring_length` with a biased probability distribution.
    In this example, each bit is '0' with probability 0.3 and '1' with probability 0.7.
    """
    frequency = Counter()
    p0 = 0.3
    p1 = 0.7
    for _ in range(shots):
        # Generate a bitstring of length bitstring_length.
        bits = ['0' if random.random() < p0 else '1' for _ in range(bitstring_length)]
        bitstring = ''.join(bits)
        frequency[bitstring] += 1
    return dict(frequency)

def true_distribution(bitstring_length: int = 10, p0: float = 0.3, p1: float = 0.7) -> Dict[str, float]:
    """
    Compute the true distribution for all bitstrings of length `bitstring_length`
    based on the given bias for 0 and 1.
    """
    true_dist = {}
    for bits in product('01', repeat=bitstring_length):
        bs = ''.join(bits)
        count0 = bs.count('0')
        count1 = bs.count('1')
        # Using the product rule.
        true_dist[bs] = (p0 ** count0) * (p1 ** count1)
    return true_dist

def total_shots_from_result(result: Dict[str, int]) -> int:
    """Compute the total number of shots from the cumulative frequency result."""
    return sum(result.values())

def main():
    
    random.seed(42)  # For reproducibility

    # Constants
    DEFAULT_SHOTS = 500
    MAX_SHOTS = 1000000000
    BITSTRING_LENGTH = 10
    threshold = 0.01
    STABILITY = 10
    true_dist = true_distribution(BITSTRING_LENGTH, p0=0.3, p1=0.7)

    # --- Test 1 ---
    incremental_exec1 = IncrementalExecution(
        stopping_criterion=lambda history, cumulative, last: delta_distance(
            history, cumulative, last, threshold=threshold, offset=1),
        stability_criterion=constant_stability_criterion(STABILITY),
        next_shots=constant_next_shots(DEFAULT_SHOTS),
        default_shots=DEFAULT_SHOTS,
        max_shots=MAX_SHOTS
    )
    result1 = incremental_exec1.run(
        runner=lambda *args, shots, **kwargs: random_bitstring_runner(shots=shots, bitstring_length=BITSTRING_LENGTH)
    )
    shots1 = total_shots_from_result(result1)
    error1 = total_variation_distance(result1, true_dist)
    print(f"Test 1: stopping=delta_distance(offset=1), stability=constant_stability_criterion, next_shots=constant_next_shots, default_shots={DEFAULT_SHOTS}, max_shots={MAX_SHOTS} | total_shots: {shots1} | error: {error1:.6f}")

    # --- Test 2 (with initial guess) ---
    initial_guess = random_bitstring_runner(shots=DEFAULT_SHOTS, bitstring_length=BITSTRING_LENGTH)
    incremental_exec2 = IncrementalExecution(
        stopping_criterion=lambda history, cumulative, last: delta_distance(
            history, cumulative, last, threshold=threshold, offset=1),
        stability_criterion=constant_stability_criterion(STABILITY),
        next_shots=constant_next_shots(DEFAULT_SHOTS),
        default_shots=DEFAULT_SHOTS,
        max_shots=MAX_SHOTS
    )
    result2 = incremental_exec2.run(
        runner=random_bitstring_runner,
        initial_guess=initial_guess,
        bitstring_length=BITSTRING_LENGTH
    )
    shots2 = total_shots_from_result(result2)
    error2 = total_variation_distance(result2, true_dist)
    print(f"Test 2: initial_guess provided, stopping=delta_distance(offset=1), stability=constant_stability_criterion, next_shots=constant_next_shots, default_shots={DEFAULT_SHOTS}, max_shots={MAX_SHOTS} | total_shots: {shots2} | error: {error2:.6f}")

    # --- Test 3 (decorator approach) ---
    incremental_exec3 = IncrementalExecution(
        stopping_criterion=lambda history, cumulative, last: delta_distance(
            history, cumulative, last, threshold=threshold, offset=2),
        stability_criterion=constant_stability_criterion(STABILITY),
        next_shots=constant_next_shots(DEFAULT_SHOTS),
        default_shots=DEFAULT_SHOTS,
        max_shots=MAX_SHOTS
    )
    @incremental_exec3
    def decorated_random_runner(*, shots: int, bitstring_length: int = BITSTRING_LENGTH) -> Dict[str, int]:
        return random_bitstring_runner(shots=shots, bitstring_length=bitstring_length)
    result3 = decorated_random_runner()
    shots3 = total_shots_from_result(result3)
    error3 = total_variation_distance(result3, true_dist)
    print(f"Test 3: decorator, stopping=delta_distance(offset=2), stability=constant_stability_criterion, next_shots=constant_next_shots, default_shots={DEFAULT_SHOTS}, max_shots={MAX_SHOTS} | total_shots: {shots3} | error: {error3:.6f}")

    # --- Test 4 (dynamic next_shots) ---
    def dynamic_next_shots(history: List[Dict[str, int]], cumulative: Dict[str, int], last: Dict[str, int]) -> int:
        total = sum(cumulative.values())
        if total < 50:
            return 100
        elif total < 200:
            return 200
        else:
            return 300

    incremental_exec4 = IncrementalExecution(
        stopping_criterion=lambda history, cumulative, last: delta_distance(
            history, cumulative, last, threshold=threshold, offset=1),
        stability_criterion=constant_stability_criterion(STABILITY),
        next_shots=dynamic_next_shots,
        default_shots=DEFAULT_SHOTS,
        max_shots=MAX_SHOTS
    )
    result4 = incremental_exec4.run(
        runner=random_bitstring_runner,
        bitstring_length=BITSTRING_LENGTH
    )
    shots4 = total_shots_from_result(result4)
    error4 = total_variation_distance(result4, true_dist)
    print(f"Test 4: stopping=delta_distance(offset=1), stability=constant_stability_criterion, next_shots=dynamic_next_shots, default_shots={DEFAULT_SHOTS}, max_shots={MAX_SHOTS} | total_shots: {shots4} | error: {error4:.6f}")

if __name__ == "__main__":
    main()