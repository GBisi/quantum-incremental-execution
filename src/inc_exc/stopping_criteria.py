import math
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple

# Type alias for frequency counts.
FreqCounts = Dict[str, int]

########################################
# Distribution and Merging Utilities
########################################

def normalised_frequency(frequency: FreqCounts) -> Dict[str, float]:
    """
    Convert frequency counts (ints) into a normalized probability distribution.
    If the total count is zero, returns zeros for each key.
    """
    total = sum(frequency.values())
    if total == 0:
        return {k: 0.0 for k in frequency}
    return {k: v / total for k, v in frequency.items()}

def merge_distributions(prev: FreqCounts, new: FreqCounts) -> FreqCounts:
    """
    Merge two frequency distributions by summing integer counts for each key.
    """
    result = prev.copy()
    for key, value in new.items():
        result[key] = result.get(key, 0) + value
    return result

def subtract_distributions(main: FreqCounts, sub: FreqCounts) -> FreqCounts:
    """
    Subtract the counts of `sub` from `main` for each key.
    """
    result = main.copy()
    for key, value in sub.items():
        result[key] = result.get(key, 0) - value
    return result

########################################
# Distance Metrics
########################################

def total_variation_distance(freq1: FreqCounts, freq2: FreqCounts) -> float:
    """
    Compute the total variation distance (TVD) between two frequency distributions.
    TVD is defined as half the L1 distance between the normalized distributions.
    """
    keys = set(freq1.keys()).union(freq2.keys())
    norm1 = normalised_frequency(freq1)
    norm2 = normalised_frequency(freq2)
    return sum(abs(norm1.get(k, 0.0) - norm2.get(k, 0.0)) for k in keys) / 2.0

def kl_divergence(freq1: FreqCounts, freq2: FreqCounts, epsilon: float = 1e-10) -> float:
    """
    Compute the Kullback–Leibler divergence between two distributions.
    Uses an epsilon value for numerical stability when q has zero values.
    """
    p = normalised_frequency(freq1)
    q = normalised_frequency(freq2)
    divergence = 0.0
    for key in set(p.keys()).union(q.keys()):
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        if p_val > 0:
            adjusted_q = q_val if q_val > 0 else epsilon
            divergence += p_val * math.log(p_val / adjusted_q)
    return divergence

def chi_square_distance(freq1: FreqCounts, freq2: FreqCounts, epsilon: float = 1e-10) -> float:
    """
    Compute the Chi-Square distance between two normalized frequency distributions.
    """
    p = normalised_frequency(freq1)
    q = normalised_frequency(freq2)
    chi_sq = 0.0
    for key in set(p.keys()).union(q.keys()):
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        chi_sq += (p_val - q_val) ** 2 / (p_val + q_val + epsilon)
    return chi_sq

########################################
# Convergence (Stopping) Criterion Factories
########################################

def delta_distance_criterion(threshold: float = 0.05,
                             offset: int = 1,
                             dist_func: Callable[[FreqCounts, FreqCounts], float] = total_variation_distance
                            ) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts], bool]:
    """
    Factory for the delta distance criterion.

    Returns:
        A function that takes (history, cumulative, last_result) and returns True
        if the computed distance (after subtracting 'offset' snapshots) is below the threshold.
    """
    def criterion(history: List[FreqCounts], cumulative: FreqCounts, last_result: FreqCounts) -> bool:
        if offset < 1:
            raise ValueError("Offset must be at least 1.")
        comparing = cumulative.copy()
        actual_offset = min(offset, len(history))
        for i in range(actual_offset):
            comparing = subtract_distributions(comparing, history[-(i + 1)])
        distance = dist_func(cumulative, comparing)
        return distance < threshold, {"distance":distance, "stopping":distance < threshold}
    return criterion

def quorum_criterion(threshold: float = 0.05,
                     offset: int = 1,
                     window: int = 1,
                     quorum_ratio: float = 0.5,
                     dist_func: Callable[[FreqCounts, FreqCounts], float] = total_variation_distance
                    ) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts], bool]:
    """
    Factory for a quorum-based convergence criterion. For each snapshot in the window,
    it applies a delta distance test (with increasing offset) and checks if a sufficient proportion of the tests pass.
    
    Returns:
        A function that takes (history, cumulative, last_result) and returns True if the number
        of tests passing the delta distance criterion meets the quorum_ratio.
    """
    def criterion(history: List[FreqCounts], cumulative: FreqCounts, last_result: FreqCounts) -> bool:
        votes = 0
        # Vary the offset over the window
        for additional_offset in range(1, window + 1):
            current_offset = additional_offset + (offset - 1)
            # Create a delta distance function for the current offset and test it.
            if delta_distance_criterion(threshold, current_offset, dist_func)(history, cumulative, last_result):
                votes += 1
        return votes >= int(np.ceil(quorum_ratio * window)), {"votes": votes, "stopping": votes >= int(np.ceil(quorum_ratio * window))}
    return criterion

def delta_distance_moving_average_criterion(threshold: float = 0.05,
                                            offset: int = 1,
                                            alpha: Optional[float] = 0.5,
                                            window: Optional[int] = None,
                                            dist_func: Callable[[FreqCounts, FreqCounts], float] = total_variation_distance
                                           ) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts], Tuple[float, bool]]:
    """
    Factory for a convergence criterion based on the exponentially weighted moving average (EWMA)
    of distances computed between the current cumulative distribution and past cumulative snapshots.
    
    Returns:
        A function that takes (history, cumulative, last_result) and returns a tuple:
        (moving_average_distance, is_converged), where is_converged is True if the moving average is below the threshold.
    """
    def criterion(history: List[FreqCounts], cumulative: FreqCounts, last_result: FreqCounts) -> bool:
        # Build cumulative snapshots.
        cumulative_history: List[FreqCounts] = []
        cum: FreqCounts = {}
        for snapshot in history:
            cum = merge_distributions(cum, snapshot)
            cumulative_history.append(cum.copy())

        if offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        # Adjust offset if larger than the history
        offset_adjusted = offset if offset <= len(cumulative_history) else len(cumulative_history)
        if offset_adjusted == 1:
            available_history = cumulative_history[:]  # Use all snapshots if offset_adjusted is 1
        else:
            available_history = cumulative_history[:-offset_adjusted]
        if not available_history:
            # raise ValueError(f"No cumulative snapshots available after applying the offset: offset={offset}, history={history}, offset_adjusted={offset_adjusted}, available_history={available_history}")
            available_history= cumulative_history[:]

        # If a window is specified, select the last 'window' snapshots.
        if window is not None:
            if window < 1:
                raise ValueError("Window must be at least 1 if specified.")
            selected_snapshots = available_history[-window:] if len(available_history) >= window else available_history
        else:
            selected_snapshots = available_history

        distances = [dist_func(cumulative, past_cumulative) for past_cumulative in selected_snapshots]
        if alpha is None:
            moving_avg_distance = sum(distances) / len(distances)
        else:
            weighted_sum = 0.0
            total_weight = 0.0
            # Reverse distances to give more weight to recent snapshots.
            for i, d in enumerate(reversed(distances)):
                weight = alpha ** i
                weighted_sum += weight * d
                total_weight += weight
            moving_avg_distance = weighted_sum / total_weight if total_weight > 0 else 0.0

        return moving_avg_distance < threshold, {"distance": moving_avg_distance, "stopping": moving_avg_distance < threshold}
    return criterion

########################################
# Utility Functions for Incremental Execution
########################################

def constant_stability_criterion(
                                k: int = 1) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts], int]:
    """
    A simple stability criterion that requires k consecutive stable iterations.
    This factory returns a function that ignores its inputs and returns k.
    """
    return lambda history, cumulative, last, info: k

def constant_next_shots(
                        default_shots: int = 10) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts], int]:
    """
    A simple next-shots function that always returns a constant number of shots.
    This factory returns a function that ignores its inputs and returns default_shots.
    """
    return lambda history, cumulative, last, info: default_shots

def square_wave_shots(default_shots: int = 10,
                      factor: float = 0.5) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts], int]:
    """
    A function that returns a number of shots based on a square wave pattern.
    The number of shots is default_shots if stopping is False, and
    int(default_shots * factor) if stopping is True.
    """
    def next_shots(history: List[FreqCounts], cumulative: FreqCounts, last: FreqCounts, info: Any) -> int:
        if info["stopping"]:
            return int(default_shots * factor)
        else:
            return default_shots
    return next_shots

def dynamic_shot_allocation(
    min_initial_iterations: int = 1,
    dist_func: Callable[[FreqCounts, FreqCounts], float] = total_variation_distance
) -> Callable[[List[FreqCounts], FreqCounts, FreqCounts, Any], int]:
    """
    Adaptive shot allocation policy based on the observed convergence rate of the empirical distribution.

    After a fixed number of initial iterations, estimates the next batch size using the ratio of successive divergences:
        \varepsilon_i = D(cumulative_i, cumulative_{i-1})
        \varepsilon_{i-1} = D(cumulative_{i-1}, cumulative_{i-2})
    and scales the next shot count:
        S_{i+1} = (\varepsilon_i^2 / \varepsilon_{i-1}) * S_i

    Args:
        min_initial_iterations: number of iterations to run with constant batch size before adapting.
        dist_func: function to compute divergence between two frequency distributions.

    Returns:
        A function next_shots(history, cumulative, last_result, info) -> int producing the
        adaptive shot count for the next iteration.
    """
    def next_shots(
        history: List[FreqCounts],
        cumulative: FreqCounts,
        last_result: FreqCounts,
        info: Any
    ) -> int:
        # Use fixed shots for the first few iterations
        if len(history) <= min_initial_iterations:
            return sum(last_result.values())
    
        # Ensure there is a previous batch to compute \varepsilon_{i-1}
        if len(history) < 2:
            return sum(last_result.values())

        # Compute cumulative count up to previous iteration
        cumulative_prev = subtract_distributions(cumulative, last_result)

        # Compute current divergence \varepsilon_i
        eps_i = dist_func(cumulative, cumulative_prev)

        # Previous batch's counts
        prev_last = history[-2]
        # Compute cumulative count two iterations ago
        cumulative_prev_prev = subtract_distributions(cumulative_prev, prev_last)
        # Compute previous divergence \varepsilon_{i-1}
        eps_prev = dist_func(cumulative_prev, cumulative_prev_prev)

        S_i = sum(last_result.values())
        # Avoid division by zero or non-positive divergence
        if eps_prev <= 0:
            return S_i

        # Estimate next shots: S_{i+1} = (eps_i^2 / eps_prev) * S_i
        S_next = int((eps_i ** 2 / eps_prev) * S_i)

        return max(S_next, sum(history[0].values()))

    return next_shots
