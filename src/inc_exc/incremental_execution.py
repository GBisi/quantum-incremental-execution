import logging
from functools import wraps
from typing import Callable, Dict, List, Any, Optional

# Type alias for frequency counts.
FreqCounts = Dict[str, int]

# Aliases for clarity.
RunnerType = Callable[..., FreqCounts]
StoppingCriterionType = Callable[[List[FreqCounts], FreqCounts, FreqCounts], bool]
StabilityCriterionType = Callable[[List[FreqCounts], FreqCounts, FreqCounts], int]
NextShotsType = Callable[[List[FreqCounts], FreqCounts, FreqCounts], int]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if not already added
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

class IncrementalExecution:
    """
    A class for executing a given runner function in incremental batches until convergence 
    or a maximum number of iterations is reached.

    Parameters:
        stopping_criterion: A callable that takes (history, cumulative_result, last_result) and returns 
                            True if the current batch is considered stable (i.e., converged for that batch).
        stability_criterion: A callable that takes (history, cumulative_result, last_result) and returns 
                             an integer threshold for the number of consecutive stable iterations required.
        next_shots: A callable that takes (history, cumulative_result, last_result) and returns an 
                    integer indicating the number of shots (iterations) to run for the next batch.
        default_shots: An integer indicating the default number of shots to run on the first iteration.
        max_shots: (Optional) Maximum total number of shots allowed. If None, there is no maximum.
        runner: (Optional) A callable that performs one execution step. It must accept a keyword argument 
                `shots` and return a dictionary containing frequency counts.
        verbose: A boolean flag to enable verbose logging of the internal state during execution.
    """

    def __init__(self,
                 stopping_criterion: StoppingCriterionType,
                 stability_criterion: StabilityCriterionType,
                 next_shots: NextShotsType,
                 default_shots: int,
                 max_shots: Optional[int] = None,
                 runner: Optional[RunnerType] = None,
                 verbose: bool = True) -> None:
        self.stopping_criterion = stopping_criterion
        self.stability_criterion = stability_criterion
        self.next_shots_fn = next_shots  
        self.default_shots = default_shots
        self.max_shots = max_shots
        self.runner = runner
        self.verbose = verbose

    def run(self,
            runner: Optional[RunnerType] = None,
            initial_guess: Optional[FreqCounts] = None,
            *args,
            **kwargs) -> FreqCounts:
        """
        Execute the runner incrementally until the stopping criteria are met or the maximum shots are reached.

        Parameters:
            runner: Optional runner function to override the one stored in the instance.
            initial_guess: Optional initial frequency distribution used to adjust the number of shots on the first iteration.
            *args, **kwargs: Additional arguments to pass to the runner function.

        Returns:
            A dictionary representing the cumulative frequency counts accumulated over all iterations.
        """
        active_runner = runner or self.runner
        if active_runner is None:
            raise ValueError("A runner must be provided either during initialization or as an argument to run().")

        cumulative_result: FreqCounts = {}
        history: List[FreqCounts] = []
        current_shots = 0
        iteration_count = 0
        stop_flag = False
        stability_counter = 0
        max_shots_allowed = self.max_shots if self.max_shots is not None else float('inf')
        info = None

        # Determine the initial number of shots.
        next_shots = self.default_shots
        if initial_guess is not None:
            # Use the initial guess to suggest a starting shot count.
            suggested_shots = self.next_shots_fn([initial_guess], initial_guess, initial_guess, info)
            next_shots = max(next_shots, suggested_shots)
            if self.verbose:
                logger.info("Initial guess provided; suggested_shots=%d, default_shots=%d -> next_shots=%d", 
                             suggested_shots, self.default_shots, next_shots)

        # Main execution loop.
        while (not stop_flag) and (current_shots < max_shots_allowed):
            iteration_count += 1
            # Ensure that we do not exceed the maximum allowed shots.
            next_shots = min(next_shots, max_shots_allowed - current_shots)
            if self.verbose:
                logger.info("Starting new batch (Iteration %d): next_shots=%d, total_shots_so_far=%d", iteration_count, next_shots, current_shots)

            # Execute the runner function with the current batch size.
            last_result = active_runner(*args, shots=next_shots, **kwargs)
            if not isinstance(last_result, dict):
                raise ValueError("Runner must return a dictionary of frequency counts.")

            # Update the cumulative result using the latest batch.
            for key, value in last_result.items():
                cumulative_result[key] = cumulative_result.get(key, 0) + value
            history.append(last_result)
            current_shots += sum(last_result.values())

            if self.verbose:
                logger.info("Iteration %d complete. Updated cumulative result. total_shots=%d", iteration_count, current_shots)

            stopping_flag, info = self.stopping_criterion(history, cumulative_result, last_result)
            # Check if the current iteration meets the stopping criterion.
            if stopping_flag:
                stability_counter += 1
                if self.verbose:
                    logger.info("Iteration %d: Stopping criterion met. Stability counter increased to %d.", iteration_count, stability_counter)
            else:
                stability_counter = 0
                if self.verbose:
                    logger.info("Iteration %d: Stopping criterion not met. Stability counter reset to 0.", iteration_count)

            # Determine if we've achieved the required consecutive stable iterations.
            required_stability = self.stability_criterion(history, cumulative_result, last_result, info)
            if self.verbose:
                logger.info("Iteration %d: Required consecutive stable iterations: %d.", iteration_count, required_stability)
            if stability_counter >= required_stability:
                stop_flag = True
                if self.verbose:
                    logger.info("Iteration %d: Stability threshold reached. Halting execution.", iteration_count)

            # Determine the number of shots for the next iteration.
            next_shots = self.next_shots_fn(history, cumulative_result, last_result, info)
            if self.verbose:
                logger.info("Iteration %d: Next shots for upcoming batch determined to be: %d.", iteration_count, next_shots)
                logger.info("--" * 40)

        # Record the total shots and iterations as attributes.
        self.shots_run = current_shots
        self.iterations = iteration_count

        if self.verbose:
            logger.info("Execution finished. Total shots executed: %d. Total iterations: %d. Final cumulative result: %s", current_shots, iteration_count, cumulative_result)
        return cumulative_result

    def __call__(self, func: RunnerType) -> Callable[..., FreqCounts]:
        """
        Enables the instance to be used as a decorator. The decorated function will be used as the runner.

        Example:

            @IncrementalExecution(stopping_criterion, stability_criterion, next_shots, default_shots, verbose=True)
            def my_runner(*, shots):
                ...

        Returns:
            A callable that, when invoked, runs the incremental execution process.
        """
        self.runner = func

        @wraps(func)
        def wrapper(*args, **kwargs) -> FreqCounts:
            return self.run(*args, **kwargs)
        return wrapper