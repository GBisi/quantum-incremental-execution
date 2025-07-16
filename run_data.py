import logging
import os
import json
from typing import Callable, Dict, List, Any
from itertools import product
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pandas as pd
import csv
from tqdm.auto import tqdm
from typing import Optional, Tuple, Callable, Dict, Any, List, Union

# Import criteria functions (adjust module path as needed)
from src.inc_exc.stopping_criteria import (
    total_variation_distance,
    delta_distance_criterion,
    quorum_criterion,
    delta_distance_moving_average_criterion,
    constant_stability_criterion,
    constant_next_shots,
)
from src.inc_exc.incremental_execution import IncrementalExecution

FOLDER = "results"
print(f"Running with folder: {FOLDER}")

# ------------------------------
# Type Aliases and Logging Setup
# ------------------------------
FreqCounts = Dict[str, int]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -----------------------------
# Exception Handler
# -----------------------------
def exception_handler(exception: Exception, params: Dict[str, Any]):
    logger.error(f"Exception occurred for params {params}: {exception}")

# -----------------------------
# SingleFileJsonRunner Class Definition
# -----------------------------
class SingleFileJsonRunner:
    def __init__(self, samples: List[FreqCounts], verbose: bool = False):
        self.results = samples
        self.index = 0
        self.verbose = verbose
        if not self.results:
            if verbose:
                logger.warning("No data samples found in file. Using empty dictionary as fallback.")
            self.results.append({})

    def __call__(self, shots: int) -> FreqCounts:
        aggregated: FreqCounts = {}
        accumulated_shots = 0
        while accumulated_shots < shots:
            if self.index >= len(self.results):
                if self.verbose:
                    logger.warning("Reached end of samples in file. Resetting index to 0.")
                self.index = 0
            sample = self.results[self.index]
            self.index += 1
            for key, count in sample.items():
                aggregated[key] = aggregated.get(key, 0) + count
            accumulated_shots += sum(sample.values())
        return aggregated

# ----------------------------------------------------------
# Score Function Both Baselines (All-In-One and A Posteriori)
# ----------------------------------------------------------
def score_func(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes incremental and baseline evaluations for each JSON in folder.
    A posteriori baseline stops at first batch where TVD(cumulative, full) <= threshold.
    """
    # Extract base parameters
    default_shots = params["default_shots"]
    next_shots_type = params["next_shots_type"]
    constant_next_shots_val = params.get("constant_next_shots_val")
    threshold = params["threshold"]
    delta_threshold = params["delta_threshold"]
    stopping_criterion_type = params["stopping_criterion_type"]
    stopping_dist_func = params["stopping_dist_func"]
    stability_type = params["stability_type"]
    stability_k = params.get("stability_k")
    folder = params["folder"]
    verbose = params.get("verbose", False)

    # Select distance function
    if stopping_dist_func == "tvd":
        dist_func = total_variation_distance
    else:
        raise ValueError("Invalid stopping_dist_func provided.")

    # Get next_shots function (support both constant and dynamic)
    if next_shots_type == "constant":
        if constant_next_shots_val is None:
            raise ValueError("constant_next_shots_val must be provided for next_shots_type 'constant'")
        next_shots_fn = constant_next_shots(default_shots=constant_next_shots_val)
    elif next_shots_type == "dynamic":
        # Import dynamic_shot_allocation from stopping_criteria
        from src.inc_exc.stopping_criteria import dynamic_shot_allocation
        # Allow grid search to set min_initial_iterations (default 1) and optionally pass dist_func
        min_initial_iterations = params.get("min_initial_iterations", 1)
        next_shots_fn = dynamic_shot_allocation(
            min_initial_iterations=min_initial_iterations,
            dist_func=dist_func
        )
    else:
        raise ValueError("Invalid next_shots_type provided.")

    # Build stopping criterion
    if stopping_criterion_type == "delta":
        dc_offset = params.get("dc_offset")
        stopping_criterion = delta_distance_criterion(threshold=threshold, offset=dc_offset, dist_func=dist_func)
    elif stopping_criterion_type == "quorum":
        qc_offset = params.get("qc_offset")
        qc_window = params.get("qc_window")
        qc_quorum_ratio = params.get("qc_quorum_ratio")
        stopping_criterion = quorum_criterion(
            threshold=threshold, offset=qc_offset, window=qc_window,
            quorum_ratio=qc_quorum_ratio, dist_func=dist_func
        )
    elif stopping_criterion_type == "dma":
        dma_offset = params.get("dma_offset")
        dma_window = params.get("dma_window")
        dma_alpha = params.get("dma_alpha")
        stopping_criterion = delta_distance_moving_average_criterion(
            threshold=threshold, offset=dma_offset,
            alpha=dma_alpha, window=dma_window, dist_func=dist_func
        )
    else:
        raise ValueError("Invalid stopping_criterion_type provided.")

    # Build stability criterion
    if stability_type == "constant":
        if stability_k is None:
            raise ValueError("stability_k must be provided for stability_type 'constant'")
        stability_criterion = constant_stability_criterion(k=stability_k)
    else:
        raise ValueError("Invalid stability_type provided.")

    records = []
    if not os.path.isdir(folder):
        raise ValueError(f"Folder {folder} does not exist.")

    # -------------------------------------------------------------------
    # Helper functions for the a posteriori baseline calculation
    # -------------------------------------------------------------------
    def normalize_counts(counter: Dict[str, int]) -> Dict[str, float]:
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()} if total > 0 else {}

    def estimate_ideal_shots_aposteriori(batch_freqs: List[Dict[str, int]], ap_threshold: float, ap_alpha: float) -> int:
        """
        Estimate ideal shots by stopping at first batch where TVD(cumulative, full) <= ap_threshold.
        """
        from collections import Counter

        if not batch_freqs:
            return 0

        # Compute full distribution
        full_counts = Counter()
        for batch in batch_freqs:
            full_counts.update(batch)
        final_dist = normalize_counts(full_counts)

        # Shots per batch
        S = sum(batch_freqs[0].values())

        # Iterate batches and stop at threshold
        cumulative_counts = Counter()
        for i, batch in enumerate(batch_freqs):
            cumulative_counts.update(batch)
            curr_dist = normalize_counts(cumulative_counts)
            tv = dist_func(curr_dist, final_dist)
            if tv <= ap_threshold:
                return (i + 1) * S
        # If never below threshold, use all shots
        return len(batch_freqs) * S

    # -------------------------------------------------------------------
    # Process each JSON file
    # -------------------------------------------------------------------
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(folder, filename)
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
        except Exception as e:
            logger.error("Error reading file %s: %s", filename, e)
            continue

        if "data" not in content or "metadata" not in content or "ground_truth" not in content["metadata"]:
            logger.warning("File %s missing required keys. Skipping.", filename)
            continue

        # Incremental execution
        runner_incr = SingleFileJsonRunner(content["data"], verbose=verbose)
        incremental_executor = IncrementalExecution(
            stopping_criterion=stopping_criterion,
            stability_criterion=stability_criterion,
            next_shots=next_shots_fn,
            default_shots=default_shots,
            max_shots=content["metadata"]["shots"]["total_shots"],
            runner=runner_incr,
            verbose=verbose,
        )
        incremental_result = incremental_executor.run()
        d_ground = dist_func(incremental_result, content["metadata"]["ground_truth"])

        # FULL baseline
        runner_full = SingleFileJsonRunner(content["data"], verbose=verbose)
        total_shots = content["metadata"]["shots"]["total_shots"]
        full_result = runner_full(shots=total_shots)
        d_full = dist_func(incremental_result, full_result)
        d_full_ground = dist_func(content["metadata"]["ground_truth"], full_result)

        # A posteriori baseline
        data_batches = content["data"]
        ideal_shots = estimate_ideal_shots_aposteriori(data_batches, delta_threshold, params.get("aposteriori_alpha", 0.3))
        runner_aposteriori = SingleFileJsonRunner(data_batches, verbose=verbose)
        aposteriori_result = runner_aposteriori(ideal_shots)
        d_aposteriori = dist_func(incremental_result, aposteriori_result)
        d_aposteriori_ground = dist_func(content["metadata"]["ground_truth"], aposteriori_result)

        shots_incremental = getattr(incremental_executor, "shots_run", 0)
        iterations_incremental = getattr(incremental_executor, "iterations", 0)
        metadata = content["metadata"]

        record = {
            # Metadata
            "file": filename,
            "circuit": metadata.get("circuit"),
            "circuit_size": metadata.get("circuit_size"),
            "seed": metadata.get("seed"),
            "backend": metadata.get("backends")["local"][0],
            # Incremental execution data
            "incremental_distance_ground": d_ground,
            "shots_incremental": shots_incremental,
            "iterations_incremental": iterations_incremental,
            # Full baseline data
            "incremental_distance_full": d_full,
            "full_distance_ground": d_full_ground,
            "shots_full": total_shots,
            # A posteriori baseline data
            "incremental_distance_aposteriori": d_aposteriori,
            "aposteriori_distance_ground": d_aposteriori_ground,
            "aposteriori_distance_full": dist_func(full_result, aposteriori_result),
            "shots_aposteriori": ideal_shots,
        }
        record.update({k: params[k] for k in params if k not in ["folder", "verbose"]})
        records.append(record)
        if verbose:
            logger.info(
                "Processed %s: d_ground=%.4f, full_d_ground=%.4f, aposteriori_d_ground=%.4f, shots_inc=%d, full_shots=%d, ideal_shots=%d",
                filename, d_ground, d_full_ground, d_aposteriori_ground, shots_incremental, total_shots, ideal_shots
            )

    if not records:
        raise ValueError(f"No valid JSON files found in folder '{folder}'.")
    return records

# ----------------------------------------------------------
# GridSearch Class Definition (No Aggregation)
# ----------------------------------------------------------
class GridSearch:
    def __init__(
        self,
        score_func: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        param_grid: Dict[str, Any],
        parallel_mode: str = "none",
        max_workers: Optional[int] = None,
        verbose: bool = True,
        exception_handler: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
    ):
        self.score_func = score_func
        self.param_grid = param_grid
        self.parallel_mode = parallel_mode
        self.max_workers = max_workers
        self.verbose = verbose
        self.exception_handler = exception_handler

        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    def _log(self, message: str):
        if self.verbose:
            logger.info(message)

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        # Generalized: support any <prefix>_type with <prefix>_params
        base_grid = {k: v for k, v in self.param_grid.items() if not k.endswith('_params')}
        params_grids = {k[:-7]: v for k, v in self.param_grid.items() if k.endswith('_params')}
        base_keys = list(base_grid.keys())
        base_values = list(base_grid.values())
        full_combinations = []
        for base_comb in product(*base_values):
            base_params = dict(zip(base_keys, base_comb))
            # For each _type key, check for corresponding _params
            type_keys = [k for k in base_params if k.endswith('_type')]
            # Build all combinations of all relevant params for all _type keys
            param_options = [{}]
            for type_key in type_keys:
                prefix = type_key[:-5]
                type_val = base_params[type_key]
                if prefix in params_grids and type_val in params_grids[prefix]:
                    # Get param grid for this type
                    param_grid = params_grids[prefix][type_val]
                    param_keys = list(param_grid.keys())
                    param_values = list(param_grid.values())
                    new_param_options = []
                    for opt in param_options:
                        for comb in product(*param_values):
                            params = dict(zip(param_keys, comb))
                            merged = {**opt, **params}
                            new_param_options.append(merged)
                    param_options = new_param_options
            # If no _type params, just use base_params
            if param_options:
                for opt in param_options:
                    params = {**base_params, **opt}
                    full_combinations.append(params)
            else:
                full_combinations.append(base_params)
        return full_combinations

    def _evaluate(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.score_func(params)

    def run(self) -> List[Dict[str, Any]]:
        self._log(f"Starting grid search (mode: {self.parallel_mode})")
        combinations = self._generate_combinations()
        all_records = []
        Executor = None
        if self.parallel_mode in {"thread", "process"}:
            Executor = ThreadPoolExecutor if self.parallel_mode == "thread" else ProcessPoolExecutor

        if Executor:
            with Executor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._evaluate, comb): comb for comb in combinations}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                    try:
                        records = future.result()
                        all_records.extend(records)
                    except Exception as e:
                        comb = futures[future]
                        if self.exception_handler:
                            self.exception_handler(e, comb)
                        else:
                            self._log(f"Error in {comb}: {e}")
        else:
            for comb in tqdm(combinations, desc="Evaluating", disable=not self.verbose):
                try:
                    records = self._evaluate(comb)
                    all_records.extend(records)
                except Exception as e:
                    if self.exception_handler:
                        self.exception_handler(e, comb)
                    else:
                        self._log(f"Error in {comb}: {e}")
        self._log(f"Grid search finished. Total records: {len(all_records)}")
        return all_records

    def export_to_csv(self, records: List[Dict[str, Any]], filepath: str):
        if not records:
            self._log("No records to export.")
            return
        headers = []
        for record in records:
            for key in record.keys():
                if key not in headers:
                    headers.append(key)
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)
        self._log(f"Exported results to {filepath}")

    def export_to_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(records)

# ----------------------------------------------------------
# Main Function: Run Grid Search and Export CSV
# ----------------------------------------------------------

# param_grid = {
#     "default_shots": [50, 100, 250, 500],
#     "next_shots_type": [ "dynamic", "constant"],
#     "next_shots_params": {
#         "constant": {"constant_next_shots_val": [50, 100, 250, 500]},
#         "dynamic": {"min_initial_iterations": [0, 1, 3, 5]}
#     },
#     "threshold": [0.01, 0.05, 0.001, 0.005],
#     "stopping_criterion_type": ["delta", "quorum", "dma"],
#     "stopping_criterion_params": {
#         "delta": {"dc_offset": [1]},
#         "quorum": {"qc_offset": [1], "qc_window": [3, 5, 10], "qc_quorum_ratio": [0.25, 0.5, 0.75]},
#         "dma": {"dma_offset": [1], "dma_window": [3, 5, 10], "dma_alpha": [None, 0.5]}
#     },
#     "stopping_dist_func": ["tvd"],
#     "stability_type": ["constant"],
#     "stability_k": [1, 3, 5, 10],
#     "folder": [FOLDER],
#     "aposteriori_alpha": [0.3],
# }


def main():
    param_grid = {
        "default_shots": [50, 100, 250],
        "next_shots_type": ["constant", "dynamic"],
        "next_shots_params": {
            "constant": {"constant_next_shots_val": [50, 100, 250]},
            "dynamic": {"min_initial_iterations": [0, 1, 3, 5]}
        },
        "threshold": [0.025, 0.05, 0.1],
        "delta_threshold": [0.2, 0.3],
        "stopping_criterion_type": ["delta", "dma", "quorum"],
        "stopping_criterion_params": {
            "delta": {"dc_offset": [1,2,3]},
            "quorum": {"qc_offset": [1,2,3], "qc_window": [3, 5], "qc_quorum_ratio": [0.25, 0.5, 0.75]},
            "dma": {"dma_offset": [1,2,3], "dma_window": [3, 5], "dma_alpha": [None, 0.5]}
        },
        "stopping_dist_func": ["tvd"],
        "stability_type": ["constant"],
        "stability_k": [1, 3, 5],
        "folder": [FOLDER],
        "aposteriori_alpha": [0.3],
    }
    
    grid_search = GridSearch(
        score_func=score_func,
        param_grid=param_grid,
        parallel_mode="process",
        verbose=False,
        exception_handler=exception_handler,
    )
    all_records = grid_search.run()
    output_csv = f"{FOLDER}-grid_search_results.csv"
    grid_search.export_to_csv(all_records, output_csv)
    print(f"Results exported to {output_csv}")

if __name__ == "__main__":
    main()
