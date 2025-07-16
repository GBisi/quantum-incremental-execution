import logging
import time
import pandas as pd

from src.inc_exc.grid_search import GridSearch

def sample_score(a, b):
    """
    A dummy score function that simulates work by sleeping for a short time
    and returns a score computed as the product of a and b.
    """
    time.sleep(0.1)  # Simulate computation delay
    return a * b

def exception_handler(e, params):
    """
    An example exception handler that logs exceptions.
    """
    logging.error(f"Exception for parameters {params}: {e}")

def main():
    # Configure logging to show INFO messages
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Define a grid for the parameters.
    param_grid = {
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    }
    
    # Initialize the GridSearch object.
    # Set early_stopping_score to None so that all combinations are evaluated.
    grid_search = GridSearch(
        score_func=sample_score, 
        param_grid=param_grid, 
        maximize=True, 
        parallel_mode='thread',  # Use threading; change to "process" for ProcessPoolExecutor or "none" for serial execution.
        max_workers=None, 
        verbose=True, 
        use_notebook=False,
        early_stopping_score=None,
        exception_handler=exception_handler
    )
    
    # Run the grid search and retrieve all results.
    result = grid_search.run(return_all=True)
    
    # Print the best parameters and best score.
    print("Best Parameters:", result["best_params"])
    print("Best Score:", result["best_score"])
    
    # Print all results.
    print("\nAll Results:")
    for params, score in result["all_results"]:
        print(f"Params: {params}, Score: {score}")
    
    # Export results to a DataFrame.
    df = grid_search.export_to_dataframe(result["all_results"])
    print("\nResults DataFrame:")
    print(df)
    
    # Optionally, export the results to a CSV file.
    csv_filepath = "grid_results.csv"
    grid_search.export_to_csv(result["all_results"], csv_filepath)
    print(f"\nResults exported to: {csv_filepath}")

if __name__ == '__main__':
    main()