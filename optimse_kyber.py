import pandas as pd
import multiprocessing
import concurrent.futures
from kyber_simulate import simulate_competition, load_top_c
import numpy as np
import time
# Worker function must be defined at the top level for multiprocessing
def evaluate_pair(args):
    x, h, top_c_subs, n_sims, n_seeds = args
    test_sub = {"x": x, "h": h, "label": "TEST"}
    # Run simulation
    df_res = simulate_competition([test_sub] + top_c_subs, n_sims=n_sims, n_seeds=n_seeds)
    # Extract score
    score = df_res[df_res["label"] == "TEST"].iloc[0]["avg_remaining"]
    return x, h, score

def find_optimal_xh():
    print("Loading top submissions from kyber_consensus.csv...")
    try:
        # Load the top 100 competitors from your CSV to simulate the crowd
        top_c_subs = load_top_c("kyber_consensus.csv", top_c=100)
    except FileNotFoundError:
        print("Error: Ensure 'kyber_consensus.csv' is in the same directory.")
        return

    # Automatically detect the number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel grid search...\n")

    print("--- PHASE 1: Coarse Grid Search ---")
    best_score = -float('inf')
    best_x, best_h = -1, -1

    # Prepare all tasks for the coarse search
    coarse_tasks = [
        (x, h, top_c_subs, 10, 3) 
        for x in range(200, 301, 5) 
        for h in range(15, 50, 2)
    ]

    # Execute tasks in parallel using a Process Pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks and track them
        futures = {executor.submit(evaluate_pair, task): task for task in coarse_tasks}
        
        # As each parallel task completes, evaluate the results
        for future in concurrent.futures.as_completed(futures):
            x, h, score = future.result()
            
            if score > best_score:
                best_score = score
                best_x, best_h = x, h
                print(f"New Best Found -> x: {best_x}, h: {best_h} (Avg Apples: {best_score:.2f})")

    print(f"\nCoarse Grid Best - x: {best_x}, h: {best_h}, score: {best_score:.2f}")
    
    print("\n--- PHASE 2: Fine Grid Search ---")
    fine_best_score = -float('inf')
    fine_best_x, fine_best_h = -1, -1

    # Define a tighter search space around our coarse best
    x_start = max(0, best_x - 30)
    x_end = min(300, best_x + 30)
    h_start = max(5, best_h - 10)
    h_end = min(60, best_h + 10)

    # Prepare tasks for fine search
    fine_tasks = [
        (x, h, top_c_subs, 20, 5) 
        for x in range(x_start, x_end + 1, 1) 
        for h in range(h_start, h_end + 1, 1)
    ]

    # Execute fine tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(evaluate_pair, task): task for task in fine_tasks}
        
        for future in concurrent.futures.as_completed(futures):
            x, h, score = future.result()
            
            if score > fine_best_score:
                fine_best_score = score
                fine_best_x, fine_best_h = x, h
                print(f"New Fine Best Found -> x: {fine_best_x}, h: {fine_best_h} (Avg Apples: {fine_best_score:.2f})")

    print("\n" + "="*50)
    print("  OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"The absolute optimal parameters are:")
    print(f"  x = {fine_best_x}")
    print(f"  h = {fine_best_h}")
    print(f"  Expected Avg Apples = {fine_best_score:.2f}")

if __name__ == "__main__":
    # Required wrap for Windows multiprocessing execution
    find_optimal_xh()