# script to test different GA parameter values
# tests each param while keeping others at default

import csv
import time
from pathlib import Path
from test_ga_no_threads import run_ga, DEFAULT_CONFIG

# all the params we want to test
EXPERIMENTS = {
    'pop_size': [20, 40, 60, 80],
    'gene_count': [3, 5, 7, 10],
    'point_mutate_rate': [0.05, 0.1, 0.2, 0.3],
    'point_mutate_amount': [0.05, 0.1, 0.25, 0.5],
    'shrink_mutate_rate': [0.1, 0.15, 0.25, 0.35],
    'grow_mutate_rate': [0.05, 0.1, 0.15, 0.25],
    'elitism_count': [1, 2, 3, 5],
}

NUM_TRIALS = 3
RESULTS_DIR = Path("results")


def test_param(param, values):
    """run all trials for one parameter"""
    results = []
    param_dir = RESULTS_DIR / param
    param_dir.mkdir(parents=True, exist_ok=True)

    for val in values:
        for t in range(NUM_TRIALS):
            print(f"\n--- {param}={val}, trial {t+1}/{NUM_TRIALS} ---")

            cfg = {param: val}
            save_dir = param_dir / f"{param}_{val}_trial_{t}"

            res = run_ga(config=cfg, save_dir=str(save_dir))

            results.append({
                'param_name': param,
                'param_value': val,
                'trial': t,
                'best_fitness': res['best_fitness'],
                'generations_run': res['generations_run'],
                'converged_at': res['converged_at']
            })

    return results


def save_to_csv(data, filename):
    """write results to csv file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # go through each parameter
    for param, vals in EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {param}")
        print(f"Values: {vals}")
        print(f"{'='*50}")

        results = test_param(param, vals)
        all_results.extend(results)

        # save after each param in case it crashes
        save_to_csv(all_results, RESULTS_DIR / "summary_results.csv")

    print("\nAll done! Check results/summary_results.csv")
