# testing if sensory input actually helps creatures climb better
# compares creatures with sensors enabled vs disabled

import csv
from pathlib import Path
from test_ga_no_threads import run_ga
import genome

# the experiment: sensors on vs sensors off
# if sensors help, the 'with_sensor' runs should have higher fitness
SENSORY_EXPERIMENTS = {
    'sensory_input': [
        {'name': 'no_sensor', 'use_sensors': False},
        {'name': 'with_sensor', 'use_sensors': True},
    ],
}

NUM_TRIALS = 3
RESULTS_DIR = Path("results_sensory")


def test_sensory(exp_name, configs):
    """run all trials for one experiment"""
    results = []
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        name = cfg['name']

        for t in range(NUM_TRIALS):
            print(f"\n--- {exp_name}: {name}, trial {t+1}/{NUM_TRIALS} ---")

            # set up the sensor config
            genome.reset_encoding_config()
            genome.set_encoding_config({'use_sensors': cfg['use_sensors']})

            save_dir = exp_dir / f"{name}_trial_{t}"

            # run it
            res = run_ga(config={}, save_dir=str(save_dir))

            results.append({
                'experiment': exp_name,
                'config': name,
                'trial': t,
                'best_fitness': res['best_fitness'],
                'generations_run': res['generations_run'],
                'converged_at': res['converged_at'],
            })

            # reset for next run
            genome.reset_encoding_config()

    return results


def save_to_csv(data, filename):
    if not data:
        return
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for exp_name, configs in SENSORY_EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Running: {exp_name}")
        print(f"Configs: {[c['name'] for c in configs]}")
        print(f"{'='*50}")

        results = test_sensory(exp_name, configs)
        all_results.extend(results)

        # save after each experiment in case something crashes
        save_to_csv(all_results, RESULTS_DIR / "sensory_results.csv")

    print(f"\nDone! Results saved to {RESULTS_DIR}/sensory_results.csv")
