# testing different body shape combinations
# similar to ga_param_tests.py but for the encoding scheme stuff

import csv
from pathlib import Path
from test_ga_no_threads import run_ga
import genome

# all the shape combos we want to test
# format: base shape (the root/body) + limb shape (everything else)
SCHEME_EXPERIMENTS = {
    'body_structure': [
        {'name': 'all_cylinder', 'base_shape': 'cylinder', 'limb_shape': 'cylinder'},
        {'name': 'all_box', 'base_shape': 'box', 'limb_shape': 'box'},
        {'name': 'all_sphere', 'base_shape': 'sphere', 'limb_shape': 'sphere'},
        {'name': 'box_base_cyl_limbs', 'base_shape': 'box', 'limb_shape': 'cylinder'},
        {'name': 'sphere_base_cyl_limbs', 'base_shape': 'sphere', 'limb_shape': 'cylinder'},
        {'name': 'cyl_base_box_limbs', 'base_shape': 'cylinder', 'limb_shape': 'box'},
    ],
}

NUM_TRIALS = 3
RESULTS_DIR = Path("results_schemes")


def test_scheme(exp_name, schemes):
    """run all trials for one experiment"""
    results = []
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    for scheme in schemes:
        name = scheme['name']

        for t in range(NUM_TRIALS):
            print(f"\n--- {exp_name}: {name}, trial {t+1}/{NUM_TRIALS} ---")

            # set up the shape config
            genome.reset_encoding_config()
            cfg = {k: v for k, v in scheme.items() if k != 'name'}
            genome.set_encoding_config(cfg)

            save_dir = exp_dir / f"{name}_trial_{t}"

            # run it
            res = run_ga(config={}, save_dir=str(save_dir))

            results.append({
                'experiment': exp_name,
                'scheme': name,
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

    for exp_name, schemes in SCHEME_EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Running: {exp_name}")
        print(f"Schemes: {[s['name'] for s in schemes]}")
        print(f"{'='*50}")

        results = test_scheme(exp_name, schemes)
        all_results.extend(results)

        # save after each experiment just in case something crashes
        save_to_csv(all_results, RESULTS_DIR / "scheme_results.csv")

    print(f"\nDone! Results saved to {RESULTS_DIR}/scheme_results.csv")
