# plots for the sensory input experiments
# bar chart comparing sensor on vs sensor off

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results_sensory")
PLOTS_DIR = Path("plots_sensory")


def plot_experiment(data, exp_name):
    """bar chart comparing sensor configs"""
    exp_data = data[data['experiment'] == exp_name]

    # group by config and get stats
    grouped = exp_data.groupby('config')['best_fitness']
    bests = grouped.max()
    means = grouped.mean()
    stds = grouped.std()

    x = range(len(bests))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar([i - width/2 for i in x], bests.values, width, label='Best')
    plt.bar([i + width/2 for i in x], means.values, width,
            yerr=stds.values, capsize=3, label='Mean (+/- std)')

    plt.xticks(x, bests.index)
    plt.xlabel('Configuration')
    plt.ylabel('Fitness')
    plt.title(f'Sensory Input: {exp_name}')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{exp_name}.png")
    plt.close()


def print_summary(data):
    """print a nice summary table"""
    results = []

    for exp in data['experiment'].unique():
        exp_data = data[data['experiment'] == exp]

        for config in exp_data['config'].unique():
            cfg_data = exp_data[exp_data['config'] == config]
            results.append({
                'experiment': exp,
                'config': config,
                'best': cfg_data['best_fitness'].max(),
                'mean': cfg_data['best_fitness'].mean(),
                'std': cfg_data['best_fitness'].std(),
            })

    df = pd.DataFrame(results)
    df.to_csv(PLOTS_DIR / "sensory_summary.csv", index=False)

    print("\nResults:")
    print(df.to_string(index=False))

    # did sensors help?
    sensor_on = df[df['config'] == 'with_sensor']['mean'].values
    sensor_off = df[df['config'] == 'no_sensor']['mean'].values
    if len(sensor_on) > 0 and len(sensor_off) > 0:
        if sensor_on[0] > sensor_off[0]:
            print(f"\nSensors helped! (+{sensor_on[0] - sensor_off[0]:.3f} mean fitness)")
        else:
            print(f"\nSensors didn't help much ({sensor_on[0] - sensor_off[0]:.3f} difference)")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    data = pd.read_csv(RESULTS_DIR / "sensory_results.csv")

    print("Making plots...")
    for exp in data['experiment'].unique():
        print(f"  {exp}")
        plot_experiment(data, exp)

    print_summary(data)

    print(f"\nPlots saved to {PLOTS_DIR}/")
