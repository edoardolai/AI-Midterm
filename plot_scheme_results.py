# plots for the encoding scheme experiments
# pretty much the same as plot_results.py but adapted for scheme data

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results_schemes")
PLOTS_DIR = Path("plots_schemes")


def plot_experiment(data, exp_name):
    """bar chart comparing schemes"""
    exp_data = data[data['experiment'] == exp_name]

    # group by scheme and get stats
    grouped = exp_data.groupby('scheme')['best_fitness']
    bests = grouped.max()
    means = grouped.mean()
    stds = grouped.std()

    x = range(len(bests))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], bests.values, width, label='Best')
    plt.bar([i + width/2 for i in x], means.values, width,
            yerr=stds.values, capsize=3, label='Mean (+/- std)')

    plt.xticks(x, bests.index, rotation=45, ha='right')
    plt.xlabel('Scheme')
    plt.ylabel('Fitness')
    plt.title(f'{exp_name}')
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

        for scheme in exp_data['scheme'].unique():
            scheme_data = exp_data[exp_data['scheme'] == scheme]
            results.append({
                'experiment': exp,
                'scheme': scheme,
                'best': scheme_data['best_fitness'].max(),
                'mean': scheme_data['best_fitness'].mean(),
                'std': scheme_data['best_fitness'].std(),
            })

    df = pd.DataFrame(results)
    df.to_csv(PLOTS_DIR / "scheme_summary.csv", index=False)

    print("\nResults:")
    print(df.to_string(index=False))

    # find the winner
    best_idx = df['mean'].idxmax()
    winner = df.loc[best_idx]
    print(f"\nBest scheme: {winner['scheme']} (mean: {winner['mean']:.3f})")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    data = pd.read_csv(RESULTS_DIR / "scheme_results.csv")

    print("Making plots...")
    for exp in data['experiment'].unique():
        print(f"  {exp}")
        plot_experiment(data, exp)

    print_summary(data)

    print(f"\nPlots saved to {PLOTS_DIR}/")
