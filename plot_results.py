"""
Plot results from parameter experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_folder = Path("results")
plots_folder = Path("plots")


def make_param_plot(data, param):
    """Line plot: best fitness (solid) and mean fitness (dotted) vs param value"""
    param_data = data[data['param_name'] == param]

    # group by param value
    grouped = param_data.groupby('param_value')['best_fitness']
    best_per_value = grouped.max()  # best across all trials
    mean_per_value = grouped.mean()  # mean across all trials

    plt.figure(figsize=(8, 5))
    plt.plot(best_per_value.index, best_per_value.values, 'o-', label='Best')
    plt.plot(mean_per_value.index, mean_per_value.values, 's--', label='Mean')

    plt.xlabel(param)
    plt.ylabel('Fitness')
    plt.title(f'{param}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_folder / f"{param}.png")
    plt.close()


def make_summary_bar(data):
    """Grouped bar chart: best and mean fitness for each parameter's best value"""
    results = []

    for param in data['param_name'].unique():
        param_data = data[data['param_name'] == param]
        # find which value gave best mean performance
        means_by_value = param_data.groupby('param_value')['best_fitness'].mean()
        best_value = means_by_value.idxmax()

        # get stats for that best value
        best_value_data = param_data[param_data['param_value'] == best_value]
        best_fit = best_value_data['best_fitness'].max()
        mean_fit = best_value_data['best_fitness'].mean()

        results.append({
            'param': param,
            'best_value': best_value,
            'best': best_fit,
            'mean': mean_fit
        })

    df = pd.DataFrame(results)

    # plot grouped bars
    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], df['best'], width, label='Best')
    plt.bar([i + width/2 for i in x], df['mean'], width, label='Mean')

    # add value labels on top
    for i, row in df.iterrows():
        plt.text(i, row['best'] + 0.05, f"{row['best_value']}", ha='center', fontsize=8)

    plt.xticks(x, df['param'], rotation=45, ha='right')
    plt.ylabel('Fitness')
    plt.title('Best Parameter Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_folder / "summary.png")
    plt.close()

    # save to csv
    df.to_csv(plots_folder / "best_params.csv", index=False)
    print("\nBest parameters:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    plots_folder.mkdir(exist_ok=True)

    print("Loading results...")
    data = pd.read_csv(results_folder / "summary_results.csv")

    print("Creating plots...")
    for param in data['param_name'].unique():
        print(f"  {param}")
        make_param_plot(data, param)

    make_summary_bar(data)

    print(f"\nDone! Plots in {plots_folder}/")
