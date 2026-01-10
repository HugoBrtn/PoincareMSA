#!/usr/bin/env python3
"""Plot re-insertion benchmark results.

Reads the CSV produced by `reinsert_benchmark.py` and creates a boxplot
comparing methods (d_infer_bary, d_infer_initvec, d_train_single, d_bary_init).

Usage:
    python3 scripts/experiments/plot_reinsert_results.py --in results/reinsert_benchmark.csv --out results/reinsert_benchmark_boxplots.png

"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(path_csv: str, out_path: str, show: bool = False):
    df = pd.read_csv(path_csv)

    # Select the distance columns we expect
    cols = [c for c in ['d_infer_bary', 'd_infer_initvec', 'd_train_single', 'd_bary_init'] if c in df.columns]
    if len(cols) == 0:
        raise RuntimeError(f'No expected distance columns found in {path_csv}')

    # Melt for seaborn
    df_m = df[cols].melt(var_name='method', value_name='distance')

    # Drop NaNs
    df_m = df_m.dropna(subset=['distance'])

    # Basic stats
    stats = df_m.groupby('method')['distance'].describe()
    print('Summary statistics:')
    print(stats)

    sns.set(style='whitegrid')

    # Prepare time data if present
    time_cols = [c for c in ['t_infer_bary', 't_infer_initvec', 't_train_single', 't_bary_init'] if c in df.columns]
    df_time = None
    if len(time_cols) > 0:
        df_time = df[time_cols].melt(var_name='method', value_name='time_s')
        df_time = df_time.dropna(subset=['time_s'])

    # Create a figure with rows:
    # - distances linear (always)
    # - distances log scale (always)
    # - times (if present)
    nrows = 2 + (1 if df_time is not None else 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 4 * nrows))
    # ensure axes is a flat list
    if nrows == 1:
        axes = [axes]

    # First: distances (linear)
    ax_dist_lin = axes[0]
    sns.boxplot(x='method', y='distance', data=df_m, ax=ax_dist_lin)
    ax_dist_lin.set_title('Reinsertion benchmark: hyperbolic distance to original embedding (linear)')
    ax_dist_lin.set_xlabel('Method')
    ax_dist_lin.set_ylabel('Poincaré distance')

    # Second: distances (log scale)
    ax_dist_log = axes[1]
    sns.boxplot(x='method', y='distance', data=df_m, ax=ax_dist_log)
    ax_dist_log.set_title('Reinsertion benchmark: hyperbolic distance to original embedding (log scale)')
    ax_dist_log.set_xlabel('Method')
    ax_dist_log.set_ylabel('Poincaré distance (log)')
    try:
        ax_dist_log.set_yscale('log')
    except Exception:
        pass

    # Third: times (optional)
    if df_time is not None:
        ax_time = axes[2]
        sns.boxplot(x='method', y='time_s', data=df_time, ax=ax_time)
        ax_time.set_title('Reinsertion benchmark: runtime per method')
        ax_time.set_xlabel('Method')
        ax_time.set_ylabel('Time (s)')
        try:
            ax_time.set_yscale('log')
        except Exception:
            pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f'Saved boxplot to {out_path}')

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='path_csv', default='experiments/results/reinsert_benchmark.csv')
    parser.add_argument('--out', dest='out_path', default='experiments/results/reinsert_benchmark_boxplots.png')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively')
    args = parser.parse_args()

    plot_results(args.path_csv, args.out_path, show=args.show)
