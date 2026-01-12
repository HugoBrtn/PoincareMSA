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
    print('Hyperbolic distances summary:')
    print(stats)

    # Qlocal / Qglobal differences statistics (Q_orig - Q_method) for each method
    method_map = [
        ('baryinit', 'Qlocal_new_baryinit', 'Qglobal_new_baryinit', 'barycenter_init'),
        ('infer_bary', 'Qlocal_new_infer_bary', 'Qglobal_new_infer_bary', 'infer_bary'),
        ('infer_initvec', 'Qlocal_new_infer_initvec', 'Qglobal_new_infer_initvec', 'infer_initvec'),
        ('train_single', 'Qlocal_new_train_single', 'Qglobal_new_train_single', 'train_single'),
    ]

    # Prepare Qlocal diffs table
    qlocal_rows = []
    for key, qlocal_col, qglobal_col, label in method_map:
        if 'Qlocal_orig' in df.columns and qlocal_col in df.columns:
            diff = (df['Qlocal_orig'] - df[qlocal_col]).dropna()
            if not diff.empty:
                descr = diff.describe()
                qlocal_rows.append({'method': label, 'count': int(descr['count']), 'mean': float(descr['mean']), 'std': float(descr['std']), 'min': float(descr['min']), '25%': float(descr['25%']), '50%': float(descr['50%']), '75%': float(descr['75%']), 'max': float(descr['max'])})

    if len(qlocal_rows) > 0:
        df_ql_stats = pd.DataFrame(qlocal_rows).set_index('method')
        print('\nQlocal differences (Qlocal_orig - Qlocal_method) summary:')
        print(df_ql_stats)

    # Prepare Qglobal diffs table
    qglobal_rows = []
    for key, qlocal_col, qglobal_col, label in method_map:
        if 'Qglobal_orig' in df.columns and qglobal_col in df.columns:
            diff = (df['Qglobal_orig'] - df[qglobal_col]).dropna()
            if not diff.empty:
                descr = diff.describe()
                qglobal_rows.append({'method': label, 'count': int(descr['count']), 'mean': float(descr['mean']), 'std': float(descr['std']), 'min': float(descr['min']), '25%': float(descr['25%']), '50%': float(descr['50%']), '75%': float(descr['75%']), 'max': float(descr['max'])})

    if len(qglobal_rows) > 0:
        df_qg_stats = pd.DataFrame(qglobal_rows).set_index('method')
        print('\nQglobal differences (Qglobal_orig - Qglobal_method) summary:')
        print(df_qg_stats)

    sns.set(style='whitegrid')

    # Prepare time data if present
    time_cols = [c for c in ['t_infer_bary', 't_infer_initvec', 't_train_single', 't_bary_init'] if c in df.columns]
    df_time = None
    if len(time_cols) > 0:
        df_time = df[time_cols].melt(var_name='method', value_name='time_s')
        df_time = df_time.dropna(subset=['time_s'])

    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # Save distance plot (linear)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    # Use whis='range' so whiskers correspond to min/max (same as describe())
    sns.boxplot(x='method', y='distance', data=df_m, ax=ax1, whis='range')
    ax1.set_title('Reinsertion benchmark: hyperbolic distance to original embedding (linear)')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Poincaré distance')
    # overlay mean marker (for easy comparison with describe())
    try:
        methods = [t.get_text() for t in ax1.get_xticklabels()]
        means = df_m.groupby('method')['distance'].mean().reindex(methods).values
        ax1.scatter(range(len(means)), means, color='red', marker='D', zorder=10)
    except Exception:
        pass
    fig1.tight_layout()
    out1 = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_distance_linear.png')
    fig1.savefig(out1, dpi=200)
    print(f'Saved distance linear boxplot to {out1}')
    plt.close(fig1)

    # Save distance plot (log)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='method', y='distance', data=df_m, ax=ax2, whis='range')
    ax2.set_title('Reinsertion benchmark: hyperbolic distance to original embedding (log scale)')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Poincaré distance (log)')
    try:
        ax2.set_yscale('log')
    except Exception:
        pass
    fig2.tight_layout()
    out2 = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_distance_log.png')
    fig2.savefig(out2, dpi=200)
    print(f'Saved distance log boxplot to {out2}')
    plt.close(fig2)

    # Save time plot (if present)
    if df_time is not None and not df_time.empty:
        # times (linear)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='time_s', data=df_time, ax=ax3, whis='range')
        ax3.set_title('Reinsertion benchmark: runtime per method (linear)')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Time (s)')
        try:
            methods_t = [t.get_text() for t in ax3.get_xticklabels()]
            means_t = df_time.groupby('method')['time_s'].mean().reindex(methods_t).values
            ax3.scatter(range(len(means_t)), means_t, color='red', marker='D', zorder=10)
        except Exception:
            pass
        fig3.tight_layout()
        out3 = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_times.png')
        fig3.savefig(out3, dpi=200)
        print(f'Saved time boxplot to {out3}')
        plt.close(fig3)

        # times (log)
        fig3l, ax3l = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='time_s', data=df_time, ax=ax3l, whis='range')
        ax3l.set_title('Reinsertion benchmark: runtime per method (log scale)')
        ax3l.set_xlabel('Method')
        ax3l.set_ylabel('Time (s)')
        try:
            ax3l.set_yscale('log')
            out3l = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_times_log.png')
            fig3l.tight_layout()
            fig3l.savefig(out3l, dpi=200)
            print(f'Saved time (log) boxplot to {out3l}')
        except Exception:
            out3l = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_times_log.png')
            fig3l.text(0.5, 0.5, 'Cannot plot time on log scale (non-positive values)', ha='center', va='center')
            fig3l.axis('off')
            fig3l.tight_layout()
            fig3l.savefig(out3l, dpi=200)
            print(f'Created placeholder time-log image at {out3l} (non-positive values)')
        plt.close(fig3l)

    # Qlocal / Qglobal per-method difference plots: Q_orig - Q_method for each method
    method_map = [
        ('baryinit', 'Qlocal_new_baryinit', 'Qglobal_new_baryinit', 'barycenter_init'),
        ('infer_bary', 'Qlocal_new_infer_bary', 'Qglobal_new_infer_bary', 'infer_bary'),
        ('infer_initvec', 'Qlocal_new_infer_initvec', 'Qglobal_new_infer_initvec', 'infer_initvec'),
        ('train_single', 'Qlocal_new_train_single', 'Qglobal_new_train_single', 'train_single'),
    ]

    # Qlocal differences
    qlocal_rows = []
    for key, qlocal_col, qglobal_col, label in method_map:
        if 'Qlocal_orig' in df.columns and qlocal_col in df.columns:
            # compute diff = Qlocal_orig - Qlocal_method
            series_orig = df['Qlocal_orig']
            series_m = df[qlocal_col]
            diff = series_orig - series_m
            # keep non-NaN diffs
            diff = diff.dropna()
            for v in diff.values:
                qlocal_rows.append({'method': label, 'Qlocal_diff': v})

    if len(qlocal_rows) > 0:
        df_ql = pd.DataFrame(qlocal_rows)
        figql, axql = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='Qlocal_diff', data=df_ql, ax=axql, whis='range')
        axql.set_title('Qlocal difference: Qlocal_orig - Qlocal_method (per method)')
        axql.set_xlabel('Method')
        axql.set_ylabel('Qlocal difference')
        try:
            methods_q = [t.get_text() for t in axql.get_xticklabels()]
            means_q = df_ql.groupby('method')['Qlocal_diff'].mean().reindex(methods_q).values
            axql.scatter(range(len(means_q)), means_q, color='red', marker='D', zorder=10)
        except Exception:
            pass
        figql.tight_layout()
        outql = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_Qlocal_diff_by_method.png')
        figql.savefig(outql, dpi=200)
        print(f'Saved Qlocal difference boxplot to {outql}')
        plt.close(figql)
        # (log-scale Qlocal plot omitted because diffs may be non-positive)
    else:
        figql, axql = plt.subplots(figsize=(6, 4))
        axql.text(0.5, 0.5, 'No Qlocal per-method data available\n(check columns)', ha='center', va='center')
        axql.axis('off')
        outql = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_Qlocal_diff_by_method.png')
        figql.tight_layout()
        figql.savefig(outql, dpi=200)
        print(f'Created placeholder Qlocal-diff image at {outql} (no data)')
        plt.close(figql)

    # Qglobal differences
    qglobal_rows = []
    for key, qlocal_col, qglobal_col, label in method_map:
        if 'Qglobal_orig' in df.columns and qglobal_col in df.columns:
            series_orig = df['Qglobal_orig']
            series_m = df[qglobal_col]
            diff = series_orig - series_m
            diff = diff.dropna()
            for v in diff.values:
                qglobal_rows.append({'method': label, 'Qglobal_diff': v})

    if len(qglobal_rows) > 0:
        df_qg = pd.DataFrame(qglobal_rows)
        figqg, axqg = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='Qglobal_diff', data=df_qg, ax=axqg, whis='range')
        axqg.set_title('Qglobal difference: Qglobal_orig - Qglobal_method (per method)')
        axqg.set_xlabel('Method')
        axqg.set_ylabel('Qglobal difference')
        try:
            methods_qg = [t.get_text() for t in axqg.get_xticklabels()]
            means_qg = df_qg.groupby('method')['Qglobal_diff'].mean().reindex(methods_qg).values
            axqg.scatter(range(len(means_qg)), means_qg, color='red', marker='D', zorder=10)
        except Exception:
            pass
        figqg.tight_layout()
        outqg = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_Qglobal_diff_by_method.png')
        figqg.savefig(outqg, dpi=200)
        print(f'Saved Qglobal difference boxplot to {outqg}')
        plt.close(figqg)
        # (log-scale Qglobal plot omitted because diffs may be non-positive)
    else:
        figqg, axqg = plt.subplots(figsize=(6, 4))
        axqg.text(0.5, 0.5, 'No Qglobal per-method data available\n(check columns)', ha='center', va='center')
        axqg.axis('off')
        outqg = os.path.join(out_dir, os.path.splitext(os.path.basename(out_path))[0] + '_Qglobal_diff_by_method.png')
        figqg.tight_layout()
        figqg.savefig(outqg, dpi=200)
        print(f'Created placeholder Qglobal-diff image at {outqg} (no data)')
        plt.close(figqg)

    if show:
        # If interactive display requested, show the distance linear plot
        fig_show, ax_show = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='distance', data=df_m, ax=ax_show)
        ax_show.set_title('Reinsertion benchmark: hyperbolic distance to original embedding (linear)')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='path_csv', default='experiments/results/results_globins/reinsert_benchmark.csv')
    parser.add_argument('--out', dest='out_path', default='experiments/results/results_globins/reinsert_benchmark_boxplots.png')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively')
    args = parser.parse_args()

    plot_results(args.path_csv, args.out_path, show=args.show)
