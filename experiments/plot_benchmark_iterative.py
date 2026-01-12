#!/usr/bin/env python3
"""
Plot batch reinsertion results (matplotlib / seaborn style matching the
old benchmark script in experiments/archives/old_benchmark).

Reads the CSV produced by `experiments/batch_reinsert.py` and creates boxplots
for:
 - runtime per method ('t') (log axis shown as powers of 10)
 - Qlocal_full - Qlocal_new (per method)
 - Qglobal_full - Qglobal_new (per method)

Saves PNGs (matplotlib) using the same visual style as the old script.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterMathtext


def plot_batch_results(path_csv: str, out_dir: str, show: bool = False):
    df = pd.read_csv(path_csv)
    os.makedirs(out_dir, exist_ok=True)

    methods_of_interest = ['bary_init', 'infer_bary', 'infer_rand', 'train_single']

    # Time plot (include map_build timings)
    methods_time = methods_of_interest + ['map_build']
    df_time = df[df['method'].isin(methods_time)][['method', 't']].dropna(subset=['t']).copy()

    # Print runtime stats if available
    if not df_time.empty:
        print('\nRuntime summary (s) by method:')
        print(df_time.groupby('method')['t'].describe())
    else:
        print('No runtime data available (t column missing or all NaN)')

    # Print and save total map-build time across trials if present
    try:
        df_map_build = df[df['method'] == 'map_build']
        if not df_map_build.empty and 't' in df_map_build.columns:
            total_map_time = df_map_build['t'].dropna().sum()
            total_path = os.path.join(out_dir, 'total_map_build_time.txt')
            with open(total_path, 'w') as fh:
                fh.write(f"total_map_build_time_s,{float(total_map_time)}\n")
            print(f'Wrote total map-build time to {total_path}: {total_map_time:.3f}s')
    except Exception:
        pass

    # Qlocal diff: Qlocal_full - Qlocal_new
    qlocal_present = 'Qlocal_full' in df.columns and 'Qlocal_new' in df.columns
    if qlocal_present:
        df['Qlocal_diff'] = df['Qlocal_full'] - df['Qlocal_new']
        df_q_local = df[['method', 'Qlocal_diff']].dropna(subset=['Qlocal_diff']).copy()
        if not df_q_local.empty:
            print('\nQlocal diff summary by method:')
            print(df_q_local.groupby('method')['Qlocal_diff'].describe())
    else:
        print('Qlocal columns not found in CSV; skipping Qlocal diff plot')

    # Qglobal diff: Qglobal_full - Qglobal_new
    qglobal_present = 'Qglobal_full' in df.columns and 'Qglobal_new' in df.columns
    if qglobal_present:
        df['Qglobal_diff'] = df['Qglobal_full'] - df['Qglobal_new']
        df_q_global = df[['method', 'Qglobal_diff']].dropna(subset=['Qglobal_diff']).copy()
        if not df_q_global.empty:
            print('\nQglobal diff summary by method:')
            print(df_q_global.groupby('method')['Qglobal_diff'].describe())
    else:
        print('Qglobal columns not found in CSV; skipping Qglobal diff plot')

    # Use seaborn whitegrid style to match old script
    sns.set(style='whitegrid')

    # --- Runtime plot ---
    if not df_time.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Use whiskers that correspond to min/max
        # matplotlib/seaborn expect whis to be a float or percentile tuple; use (0,100)
        sns.boxplot(x='method', y='t', data=df_time, ax=ax, whis=(0, 100))
        ax.set_title('Runtime per method (s)')
        ax.set_xlabel('Method')
        ax.set_ylabel('Time (s)')
        # overlay mean marker
        try:
            methods = [t.get_text() for t in ax.get_xticklabels()]
            means = df_time.groupby('method')['t'].mean().reindex(methods).values
            ax.scatter(range(len(means)), means, color='red', marker='D', zorder=10)
        except Exception:
            pass

        # Save linear time plot
        out_time = os.path.join(out_dir, 'batch_runtime_box.png')
        fig.tight_layout()
        fig.savefig(out_time, dpi=200)
        print(f'Saved runtime (linear) boxplot to {out_time}')
        plt.close(fig)

        # Log-scale time plot with ticks as powers of 10 (10^n)
        # Ensure all times are positive
        if (df_time['t'] > 0).all():
            figl, axl = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='method', y='t', data=df_time, ax=axl, whis=(0, 100))
            axl.set_title('Runtime per method (log scale, powers of 10)')
            axl.set_xlabel('Method')
            axl.set_ylabel('Time (s)')
            # set log scale and format ticks as 10^n
            try:
                axl.set_yscale('log')
                axl.yaxis.set_major_locator(LogLocator(base=10.0))
                axl.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                # overlay means on log plot as well
                methods_l = [t.get_text() for t in axl.get_xticklabels()]
                means_l = df_time.groupby('method')['t'].mean().reindex(methods_l).values
                axl.scatter(range(len(means_l)), means_l, color='red', marker='D', zorder=10)
            except Exception:
                pass
            figl.tight_layout()
            out_time_log = os.path.join(out_dir, 'batch_runtime_box_log.png')
            figl.savefig(out_time_log, dpi=200)
            print(f'Saved runtime (log) boxplot to {out_time_log}')
            plt.close(figl)
        else:
            figl, axl = plt.subplots(figsize=(8, 5))
            axl.text(0.5, 0.5, 'Cannot plot time on log scale (non-positive values)', ha='center', va='center')
            axl.axis('off')
            figl.tight_layout()
            out_time_log = os.path.join(out_dir, 'batch_runtime_box_log.png')
            figl.savefig(out_time_log, dpi=200)
            print(f'Created placeholder time-log image at {out_time_log} (non-positive values)')
            plt.close(figl)

    # --- Qlocal diff plot ---
    if qlocal_present and not df_q_local.empty:
        figq, axq = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='Qlocal_diff', data=df_q_local, ax=axq, whis=(0, 100))
        axq.set_title('Qlocal difference: Qlocal_full - Qlocal_new')
        axq.set_xlabel('Method')
        axq.set_ylabel('Qlocal difference')
        try:
            methods_q = [t.get_text() for t in axq.get_xticklabels()]
            means_q = df_q_local.groupby('method')['Qlocal_diff'].mean().reindex(methods_q).values
            axq.scatter(range(len(means_q)), means_q, color='red', marker='D', zorder=10)
        except Exception:
            pass
        outql = os.path.join(out_dir, 'batch_Qlocal_diff_box.png')
        figq.tight_layout()
        figq.savefig(outql, dpi=200)
        print(f'Saved Qlocal-diff boxplot to {outql}')
        plt.close(figq)
    else:
        outql = os.path.join(out_dir, 'batch_Qlocal_diff_box.png')
        figql, axql = plt.subplots(figsize=(6, 4))
        axql.text(0.5, 0.5, 'No Qlocal per-method data available\\n(check columns)', ha='center', va='center')
        axql.axis('off')
        figql.tight_layout()
        figql.savefig(outql, dpi=200)
        print(f'Created placeholder Qlocal-diff image at {outql} (no data)')
        plt.close(figql)

    # --- Qglobal diff plot ---
    if qglobal_present and not df_q_global.empty:
        figg, axg = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='method', y='Qglobal_diff', data=df_q_global, ax=axg, whis=(0, 100))
        axg.set_title('Qglobal difference: Qglobal_full - Qglobal_new')
        axg.set_xlabel('Method')
        axg.set_ylabel('Qglobal difference')
        try:
            methods_g = [t.get_text() for t in axg.get_xticklabels()]
            means_g = df_q_global.groupby('method')['Qglobal_diff'].mean().reindex(methods_g).values
            axg.scatter(range(len(means_g)), means_g, color='red', marker='D', zorder=10)
        except Exception:
            pass
        outqg = os.path.join(out_dir, 'batch_Qglobal_diff_box.png')
        figg.tight_layout()
        figg.savefig(outqg, dpi=200)
        print(f'Saved Qglobal-diff boxplot to {outqg}')
        plt.close(figg)
    else:
        outqg = os.path.join(out_dir, 'batch_Qglobal_diff_box.png')
        figqg, axqg = plt.subplots(figsize=(6, 4))
        axqg.text(0.5, 0.5, 'No Qglobal per-method data available\\n(check columns)', ha='center', va='center')
        axqg.axis('off')
        figqg.tight_layout()
        figqg.savefig(outqg, dpi=200)
        print(f'Created placeholder Qglobal-diff image at {outqg} (no data)')
        plt.close(figqg)

    if show:
        # Show the runtime linear plot (if present)
        try:
            if not df_time.empty:
                img = plt.imread(os.path.join(out_dir, 'batch_runtime_box.png'))
                plt.figure(figsize=(8, 5))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='path_csv', default='experiments/batch_reinsert_out/batch_reinsert_results.csv')
    parser.add_argument('--out_dir', default='experiments/batch_reinsert_out')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    plot_batch_results(args.path_csv, args.out_dir, show=args.show)
