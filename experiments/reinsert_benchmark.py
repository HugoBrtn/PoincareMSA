#!/usr/bin/env python3
"""
Benchmark de réinsertion :
- pour une liste d'indices, retire le point (features + embedding)
- ré-infère le point par plusieurs méthodes
- calcule les distances hyperboliques entre la position ré-inférée et la position originale
- stocke les résultats dans un CSV

Usage (depuis la racine du projet) :
python3 scripts/experiments/reinsert_benchmark.py --path_embedding test_add_point_2/PM5sigma=1.00gamma=1.00cosinepca=0_seed4.csv \
    --path_features /home/hugo/Bureau/PoincareMSA/test_add_point_2/features.csv --out results/reinsert_benchmark.csv

"""

from __future__ import annotations

import sys

# Add the project root to Python path
project_root = "/home/hugo/Bureau/PoincareMSA"
if project_root not in sys.path:
    sys.path.append(project_root)


import argparse
import os
import numpy as np
import pandas as pd
import torch
import time
from sklearn.metrics.pairwise import pairwise_distances
from scripts.build_poincare_map.model import PoincareEmbedding


def run_benchmark(path_embedding: str,
                  path_features: str,
                  out_csv: str,
                  indices: list[int] | None = None,
                  max_idx: int = 100,
                  distance_metric: str = 'cosine',
                  n_bary_k: int = 5,
                  infer_steps: int = 500,
                  infer_lr: float = 0.05,
                  train_steps: int = 500,
                  train_lr: float = 0.02,
                  train_k: int = 30,
                  train_lambda: float = 5.0,
                  device: str = 'cpu'):
    # Load data
    df = pd.read_csv(path_embedding)
    if 'pm1' not in df.columns or 'pm2' not in df.columns:
        raise ValueError('Le fichier d\'embeddings doit contenir pm1 et pm2')
    embs_all = df[['pm1', 'pm2']].to_numpy(dtype=float)
    N, dim = embs_all.shape

    # Load features (try to mimic notebook loading)
    feats_df = pd.read_csv(path_features, index_col=0, header=None)

    # ensure indices list
    if indices is None:
        indices = list(range(min(max_idx, N)))

    results = []

    for idx in indices:
        print(f"Processing index {idx} / {max_idx - 1}")    
        try:
            # original coordinates
            orig_emb = embs_all[idx].copy()
            # original feature row (use numpy array to avoid index-label issues)
            feats_array = feats_df.to_numpy()
            orig_feat = np.asarray(feats_array[idx]).reshape(1, -1)

            # build reduced datasets (remove index)
            mask = np.ones(N, dtype=bool)
            mask[idx] = False
            embs_rem = embs_all[mask]

            # for features, remove the row by POSITION using numpy.delete to keep shapes aligned
            feats_rem = np.delete(feats_array, idx, axis=0)

            # build model with remaining embeddings
            model = PoincareEmbedding(size=embs_rem.shape[0], dim=dim, gamma=1.0, lossfn='klSym', Qdist='laplace', cuda=False)
            with torch.no_grad():
                model.lt.weight.data = torch.from_numpy(embs_rem).float()

            # compute target from original feature -> remaining features
            d = pairwise_distances(orig_feat, feats_rem, metric=distance_metric).flatten()
            target = np.exp(- (model.gamma if hasattr(model, 'gamma') else 1.0) * d)
            if target.sum() <= 0:
                target = np.ones_like(target) / float(len(target))
            else:
                target = target / target.sum()

            # barycenter warm-start (using top-k of the target)
            kb = min(n_bary_k, len(target))
            topk = np.argsort(-target)[:kb]
            neighbor_embs = torch.tensor(embs_rem[topk], dtype=torch.float32)
            neighbor_w = torch.tensor(target[topk], dtype=torch.float32)
            neighbor_w = neighbor_w / neighbor_w.sum()

            # compute barycenter (karcher) and measure time
            try:
                t0 = time.perf_counter()
                x0 = model.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=200, tol=1e-7, alpha=1.0, device=device, method='karcher')
                t_bary_init = time.perf_counter() - t0
                x0_np = x0.detach().cpu().numpy().reshape(-1)
            except Exception as e:
                print('Barycenter (karcher) failed:', e)
                x0_np = np.full(dim, np.nan)
                t_bary_init = np.nan

            # method A: infer with barycenter init (uses model.hyperbolic_barycenter internally)
            try:
                t0 = time.perf_counter()
                new_emb_bary = model.infer_embedding_for_point(target, n_steps=infer_steps, lr=infer_lr, init='barycenter', device=device)
                t_infer_bary = time.perf_counter() - t0
            except Exception as e:
                print('infer_embedding_for_point(init=barycenter) failed:', e)
                new_emb_bary = np.full(dim, np.nan)
                t_infer_bary = np.nan

            # method B: infer with explicit init_vec (naive random placement in the Poincaré ball)
            try:
                # draw a random point inside the Poincaré ball (uniform radius distribution)
                max_norm = 1.0 - 1e-6
                vec = np.random.normal(size=dim)
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                radius = max_norm * (np.random.rand() ** (1.0 / float(dim)))
                naive_init_vec = (vec * radius).astype(float)

                t0 = time.perf_counter()
                new_emb_bary2 = model.infer_embedding_for_point(
                    target,
                    n_steps=infer_steps,
                    lr=infer_lr,
                    init='random',
                    init_vec=naive_init_vec,
                    device=device,
                )
                t_infer_initvec = time.perf_counter() - t0
            except Exception as e:
                print('infer_embedding_for_point(init_vec) failed:', e)
                new_emb_bary2 = np.full(dim, np.nan)
                t_infer_initvec = np.nan

            # method C: refine with train_single_point (local attraction)
            try:
                t0 = time.perf_counter()
                new_emb_train, losses = model.train_single_point(target, n_steps=train_steps, lr=train_lr, init='barycenter', device=device, k=train_k, lambda_local=train_lambda)
                t_train_single = time.perf_counter() - t0
            except Exception as e:
                print('train_single_point failed:', e)
                new_emb_train = np.full(dim, np.nan)
                t_train_single = np.nan

            # diagnostics: barycenter alone distance to orig (use embs_rem topk??) For this case compute distance between x0 and original point
            try:
                orig_t = torch.tensor(orig_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                def to_tensor(v):
                    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                d_infer_bary = float('nan')
                d_infer_bary2 = float('nan')
                d_train_single = float('nan')
                d_bary_init = float('nan')

                if not np.isnan(new_emb_bary).any():
                    d_infer_bary = model.dist.apply(to_tensor(new_emb_bary), orig_t).item()
                if not np.isnan(new_emb_bary2).any():
                    d_infer_bary2 = model.dist.apply(to_tensor(new_emb_bary2), orig_t).item()
                if not np.isnan(new_emb_train).any():
                    d_train_single = model.dist.apply(to_tensor(new_emb_train), orig_t).item()
                if not np.isnan(x0_np).any():
                    d_bary_init = model.dist.apply(to_tensor(x0_np), orig_t).item()

            except Exception as e:
                print('Distance computation failed:', e)
                d_infer_bary = d_infer_bary2 = d_train_single = d_bary_init = np.nan

            results.append({
                'idx': idx,
                'd_infer_bary': d_infer_bary,
                'd_infer_initvec': d_infer_bary2,
                'd_train_single': d_train_single,
                'd_bary_init': d_bary_init,
                't_bary_init': t_bary_init,
                't_infer_bary': t_infer_bary,
                't_infer_initvec': t_infer_initvec,
                't_train_single': t_train_single,
            })

        except Exception as e_outer:
            print(f'Failed for idx {idx}:', e_outer)
            results.append({
                'idx': idx,
                'd_infer_bary': np.nan,
                'd_infer_initvec': np.nan,
                'd_train_single': np.nan,
                'd_bary_init': np.nan,
            })

    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print('Saved results to', out_csv)
    return df_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_embedding', default='test_add_point_2/PM5sigma=1.00gamma=1.00cosinepca=0_seed4.csv',
                        help='Chemin vers le fichier d\'embeddings (csv)')
    parser.add_argument('--path_features', default='/home/hugo/Bureau/PoincareMSA/test_add_point_2/features.csv',
                        help='Chemin vers le fichier de features (csv)')
    parser.add_argument('--out', default='experiments/results/reinsert_benchmark.csv',
                        help='Chemin du fichier CSV de sortie')
    parser.add_argument('--max_idx', type=int, default=10, help='Nombre maximal d\'indices à tester')
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    # If run from an editor like VSCode (Run Python File) the defaults above
    # allow immediate execution without CLI args. They can still be overridden
    # when calling the script from a terminal.
    run_benchmark(args.path_embedding, args.path_features, args.out, max_idx=args.max_idx, device=args.device)
