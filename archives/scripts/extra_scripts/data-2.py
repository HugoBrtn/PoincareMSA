# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.sparse import csgraph

import pandas as pd
import numpy as np
import torch
import os

import timeit
import re
from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--path", type=str,
        required=True,
        help="Path of aatmx file"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=872,
        help="Max length",
    )
    return parser

# Construct a padded numpy matrices for a given PSSM matrix
def construct_tensor(fpath):
    ansarr = np.loadtxt(fpath).reshape(-1)
    # quit()
    # ansarr = np.zeros((maxlen, 20))
    # ansarr[:arr.shape[0], :] = arr
    return np.array(ansarr)


def prepare_data(fpath, withroot = True, fmt='.aamtx'):
    # print([x[0] for x in os.walk(fpath)])
    # subfolders = [f.path for f in os.listdir(fpath) if f.is_dir() ]   
    # fmt = '.aamtx'
    proteins = [s for s in os.listdir(fpath) if fmt in s]
    n_proteins = len(proteins)
    print(f"{n_proteins-1} proteins found in folder {fpath}.")

    if not withroot:
        proteins.remove("0.aamtx")
        n_proteins = len(proteins)
        print("No root detected")

    protein_file = proteins[0]
    #print(protein_file)
    fin = f'{fpath}/{protein_file}'    

    a = construct_tensor(fin)

    features = np.zeros([n_proteins, len(a)])
    labels = []
    print("Prepare data: tensor construction")
    for i, protein_name in enumerate(proteins):
        #print(i, protein_name)
        fin = f'{fpath}/{protein_name}'
        features[i, :] = construct_tensor(fin)
        labels.append(protein_name.split('.')[0])
    print("Prepare data: successfully terminated")
    return torch.Tensor(features), np.array(labels)



# def prepare_data(fin, with_labels=True, normalize=False, n_pca=0):
#     """
#     Reads a dataset in CSV format from the ones in datasets/
#     """
#     df = pd.read_csv(fin + '.csv', sep=',')
#     n = len(df.columns)

#     if with_labels:
#         x = np.double(df.values[:, 0:n - 1])
#         labels = df.values[:, (n - 1)]
#         labels = labels.astype(str)
#         colnames = df.columns[0:n - 1]
#     else:
#         x = np.double(df.values)
#         labels = ['unknown'] * np.size(x, 0)
#         colnames = df.columns

#     n = len(colnames)

#     idx = np.where(np.std(x, axis=0) != 0)[0]
#     x = x[:, idx]

#     if normalize:
#         s = np.std(x, axis=0)
#         s[s == 0] = 1
#         x = (x - np.mean(x, axis=0)) / s

#     if n_pca:
#         if n_pca == 1:
#             n_pca = n

#         nc = min(n_pca, n)
#         pca = PCA(n_components=nc)
#         x = pca.fit_transform(x)

#     labels = np.array([str(s) for s in labels])

#     return torch.DoubleTensor(x), labels


def connect_knn(KNN, distances, n_components, labels):
    """
    Given a KNN graph, connect nodes until we obtain a single connected
    component.
    """
    c = [list(labels).count(x) for x in np.unique(labels)]

    cur_comp = 0
    while n_components > 1:
        idx_cur = np.where(labels == cur_comp)[0]
        idx_rest = np.where(labels != cur_comp)[0]
        d = distances[idx_cur][:, idx_rest]
        ia, ja = np.where(d == np.min(d))
        i = ia
        j = ja

        KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
        KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

        nearest_comp = labels[idx_rest[j]]
        labels[labels == nearest_comp] = cur_comp
        n_components -= 1

    return KNN


def compute_rfa(features=None, distance_matrix=None, # mode='features',
    k_neighbours=15, distfn='sym', connected=False, sigma=1.0, distlocal='minkowski', output_path=None):
    """
    Computes the target RFA similarity matrix. The RFA matrix of
    similarities relates to the commute time between pairs of nodes, and it is
    built on top of the Laplacian of a single connected component k-nearest
    neighbour graph of the data.
    """
    start = timeit.default_timer()

    # # Verify that kneighbors_graph can also take a distance matrix as input
    # # and that both KNN matrices are equal:
    # if features is not None:
    #     # Calculate a distance matrix with pairwise_distances from sklearn
    #     sklearn_distance_matrix = pairwise_distances(features, metric=distlocal)
    #     # Compute the KNN matrices
    #     KNN_distance_matrix = kneighbors_graph(sklearn_distance_matrix,
    #                                            k_neighbours,
    #                                            mode='distance',
    #                                            metric='precomputed',
    #                                            include_self=False).toarray()
    #     KNN_features = kneighbors_graph(features,
    #                                     k_neighbours,
    #                                     mode='distance',
    #                                     metric=distlocal,
    #                                     include_self=False).toarray()
    #     # Verify that both KNN matrices are equal
    #     same_graph = np.array_equal(KNN_distance_matrix, KNN_features)
    #     print(f"KNN matrices are equal: {same_graph}")  # True
    #     KNN = KNN_distance_matrix if same_graph else KNN_features
    # Indeed, kneighbors_graph can take a distance matrix as input if metric='precomputed'
    # The valid distance metrics for kneighbors_graph and pairwise distances are:
    # ['cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean', 'precomputed']
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    # Compute the KNN matrix
    # Using the features or a user provided distance matrix
    if features is not None or distance_matrix is not None:
        # Use distance_matrix if provided, otherwise use the features
        data = distance_matrix if distance_matrix is not None else features
        metric = 'precomputed' if distance_matrix is not None else distlocal
        KNN = kneighbors_graph(data,
                               k_neighbours,
                               mode='distance',
                               metric=metric,
                               include_self=False).toarray()
        # Symmetrize the KNN matrix
        if 'sym' in distfn.lower():
            KNN = np.maximum(KNN, KNN.T)
        else:
            KNN = np.minimum(KNN, KNN.T)
        # Handle connected components
        n_components, labels = csgraph.connected_components(KNN)
        if connected and (n_components > 1):
            # Use the features to calculate pairwise distances if needed
            distances = pairwise_distances(features, metric=distlocal) if distance_matrix is None else data
            KNN = connect_knn(KNN, distances, n_components, labels)
            # Save distance matrix as CSV file, Numpy array
            distances_path = os.path.join(output_path, 'distance_matrix.csv')
            np.savetxt(distances_path, distances, delimiter=",")
        # Save the KNN matrix as CSV file, NumPy array
        if output_path is not None:
            KNN_output_path = os.path.join(output_path, 'KNN_matrix.csv')
            np.savetxt(KNN_output_path, KNN, delimiter=",")
            print(f"KNN matrix CSV file saved to {KNN_output_path}")

    # # If mode is not 'features' and no distance_matrix is provided, assume KNN is already computed
    # else:
    #     KNN = features

    # Compute the similarity matrix S
    if distlocal == 'minkowski':
        # sigma = np.mean(features)
        S = np.exp(-KNN / (sigma*features.size(1)))
        # sigma_std = (np.max(np.array(KNN[KNN > 0])))**2
        # print(sigma_std)
        # S = np.exp(-KNN / (2*sigma*sigma_std))
    else:
        S = np.exp(-KNN / sigma)

    # Compute the Laplacian
    S[KNN == 0] = 0
    print("Computing laplacian...")    
    L = csgraph.laplacian(S, normed=False)
    print(f"Laplacian computed in {(timeit.default_timer() - start):.2f} sec")

    # Compute the RFA matrix
    print("Computing RFA...")
    start = timeit.default_timer()
    RFA = np.linalg.inv(L + np.eye(L.shape[0]))
    RFA[RFA==np.nan] = 0.0
    print(f"RFA computed in {(timeit.default_timer() - start):.2f} sec")

    return torch.Tensor(RFA)
