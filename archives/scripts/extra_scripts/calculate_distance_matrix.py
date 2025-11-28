"""
# Script to calculate a distance matrix for given features using a specified metric.
# Utilizes scipy.spatial.distance.pdist for distance calculations and
# scipy.spatial.distance.squareform to convert to a square matrix format.
# The resulting distance matrix will be used as input to sklearn.neighbors.kneighbors_graph
# to compute the k-nearest neighbors (KNN) matrix.
"""


import argparse
import importlib
import os
import inspect
import numpy as np
from scipy.spatial.distance import pdist, squareform
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
# scipy.spatial.distance.pdist(X, metric='euclidean', *, out=None, **kwargs)
# metric : str or function, optional
# metric (str) = ['Bray-Curtis', 'Canberra', 'Chebyshev', 'City Block', 'Correlation', 'Cosine', 'Euclidean', 'Jensen-Shannon', 'Mahalanobis', 'Minkowski', 'Standardized Euclidean', 'Squared Euclidean']


parser = argparse.ArgumentParser(description='Calculate a distance matrix from a CSV file containing the features.')
parser.add_argument('--features', type=str, required=True, help='Path to the CSV file containing the features.')
parser.add_argument('--metric', type=str, default='cosine', help='The distance metric to use. See scipy.spatial.distance.pdist for valid metrics.')
parser.add_argument('--metric_module', type=str, default=None, help='The module containing the personalized distance metric to use.')
parser.add_argument('--output_path', type=str, help='Path to save the distance matrix.')
args = parser.parse_args()


def calculate_distance_matrix(features, output_path, metric='cosine', metric_module=None):
    """
    Calculate a distance matrix for given features using a specified metric.

    :param features: NumPy array of features where each row is a feature vector.
    :param metric: The distance metric to use (see scipy.spatial.distance.pdist for valid metrics). Or a personalized function.
    :return: A squareform distance matrix.
    """

    # Read the features CSV file as a NumPy array
    features = np.loadtxt(features, delimiter=',')
    print('Features loaded')

    # If a personalized distance metric is provided, import the module and get the function
    if args.metric_module is not None:
        # Import the module containing the personalized distance metric
        metric_module = importlib.import_module(args.metric_module.replace('.py', ''))
        # Get the personalized distance metric function
        metric = inspect.getmembers(metric_module, inspect.isfunction)[0][1]
    # Otherwise, use the metric provided by the user
    else:
        metric = args.metric

    # Calculate the pairwise distances between the rows of a matrix (i.e. features)
    # Retuns a ndarray (condensed distance matrix)
    distances = pdist(features, metric=metric)

    # Convert the condensed distance matrix to a squareform distance matrix
    distance_matrix = squareform(distances)
    print('Distance matrix calculated')
    print(distance_matrix)

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the distance matrix as CSV file, Numpy array
    distance_matrix_path = os.path.join(output_path, 'distance_matrix.csv')
    np.savetxt(distance_matrix_path, distance_matrix, delimiter=",")
    print(f"Distance matrix CSV file saved to {distance_matrix_path}")


if __name__ == '__main__':
    calculate_distance_matrix(args.features, args.output_path, args.metric, args.metric_module)
