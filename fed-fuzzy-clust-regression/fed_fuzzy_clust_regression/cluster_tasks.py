import os

import numpy as np
import pandas as pd
import wandb

from global_learner import GlobalClusterer
from local_learners import FuzzyCMeansClient
from scipy.spatial.distance import cdist


class FedClusterModelWrapper:
    def __init__(self, fitted_model):
        self.model = fitted_model

        self.predict_model = fitted_model.local_learners[0] # the local learners all have the same global model after training

    def predict(self, X):
        return self.predict_model.predict(X)

def fed_clustering_model_from_partitions(data_dir: str, num_partitions: int, num_clusters: int, m: int,
                                         max_iter_local: int, tol_local: float, max_global_iter: int,
                                         tol_global: float):
    """
    Trains a federated clustering model from the partitions of the dataset stored in the data_dir.

    :param data_dir: Directory containing the partitions of the dataset.
    :param num_partitions: Number of partitions in the dataset (= number of clients).
    :param num_clusters: Number of clusters.
    :param m: Fuzzy parameter.
    :param max_iter_local: How many iterations to run for each local learner in one round.
    :param tol_local: Convergence tolerance for local learners.
    :param max_global_iter: How many rounds to run the global learner for.
    :param tol_global: Convergence tolerance for the global learner.
    :return: A fitted global clustering model.
    """
    # Initialize local learners
    local_learners_list = []
    for client_id in range(num_partitions):
        # Load data for a client
        path_to_data = os.path.join(data_dir, f'partition_{client_id}', 'data.csv')
        X_client = pd.read_csv(path_to_data)
        # We need to drop the target and the label columns
        X_client = X_client.drop(columns=['target', 'label']).values

        # Initialize local learner
        local_learner = FuzzyCMeansClient(client_data=X_client, num_clusters=num_clusters, m=m, max_iter=max_iter_local,
                                          tol=tol_local)
        local_learners_list.append(local_learner)

    # Initialize global learner
    global_clusterer = GlobalClusterer(local_learners=local_learners_list, num_clusters=num_clusters,
                                       data_dim=X_client.shape[1], max_rounds=max_global_iter,
                                       global_center_update='kmeans', weighing_function=None, tol=tol_global)

    # Train global learner
    global_clusterer.fit()

    return global_clusterer