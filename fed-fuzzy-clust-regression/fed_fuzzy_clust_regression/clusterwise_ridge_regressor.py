# python
from typing import List
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import RegressorMixin, BaseEstimator
from scipy.spatial import distance
from sklearn.metrics import r2_score


class ClusterwiseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Clusterwise Ridge Regression, assumes a fitted fuzzy cluster model is provided.
    For each cluster, a Ridge regression model is trained using sample weights
    derived from the fuzzy memberships of samples to that cluster.
    During prediction, the outputs of each cluster-specific Ridge model are combined by calculating a weighted average.
    """

    def __init__(self, cluster_centers, m: float, alpha: float = 1.0, **ridge_kwargs):

        # Parameters from a fitted fuzzy cluster model
        self.cluster_centers = cluster_centers
        self.m = m
        self.n_clusters = cluster_centers.shape[0]

        self.alpha = alpha
        self.ridge_kwargs = ridge_kwargs
        self.cluster_models: List[RegressorMixin] = []
        self.is_fitted = False

    def calculate_cluster_memberships(self, X):
        """
        Calculate membership values for each sample to each cluster.
        :param X: Array of samples
        :return: Membership matrix of shape (n_samples, n_clusters)
        """
        X = np.asarray(X)
        power = 2.0 / (self.m - 1.0)
        dist_to_centers = distance.cdist(X, self.cluster_centers)

        # Avoid division by zero
        epsilon = 1e-10
        dist_to_centers = np.maximum(dist_to_centers, epsilon)

        U_denom = np.zeros((X.shape[0], self.n_clusters))

        for j in range(self.n_clusters):
            cj_dists = np.repeat(dist_to_centers[:, j], self.n_clusters).reshape(X.shape[0], -1)
            dists_normalized = np.power(cj_dists / dist_to_centers, power)
            U_denom[:, j] = dists_normalized.sum(axis=1)

        membership_matrix = 1.0 / U_denom

        return membership_matrix

    def fit(self, X, y):
        """
        Fit one Ridge model per cluster using membership values as sample weights.

        :param X: array-like, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # get membership matrix from fitted cluster model
        memberships = np.asarray(self.calculate_cluster_memberships(X_arr))
        if memberships.ndim != 2:
            raise ValueError("Cluster model `predict` must return a 2D membership matrix")

        n_samples, n_clusters = memberships.shape
        if n_samples != X_arr.shape[0]:
            raise ValueError("Membership rows must match number of samples")

        self.cluster_models = []

        # Fit one Ridge per cluster, using membership column as sample_weight
        for k in range(n_clusters):
            weights_k = memberships[:, k].astype(float)
            # If all weights are zero, fit on tiny regularization fallback (fit on full data with tiny weights)
            if np.allclose(weights_k, 0.0):
                # fallback: fit on full data with very small weights to avoid failures
                rk = Ridge(alpha=self.alpha, **self.ridge_kwargs)
                rk.fit(X_arr, y_arr, sample_weight=weights_k)
            else:
                rk = Ridge(alpha=self.alpha, **self.ridge_kwargs)
                rk.fit(X_arr, y_arr, sample_weight=weights_k)
            self.cluster_models.append(rk)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predict using the weighted ensemble of cluster-specific Ridge models.

        :param X: Array-like of samples to score, shape (n_samples, n_features)
        :return: Array of predictions, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call `fit` before `predict`.")

        X_arr = np.asarray(X)
        memberships = np.asarray(self.calculate_cluster_memberships(X_arr))
        if memberships.ndim != 2:
            raise ValueError("Cluster model `predict` must return a 2D membership matrix")

        n_samples, n_clusters = memberships.shape
        if n_clusters != len(self.cluster_models):
            raise ValueError("Number of clusters from cluster model does not match trained cluster models")

        # collect predictions from each cluster model: shape (n_clusters, n_samples)
        preds = np.stack([m.predict(X_arr) for m in self.cluster_models], axis=1)  # (n_samples, n_clusters)

        # weighted average per sample
        weighted = preds * memberships  # membership sums to 1
        y_pred = np.sum(weighted, axis=1) # weighted average

        return y_pred

    def score(self, X, y):

        return r2_score(y, self.predict(X))
