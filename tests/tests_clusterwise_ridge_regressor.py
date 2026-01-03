# python
import numpy as np
import pytest
from numpy.testing import assert_allclose

from fed_fuzzy_clust_regression.clusterwise_ridge_regressor import ClusterwiseRidgeRegressor

def _get_centers():

    return np.array([[1.0], [10.0]])

def _make_linearly_separable_data():
    """
    Create a simple 1D dataset with two clusters:
    - cluster 0 around x=0 with y = 2 * x
    - cluster 1 around x=10 with y = 3 * x
    Returns X (n,1) and y (n,)
    """
    # create 10 random values by drawing from a normal distribution centered around 0.1
    centers = _get_centers()
    X_clust1 = np.random.normal(loc=centers[0][0], scale=0.01, size=10)
    y_clust1 = 2.0 * X_clust1
    # create 10 random values by drawing from a normal distribution centered around 1.0
    X_clust2 = np.random.normal(loc=centers[1][0], scale=0.01, size=10)
    y_clust2 = 3.0 * X_clust2

    X = np.concatenate((X_clust1, X_clust2), axis=0).reshape(-1, 1)
    y = np.concatenate((y_clust1, y_clust2), axis=0).reshape(-1,)

    return X, y


def test_predict_before_fit_raises():
    centers = _get_centers()
    model = ClusterwiseRidgeRegressor(cluster_centers=centers, m=2.0, alpha=1.0)
    X_test = np.array([[0.0]])
    with pytest.raises(RuntimeError):
        model.predict(X_test)


def test_fit_creates_one_ridge_per_cluster_and_predicts_correctly():
    # Arrange
    X, y = _make_linearly_separable_data()
    centers = _get_centers()
    model = ClusterwiseRidgeRegressor(cluster_centers=centers, m=2.0, alpha=0.0)

    # Act
    model.fit(X, y)

    # Assert cluster models created
    assert len(model.cluster_models) == model.n_clusters == 2

    # Predictions: point exactly at cluster 0 center -> slope ~2
    x0 = centers[0][0].reshape(-1, 1)
    pred0 = model.predict(x0)
    # output membership matrix
    memberships = model.calculate_cluster_memberships(x0)
    assert_allclose(memberships[0, 0], 1.0, rtol=0.01)

    assert pred0.shape == (1,)
    assert_allclose(pred0[0], 2.0 * x0[0, 0], rtol=0.01)

    # Point exactly at cluster 1 center -> slope ~3
    x1 = centers[1][0].reshape(-1, 1)
    pred1 = model.predict(x1)
    assert_allclose(pred1[0], 3.0 * x1[0, 0], rtol=0.01)


def test_weighted_average_for_middleground_point():
    # Arrange: same setup as before
    X, y = _make_linearly_separable_data()
    centers = _get_centers()
    model = ClusterwiseRidgeRegressor(cluster_centers=centers, m=2.0, alpha=0.0)
    model.fit(X, y)

    # Act: point equidistant between centers should get membership ~0.5/0.5
    xm = np.mean(centers).reshape(-1, 1)
    pred = model.predict(xm)[0]

    memberships = model.calculate_cluster_memberships(xm)
    assert_allclose(memberships[0, 0], 0.5, rtol=0.05)

    expected = 0.5 * (2.0 * xm[0, 0] + 3.0 * xm[0, 0])
    assert_allclose(pred, expected, rtol=0.1)


def test_membership_matrix_shape_and_values_sum_to_one():
    # Access internal membership computation (name-mangled) to assert shape and normalization.
    centers = np.array([[0.0], [10.0]])
    model = ClusterwiseRidgeRegressor(cluster_centers=centers, m=2.0, alpha=1.0)

    X = np.array([[0.0], [5.0], [10.0]])
    # call the private method via name-mangled attribute
    U = model.calculate_cluster_memberships(X)

    # Shape: (n_samples, n_clusters)
    assert U.shape == (3, 2)

    # Memberships should be in (0,1) and rows sum to ~1
    assert np.all(U >= 0.0) and np.all(U <= 1.0)
    row_sums = U.sum(axis=1)
    assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-8)
