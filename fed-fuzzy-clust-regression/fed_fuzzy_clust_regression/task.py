"""fed-fuzzy-clust-regression: A Flower / sklearn app."""
import os
from typing import List

import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import wandb

from .fixed_data_federated_dataset import FixedDataFederatedDataset
from .data_partition_utils import create_dataset_from_dat_file
from .clusterwise_ridge_regressor import ClusterwiseRidgeRegressor
from .preprocessing import get_preprocessing_pipeline

fds = None  # Cache FederatedDataset
preprocessing_pipeline = None  # Cache preprocessing pipeline


def get_path_for_dataset_name(dataset_name: str):
    this_dir = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(this_dir, '..', '..', 'data', f"{dataset_name}.dat"))

    # Check if the path exists
    if os.path.exists(path):
        return path
    else:
        raise ValueError(f"Dataset {dataset_name} not found in data folder")


def get_federated_dataset():
    global fds
    return fds

def get_working_dir():
    this_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(this_dir, 'working_dir'))

def load_data(partition_id: int, num_partitions: int, dataset_name: str, data_dir_base: str, dirichlet_alpha: float, q_lower: float = 0.01,
              q_upper: float = 0.99, apply_preprocessing: bool = True):
    global fds
    path_to_data = get_path_for_dataset_name(dataset_name)

    if fds is None:
        print("Creating FederatedDataset")
        # Check if the path exists
        assert path_to_data is not None, f"Dataset {dataset_name} not found in data folder"

        # create the dataset from the dat file
        ds = create_dataset_from_dat_file(path_to_data, test_size=0.2, seed=43)
        # Create FederatedDataset from the dataset
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=dirichlet_alpha, partition_by="label")
        if apply_preprocessing:
            preprocessing_pipeline = get_preprocessing_pipeline(q_lower=q_lower, q_upper=q_upper,
                                                                include_non_numeric=True)
        else:
            preprocessing_pipeline = None
        fds = FixedDataFederatedDataset(ds, dataset_name, data_dir_base, partitioners_dict={"train": partitioner},
                                        preprocessing_pipeline=preprocessing_pipeline)

    dataset = fds.load_partition(partition_id, "train")  # .with_format("pandas")[:]
    feature_columns = [col for col in dataset.columns if col not in ["target", "label"]]

    X = dataset[feature_columns]
    y = dataset["target"]

    # Split into train and validation sets by randomly selecting 10% of the data
    X_train = X[:int(0.9 * len(X))]
    X_val = X[int(0.9 * len(X)):]
    y_train = y[:int(0.9 * len(y))]
    y_val = y[int(0.9 * len(y)):]

    return X_train, X_val, y_train, y_val


def get_initial_model(alpha: float, **ridge_kwargs):
    """
    Returns a ClusterwiseRidgeRegressor object with empty cluster parameters
    :param alpha: Alpha in ridge regression.
    :param ridge_kwargs: Parameters passed to Ridge constructor.
    :return: ClusterwiseRidgeRegressor object
    """
    dummy_centers = np.zeros((1, 1))
    return ClusterwiseRidgeRegressor(cluster_centers=dummy_centers, m=-1.0, alpha=alpha, **ridge_kwargs)


def get_model(cluster_centers, m, alpha: float, **ridge_kwargs):
    return ClusterwiseRidgeRegressor(cluster_centers=cluster_centers, m=m, alpha=alpha, **ridge_kwargs)


def get_model_params(model: ClusterwiseRidgeRegressor):
    """
    Returns the model parameters as a list of lists
    Each inner list contains a tuple of the model's coefficients and intercept.

    :param model: ClusterwiseRidgeRegressor object
    :return: List of lists of tuples
    """
    arrays: List[np.ndarray] = []

    # Ensure cluster_models exists
    if not hasattr(model, "cluster_models") or len(model.cluster_models) == 0:
        raise ValueError(
            "Model has no cluster_models. Ensure clusterwise model is initialized before calling get_model_params.")

    # First, append regressor params for each cluster model -> (coef, intercept)
    for cm in model.cluster_models:
        if not isinstance(cm, Ridge):
            raise ValueError("Cluster model is not a Ridge regression model.")
        # coef_
        arrays.append(np.asarray(cm.coef_, dtype=float))
        # intercept_ (always include to keep fixed ordering)
        intercept_val = getattr(cm, "intercept_", 0.0)
        arrays.append(np.asarray(intercept_val, dtype=float))

    # Append cluster metadata
    centers = getattr(model, "cluster_centers", None)
    if centers is None:
        raise ValueError("Model has no cluster_centers.")
    arrays.append(np.asarray(centers, dtype=float))

    # m (fuzzy parameter)
    arrays.append(np.asarray(getattr(model, "m", np.nan), dtype=float))

    # n_clusters
    arrays.append(np.asarray(getattr(model, "n_clusters", len(model.cluster_models)), dtype=int))

    return arrays


def set_model_params(model: ClusterwiseRidgeRegressor, params_list: List[np.ndarray]) -> ClusterwiseRidgeRegressor:
    """
    Parse the flat params_list produced by get_model_params and set attributes on `model`.

    Expected format:
      [coef_0, intercept_0, coef_1, intercept_1, ..., cluster_centers, m, n_clusters]
    """
    if len(params_list) < 3:
        raise ValueError("params_list too short to contain cluster models and regression params.")

    # Extract cluster model params
    n_clusters_arr = params_list[-1]
    m_arr = params_list[-2]
    centers_arr = params_list[-3]

    try:
        n_clusters = int(np.asarray(n_clusters_arr).item())
    except Exception as e:
        raise ValueError("Failed to parse n_clusters from params_list") from e

    m = np.asarray(m_arr).item()
    centers = np.asarray(centers_arr, dtype=float)

    # Validate shapes
    if centers.ndim != 2 or centers.shape[0] != n_clusters:
        raise ValueError(
            f"cluster_centers shape does not match n_clusters. Got n_clusters={n_clusters}, centers.shape={centers.shape}.")

    # n_features = centers.shape[1]

    # Remaining arrays are regressor params
    reg_arrays = params_list[:-3]
    expected_len = n_clusters * 2  # coef + intercept per cluster
    if len(reg_arrays) != expected_len:
        raise ValueError(f"Expected {expected_len} regressor arrays, got {len(reg_arrays)}")

    # Create cluster Ridge models and set params
    cluster_models = []
    idx = 0
    for i in range(n_clusters):
        coef = np.asarray(reg_arrays[idx], dtype=float);
        idx += 1
        intercept = np.asarray(reg_arrays[idx], dtype=float);
        idx += 1

        # Create Ridge instance preserving provided alpha if available on model
        ridge_kwargs = {}
        # Try to preserve any ridge kwargs on the existing model (if present)
        if hasattr(model, "alpha"):
            ridge_kwargs["alpha"] = getattr(model, "alpha")
        rk = Ridge(**ridge_kwargs)
        # Assign attributes expected by sklearn / serialization
        rk.coef_ = coef.reshape(-1)
        # Ensure intercept is scalar
        try:
            rk.intercept_ = float(np.asarray(intercept).item())
        except Exception:
            rk.intercept_ = 0.0
        rk.fit_intercept = True
        cluster_models.append(rk)

    # Attach to model
    model.cluster_models = cluster_models
    model.cluster_centers = centers
    model.m = float(m) if not np.isnan(m) else None
    model.n_clusters = int(n_clusters)
    model.is_fitted = True

    return model


def set_initial_regression_params(model, n_features: int):
    n_clusters = model.n_clusters  # Assumes cluster model has been initialized

    cm_list = []
    for clust_n in range(n_clusters):
        regr_model = Ridge(alpha=model.alpha)
        regr_model.coef_ = np.zeros(n_features)
        regr_model.intercept_ = 0.0
        cm_list.append(regr_model)
    model.cluster_models = cm_list

    return model


def evaluate_on_test_set(model):
    global fds

    if fds is None:
        raise RuntimeError("No FederatedDataset available. Ensure load_data has been called.")

    # Load the global test dataset (held-back 20% not partitioned)
    test_dataset = fds.load_test_split()
    feature_columns = [col for col in test_dataset.columns if col not in ["target", "label"]]
    print(f"Loaded {len(test_dataset)} test examples with features {feature_columns}.")

    X_test = test_dataset[feature_columns]
    y_true = test_dataset["target"]

    if len(X_test) == 0:
        print("No test examples found.")
        return

    y_pred = np.asarray(model.predict(X_test)).astype(float)
    y_true = np.asarray(y_true).astype(float)

    mae = np.mean(np.abs(y_pred - y_true))
    # MAPE: avoid divide-by-zero
    mask = np.isfinite(y_true) & (y_true != 0)
    mape = np.nan
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # log metrics to wandb via wandb API

    wandb.log({"test/mae": mae, "test/mape": mape, "test/rmse": rmse, "test/r2": r2})

    print(
        f"Server-side test evaluation: n={len(y_true)}, MAE={mae:.4f}, MAPE={np.nan if np.isnan(mape) else mape:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


def calculate_fuzzy_db_index(data, centers, assignment_matrix):
    n_data_points = data.shape[0]
    _num_clusters = assignment_matrix.shape[1]

    S = []
    # First, calculate the "cluster spreads" for each cluster
    for i in range(_num_clusters):
        center_i = centers[i].reshape((-1, data.shape[1]))
        dists_to_center_i = cdist(data, center_i, metric='euclidean')
        sum_dist_i = dists_to_center_i.sum()
        assignment_clust_i = assignment_matrix[:, i]

        S_i = 1.0 * sum_dist_i / n_data_points
        avg_membership = 1.0 * assignment_clust_i.sum() / n_data_points
        S_i *= avg_membership
        S.append(S_i)

    # Second, calculate how well the centers are separated
    M = cdist(centers, centers, metric='minkowski', p=2)

    # Third, calculate the cluster separation index for each pair of clusters
    R = np.zeros(shape=(_num_clusters, _num_clusters))
    for i in range(_num_clusters):
        for j in range(_num_clusters):
            if i == j:
                # R[i,j] = 0
                continue
            else:
                R[i, j] = (S[i] + S[j]) / M[i, j]

    # Finally, calculate fuzzy Davies Bouldin
    # I.e., for each cluster, identify the "worst separated" cluster and average for all clusters
    fuzzy_db = R.max(axis=1).sum() / _num_clusters
    return fuzzy_db


def calculate_clustering_metrics(cluster_model, X_test):
    memberships = cluster_model.local_learners[0].predict(
        X_test)  # all local learners have the same global model after training

    fuzzy_db = calculate_fuzzy_db_index(X_test,
                                        cluster_model.cluster_centers, memberships)

    return fuzzy_db


def evaluate_clustering_on_train_set(model, wandb_run_id=None, wandb_project_name=None):
    # We know the DB index can be analogously calculated in the federated setting (see paper)
    global fds

    # get train split
    train_data = fds.load_train_split()
    feature_columns = [col for col in train_data.columns if col not in ["target", "label"]]
    X = train_data[feature_columns]

    fuzzy_db = calculate_clustering_metrics(model, X)
    print(f"Fuzzy DB index on train set: {fuzzy_db}")
    # log metrics to wandb
    wandb.log({"train/fuzzy_db": fuzzy_db})


def evaluate_clustering_on_test_set(model, wandb_run_id=None, wandb_project_name=None):
    # We know the DB index can be analogously calculated in the federated setting (see paper)
    global fds

    # get train split
    test_data = fds.load_test_split()
    feature_columns = [col for col in test_data.columns if col not in ["target", "label"]]
    X = test_data[feature_columns]

    fuzzy_db = calculate_clustering_metrics(model, X)
    print(f"Fuzzy DB index on train set: {fuzzy_db}")

    # log metrics to wandb
    wandb.log({"test/fuzzy_db": fuzzy_db})

def evaluate_clustering(model, wandb_run_id=None, wandb_project_name=None):
    evaluate_clustering_on_train_set(model, wandb_run_id, wandb_project_name)
    evaluate_clustering_on_test_set(model, wandb_run_id, wandb_project_name)