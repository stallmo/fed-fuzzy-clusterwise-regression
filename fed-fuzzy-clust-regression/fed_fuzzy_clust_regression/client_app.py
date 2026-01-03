"""fed-fuzzy-clust-regression: A Flower / sklearn app."""
import os
import time
import traceback
import warnings

import numpy as np
import wandb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import root_mean_squared_error

from fed_fuzzy_clust_regression.task import (
    get_model,
    get_initial_model,
    get_model_params,
    load_data,
    set_initial_regression_params,
    set_model_params,
    get_working_dir
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Create LogisticRegression Model
    alpha = context.run_config["ridge-alpha"]

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    model = get_initial_model(alpha=alpha)  # this is a model without fitted cluster model
    set_model_params(model, ndarrays)  # this sets the model parameters

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    dataset_name = context.run_config["dataset-name"]
    dirichlet_alpha = context.run_config["dirichlet-alpha"]
    q_lower = context.run_config["clip-lower-q"]
    q_upper = context.run_config["clip-upper-q"]
    working_dir = get_working_dir()

    X_train, X_val, y_train, y_val = load_data(partition_id, num_partitions, data_dir_base=working_dir,
                                               dataset_name=dataset_name,
                                               dirichlet_alpha=dirichlet_alpha, q_lower=q_lower, q_upper=q_upper)

    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train the model on local data
        model.fit(X_train, y_train)

    # Compute R2 score
    y_pred = model.predict(X_train)
    r2score = model.score(X_train, y_train)
    # Compute RMSE score
    rmse = root_mean_squared_error(y_true=y_train, y_pred=y_pred)
    # Compute MAE score
    mae = np.mean(np.abs(y_train - y_pred))
    # Compute MAPE score
    # MAPE: avoid divide-by-zero
    mask = np.isfinite(y_train) & (y_train != 0)
    mape = np.nan
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_pred[mask] - y_train[mask]) / y_train[mask]))

    # Compute fuzzy cardinalities to share for fuzzy averaging on server side
    memberships = model.calculate_cluster_memberships(X_train)
    fuzzy_cardinalities = np.sum(memberships, axis=0).tolist()

    print(f"Fuzzy cardinalities of partition {partition_id}: {fuzzy_cardinalities}")

    # Construct and return reply Message
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)
    metrics = {"num-examples": len(X_train), "r2": r2score, "rmse": rmse, "mae": mae, "mape": mape,
               "fuzzy_cardinalities": fuzzy_cardinalities}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

def log_metrics_wandb(wandb_log_dict: dict, project: str, run_id: str, partition_id: int,
                      sdk_init_attempts: int = 5, api_attempts: int = 3) -> bool:
    """
    Try to log metrics reliably from a client:
    1) Prefer SDK attach (reliable time-series) with retries/backoff.
    2) Fall back to wandb.Api().run(...).summary.update(...) with retries.
    Returns True on success, False otherwise.
    """
    # 1) Try SDK attach if we have credentials
    for attempt in range(1, sdk_init_attempts + 1):
        try:
            print(f"[client {partition_id}] wandb SDK init attempt {attempt}/{sdk_init_attempts} for {project}/{run_id}")
            wandb.init(project=project, id=run_id, resume="must", reinit=True)
            wandb.log(wandb_log_dict)
            wandb.finish()
            print(f"[client {partition_id}] Logged metrics via SDK: {wandb_log_dict}")
            return True
        except Exception as exc:
            print(f"[client {partition_id}] wandb SDK init/log failed (attempt {attempt}): {exc}")
            traceback.print_exc()
            # small exponential backoff
            time.sleep(0.5 * (2 ** (attempt - 1)))
    print(f"[client {partition_id}] SDK logging failed after {sdk_init_attempts} attempts, falling back to API")

    # 2) Fallback: use the API summary update (best-effort, scalar summary)
    for attempt in range(1, api_attempts + 1):
        try:
            print(f"[client {partition_id}] wandb.Api run update attempt {attempt}/{api_attempts} for {project}/{run_id}")
            api = wandb.Api()
            run = api.run(f"{project}/{run_id}")
            run.summary.update(wandb_log_dict)
            run.update()
            print(f"[client {partition_id}] Logged metrics via Api.summary: {wandb_log_dict}")
            return True
        except Exception as exc:
            print(f"[client {partition_id}] wandb.Api update failed (attempt {attempt}): {exc}")
            traceback.print_exc()
            time.sleep(0.5 * attempt)

    print(f"[client {partition_id}] All W&B logging attempts failed. Ensure WANDB_API_KEY, network access, and upgrade wandb.")
    return False

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on test data."""
    partition_id = context.node_config["partition-id"]

    # Create Model
    ridge_alpha = context.run_config["ridge-alpha"]
    model = get_initial_model(alpha=ridge_alpha)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    num_partitions = context.node_config["num-partitions"]
    dataset_name = context.run_config["dataset-name"]
    dirichlet_alpha = context.run_config["dirichlet-alpha"]
    working_dir = get_working_dir()

    X_train, X_val, y_train, y_val = load_data(partition_id, num_partitions, data_dir_base=working_dir,
                                               dataset_name=dataset_name,
                                               dirichlet_alpha=dirichlet_alpha)  # this is client side evaluation on train data

    # Evaluate the model on local data
    y_train_pred = model.predict(X_val)

    mean_abs_error = np.abs(y_train_pred - y_val).mean()
    pct_errors = np.abs((y_train_pred - y_val) / y_val)
    finite_mask = np.isfinite(pct_errors)
    if finite_mask.sum() > 0:
        mean_perc_error = float(np.mean(pct_errors[finite_mask]))
    else:
        mean_perc_error = float("nan")

    r2score = model.score(X_val, y_val)
    rmse = root_mean_squared_error(y_true=y_val, y_pred=y_train_pred)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_val),
        "val_mae": mean_abs_error,
        "val_mape": mean_perc_error,
        "val_rmse": rmse,
        "val_r2": r2score
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
