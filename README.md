# FedFCR — Federated Fuzzy Clusterwise Regression

## 1\. Introduction
This code accompanies the paper titled _A Practical Framework for Federated Fuzzy Clusterwise Regression, Preprocessing and Hyperparameter Tuning_. It implements the experiments described in the paper for both non\-IID and IID federated learning scenarios.

Key components:
- Federated fuzzy clustering and federated regression experiments
- Preprocessing utilities and reproducible experiment configs
- Hyperparameter tuning driver that runs experiments and logs everything to Weights & Biases (W&B)

## 2\. The core method (FedFCR)
FedFCR proceeds in two main stages:

1. Federated fuzzy c\-means clustering:
   - A federated fuzzy c\-means formulation is used to obtain cluster memberships across clients.
   - The formulation is robust to heterogeneous (non\-IID) data distributions.

2. Clusterwise federated Ridge Regression:
   - For each fuzzy cluster, a Ridge Regression problem is solved in a federated way.
   - Each client computes a local optimum for the cluster\-specific ridge problem.
   - The server aggregates client models using fuzzy averaging to produce final cluster models.

This combination yields a model ensemble aligned with fuzzy cluster memberships and designed for federated settings.

## 3\. How to run
Prerequisites:
- Python 3.9+ and project dependencies installed (see project `pyproject.toml`)
- Flower framework installed (used for running federated experiments)
- Logged in to Weights & Biases: run `wandb login` or set `WANDB_API_KEY` in the environment

### Run a single experiment
1. Update the run config in the `fed-fuzzy-clust-regression` subfolder (the Flower app uses a run config to set dataset and hyperparameters). Example config values you may change: dataset name, `ridge-alpha`, `num-clusters`, `fuzzy-m`, `clip-lower-q`, `clip-upper-q`.

2. From the project root (or from within `fed-fuzzy-clust-regression`) run via Flower. Example:
```bash
cd fed-fuzzy-clust-regression
flwr run . --run-config 'dataset-name="wizmir" ridge-alpha=1.0 num-clusters=5 fuzzy-m=1.5 clip-lower-q=0.01 clip-upper-q=0.99'
```

Notes:
* All experiment output (metrics, artifacts, configs) is logged to W&B. You must be logged in to W&B for logging and for fetching results programmatically.
* The Flower subdirectory contains its own README with details specific to running the federated app — consult fed-fuzzy-clust-regression/README.md for Flower-specific instructions.

### Run hyperparameter tuning
* The hyperparameter tuning driver is fed-fuzzy-clust-regression/hyperparameter_tuning.py. It launches repeated runs and uses Optuna plus W&B to track trials.
* Ensure W&B login is active and dataset files are available (see DATA_DIR in the script or set an environment variable as needed).

Example:
```bash
# ensure wandb is logged in
wandb login

# run hyperparameter tuning (edit DATA_DIR inside the script or set it appropriately)
python3 fed-fuzzy-clust-regression/hyperparameter_tuning.py
```

## 4.Results
* All results are stored on Weights & Biases under the project names fed-fuzzy-clust-regression-<dataset>.
* Inspect experiment runs and analytics on the W&B web UI after experiments conclude.

### Download / summarize results locally
* For convenience, results can be exported/downsampled using the helper script create_test_validation_file.py.
* Example:
```bash 
python3 create_test_validation_file.py
```
* Example outputs and expected format are provided in the test_results directory. Inspect that folder for sample CSV/JSON outputs produced by the helper script.

## Quick pointers
* Edit fed-fuzzy-clust-regression run configs for reproducible experiments.
* Use ```fed-fuzzy-clust-regression/hyperparameter_tuning.py``` for automated tuning and repeated evaluations.
* Login to W&B before running experiments: ```wandb login```.

