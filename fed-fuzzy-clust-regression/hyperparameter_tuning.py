import os
import subprocess
import tomllib
import gc

import wandb
import optuna

N_TRIALS = 50 # optuna trials for one study
N_REPEAT_RUNS = 10 # repeat best run this many times to get stable estimate of performance
DATASET_NAME = None # will be set (and updated) later
DATA_DIR = "/Users/morris/code/fuzzy_clust_regression/data" # path to data directory containing .dat files

ALL_RUN_IDS = []

def add_run_failed_tag(run):
    if "validation_failed_run" not in run.tags:
        run.tags = run.tags + ["validation_failed_run"]
        run.update()

def calculate_validation_rmse(run) -> float:
    """
    Calculate weighted average of validation RMSEs across clients.
    :param run: A wandb run object.
    :return: Average of validation RMSEs, weighted by number of examples per client.
    """

    # Get each client's val_mape and num-examples to compute weighted average

    # Extract validation RMSEs
    try:
        val_rmses = [run.summary.get(f'client_{c}/val_rmse') for c in range(run.config["num-clients"])]
        val_num_examples = [run.summary.get(f'client_{c}/num-examples') for c in range(run.config["num-clients"])]
    except KeyError:
        add_run_failed_tag(run)
        return float('nan') # this will signal optuna that the trial failed

    # Calculate weighted average of RMSEs
    running_rmse = 0.0
    running_sum_examples = 0.0
    failed_clients = 0
    for val_rmse, num_examples in zip(val_rmses, val_num_examples):
        if val_rmse is None or num_examples is None:
            failed_clients += 1
            continue

        running_rmse += val_rmse * num_examples
        running_sum_examples += num_examples

    # If more than half of the clients failed, consider the trial failed
    if failed_clients/len(val_rmses) > 0.5:
        # Add a tag to the run indicating failure
        add_run_failed_tag(run)
        return float('nan')

    weighted_rmse = running_rmse / running_sum_examples

    # Add the validation RMSE to wandb
    run.summary.update({"validation_rmse": weighted_rmse})

    return weighted_rmse

def trigger_run(alpha, num_clusters, fuzzy_m, clip_lower_q, clip_upper_q):
    cmd = ["flwr", "run", ".", "--run-config",
           f'dataset-name="{DATASET_NAME}" ridge-alpha={alpha} num-clusters={num_clusters} fuzzy-m={fuzzy_m} clip-lower-q={clip_lower_q} clip-upper-q={clip_upper_q}']

    # Create a cleaned environment for the child process to avoid inheriting WANDB_* variables
    child_env = os.environ.copy()
    for key in list(child_env.keys()):
        if key.startswith("WANDB_"):
            child_env.pop(key, None)

    subprocess.run(cmd, check=True)

    # get all runs in the project
    # we need to create a new API instance since it is lazy and does not refresh automatically
    runs = wandb.Api().runs(wandb_project_name, filters={"state": "finished"}, order="-created_at")
    latest_run = runs[0]
    ALL_RUN_IDS.append(latest_run.id)
    print(f"Latest run: {latest_run.id}")
    print(f"Latest run name: {latest_run.name}")

    # Return the latest run in the project
    return latest_run

def objective(trial: optuna.Trial) -> float:
    """Objective function to be optimized."""
    alpha = trial.suggest_float("ridge_alpha", 0.0, 10.0)
    num_clusters = trial.suggest_int("num_clusters", 1, 30)
    fuzzy_m = trial.suggest_float("fuzzy_m", 1.01, 4.0)
    clip_lower_q = trial.suggest_float("clip_lower_q", 0.0, 0.1, step=0.01)
    clip_upper_q = trial.suggest_float("clip_upper_q", 0.9, 1.0, step=0.01)

    run = trigger_run(alpha, num_clusters, fuzzy_m, clip_lower_q, clip_upper_q)
    return calculate_validation_rmse(run)

if __name__ == "__main__":
    # Base config (load pyproject.toml file)
    with open("pyproject.toml", "rb") as f:
        base_config = tomllib.load(f)

    wandb_api = wandb.Api()

    # Find all .dat files in DATA_DIR
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
    dataset_names = [os.path.splitext(f)[0] for f in data_files]
    print(f"Found {len(data_files)} data files in {DATA_DIR}")

    for dataset_name in dataset_names:
        print(f"Starting hyperparameter tuning for {dataset_name}")
        # set global dataset name
        DATASET_NAME = dataset_name

        # derive W&B project name from dataset name (will be used by optuna study)
        #dataset_name = base_config["tool"]["flwr"]["app"]["config"]["dataset-name"]
        wandb_project_name = f"fed-fuzzy-clust-regression-{dataset_name}"

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, )

        # Print best hyperparameters
        print(f"Best hyperparameters found: {study.best_params}")

        # Get best trial number
        best_trial = study.best_trial
        best_trial_run_id = ALL_RUN_IDS[best_trial.number]
        print(f"Best trial run ID: {best_trial_run_id}")

        # Add a tag to the best run in the project
        best_run = wandb_api.run(f"{wandb_project_name}/{best_trial_run_id}")
        best_run.tags = best_run.tags + ["validation_best_run"]
        best_run.update()

        # Repeat the best run ten times to get more stable estimate of performance
        for repeat_idx in range(N_REPEAT_RUNS):
            print(f"Repeating best run, iteration {repeat_idx + 1}/{N_REPEAT_RUNS}")
            print(f"Best hyperparameters: {best_trial.params}")

            run = trigger_run(alpha=best_trial.params["ridge_alpha"],
                              num_clusters=best_trial.params["num_clusters"],
                              fuzzy_m=best_trial.params["fuzzy_m"],
                              clip_lower_q=best_trial.params["clip_lower_q"],
                              clip_upper_q=best_trial.params["clip_upper_q"])

            val_rmse = calculate_validation_rmse(run)
            print(f"Repeat {repeat_idx + 1}/{N_REPEAT_RUNS}, Validation RMSE: {val_rmse}")
            # Tag the repeat runs
            run.tags = run.tags + ["validation_best_run_repeat"]
            run.update()

            # Clean up memory by collecting garbage
            gc.collect()

        # Make sure to reset run ids for next dataset
        ALL_RUN_IDS = []

        # Print number of completed trials
        print(f"Number of completed trials: {len(study.trials)}")
        # Print number of failed trials
        n_failed_trials = len(
            [t for t in study.trials if t.value is None or (isinstance(t.value, float) and (t.value != t.value))])
        print(f"Number of failed trials: {n_failed_trials}")

        print("Hyperparameter tuning completed.")
