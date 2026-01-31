import os
import subprocess
import tomllib
import gc
import time

import wandb
import optuna

RUN_TESTS_ONLY = False

N_TRIALS = 50 # optuna trials for one study
N_REPEAT_RUNS = 10 # repeat best run this many times to get stable estimate of performance
DELETE_EXISTING = False # if True will delete existing runs and start fresh. Otherwise, will append to existing project
DATA_DIR = "/Users/morris/code/fuzzy_clust_regression/data" # path to data directory containing .dat files

DATASET_NAME = None # will be set (and updated) later
DIRICHLET_ALPHA = None
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

def run_subprocess_and_get_latest_run(cmd):
    # Create a cleaned environment for the child process to avoid inheriting WANDB_* variables
    child_env = os.environ.copy()
    for key in list(child_env.keys()):
        if key.startswith("WANDB_"):
            child_env.pop(key, None)

    subprocess.run(cmd, check=True, env=child_env)

    # we need to create a new API instance since it is lazy and does not refresh automatically
    runs = wandb.Api().runs(wandb_project_name, order="-created_at")
    latest_run = runs[0]
    print(f"Latest run: {latest_run.id}")
    print(f"Latest run name: {latest_run.name}")

    # Return the latest run in the project
    return latest_run



def trigger_run(alpha, num_clusters, fuzzy_m, clip_lower_q, clip_upper_q, max_attempts = 3):
    """
    Trigger a new FLWR run with the given hyperparameters and return the latest run.
    :param alpha: Ridge regression alpha parameter.
    :param num_clusters: Number of clusters.
    :param fuzzy_m: Fuzziness parameter.
    :param clip_lower_q: Preprocessing lower quantile clipping.
    :param clip_upper_q: Preprocessing upper quantile clipping.
    :param max_attempts: How many times to retry in case of subprocess failure.
    :return: A wandb run object representing the latest run (or None if all attempts fail).
    """

    if DIRICHLET_ALPHA is None:
        # use default value from pyproject.toml
        cmd = ["flwr", "run", ".", "--run-config",
               f'dataset-name="{DATASET_NAME}" ridge-alpha={alpha} num-clusters={num_clusters} fuzzy-m={fuzzy_m} clip-lower-q={clip_lower_q} clip-upper-q={clip_upper_q}']
    else:
        cmd = ["flwr", "run", ".", "--run-config",
               f'dataset-name="{DATASET_NAME}" ridge-alpha={alpha} num-clusters={num_clusters} fuzzy-m={fuzzy_m} clip-lower-q={clip_lower_q} clip-upper-q={clip_upper_q} dirichlet-alpha={DIRICHLET_ALPHA}']

    attempt = 0
    while attempt < max_attempts:
        try:
            latest_run = run_subprocess_and_get_latest_run(cmd)
            ALL_RUN_IDS.append(latest_run.id)
            break
        except subprocess.CalledProcessError as e:
            attempt += 1
            print(f"Subprocess failed on attempt {attempt}/{max_attempts} with error: {e}")
            if attempt == max_attempts:
                return None

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
    if run is None:
        return float('nan') # signal optuna that the trial failed

    # wait a bit for wandb to process the run
    time.sleep(1)
    return calculate_validation_rmse(run)

sets_done = []

eval_sets = [
    "wizmir",
    "delta_elv",
    "elevators",
    "california",
    "house",
    "mortgage",
    "wankara",
    "treasury"
]

if __name__ == "__main__":
    # Base config (load pyproject.toml file)
    with open("pyproject.toml", "rb") as f:
        base_config = tomllib.load(f)

    wandb_api = wandb.Api()

    # Find all .dat files in DATA_DIR
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
    dataset_names = [os.path.splitext(f)[0] for f in data_files]
    print(f"Found {len(data_files)} data files in {DATA_DIR}")

    for dataset_name in eval_sets:
        if dataset_name in sets_done:
            print(f"Skipping dataset {dataset_name} since it is already done.")
            continue

        wandb_project_name = f"fed-fuzzy-clust-regression-{dataset_name}"
        if not RUN_TESTS_ONLY:
            print(f"Starting hyperparameter tuning for {dataset_name}")
            # set global dataset name
            DATASET_NAME = dataset_name
            # Make sure to reset run ids for next dataset
            ALL_RUN_IDS = []

            # derive W&B project name from dataset name (will be used by optuna study)
            #dataset_name = base_config["tool"]["flwr"]["app"]["config"]["dataset-name"]


            # Check if project exists and how many runs it has
            try:
                runs = wandb_api.runs(wandb_project_name)#, filters={"state": "finished"})
                n_runs = len(runs)
                print(f"Project {wandb_project_name} exists with {n_runs} runs.")
            except ValueError:
                n_runs = 0
                print(f"Project {wandb_project_name} does not exist yet.")

            # continue if there are enough runs already
            if DELETE_EXISTING:
                if n_runs >= 1000:
                    print(f"Skipping hyperparameter tuning for {dataset_name} since there are already {n_runs} runs.")
                    continue
                elif n_runs > 0:
                    # Delete all runs in the project to start fresh
                    print(f"Deleting all existing runs in project {wandb_project_name} to start fresh.")
                    for run in runs:
                        run.delete()
                        #run.update()
                    # wait a bit for wandb to process deletions
                    print("Waiting 10 seconds for wandb to process deletions...")
                    time.sleep(10)

            else:
                print(f"Appending to project for {dataset_name} with existing {n_runs} runs.")

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

        # Fetch best run from wandb
        completed_sets = []
        if dataset_name in completed_sets:
            print(f"Skipping repeat runs for completed dataset {dataset_name}")
            continue
        try:
            best_run = wandb_api.runs(wandb_project_name,
                                      filters={"state": "finished", "tags": {"$in": ["validation_best_run"]},
                                               "config.dirichlet-alpha": 100,
                                               "config.clust-max-iter-global": 1}, order="-created_at")[0]
        except:
            print(f"Project {wandb_project_name} does not have a best run yet. Skipping repeat runs.")
            continue
        # Ensure DATASET_NAME is set
        DATASET_NAME = dataset_name
        # Repeat the best run ten times to get more stable estimate of performance
        for dirichlet_alpha in [100.0, 0.5]:
            DIRICHLET_ALPHA = dirichlet_alpha
            print(f"Setting DIRICHLET_ALPHA to {DIRICHLET_ALPHA} for dataset {dataset_name}")
            for repeat_idx in range(N_REPEAT_RUNS):
                print(f"Repeating best run, iteration {repeat_idx + 1}/{N_REPEAT_RUNS}")

                run = trigger_run(alpha=best_run.config["ridge-alpha"],
                                  num_clusters=best_run.config["num-clusters"],
                                  fuzzy_m=best_run.config["fuzzy-m"],
                                  clip_lower_q=best_run.config["clip-lower-q"],
                                  clip_upper_q=best_run.config["clip-upper-q"])

                val_rmse = calculate_validation_rmse(run)
                print(f"Repeat {repeat_idx + 1}/{N_REPEAT_RUNS}, Validation RMSE: {val_rmse}")
                # Tag the repeat runs
                run.tags = run.tags + ["validation_best_run_repeat"]
                run.update()

                # Clean up memory by collecting garbage
                gc.collect()
        DIRICHLET_ALPHA = None # reset for next dataset
