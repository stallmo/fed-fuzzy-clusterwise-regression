import wandb
import pandas as pd
import os

if __name__ == '__main__':
    DIRICHLET_ALPHA = 0.5
    CLUST_MAX_ITER = 1

    # get all dataset names
    data_dir = "/Users/morris/code/fuzzy_clust_regression/data"  # path to data directory containing .dat files
    # Find all .dat files in DATA_DIR
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    dataset_names = [os.path.splitext(f)[0] for f in data_files]
    dataset_names = [
        "wizmir",
        "delta_elv",
        "elevators",
        "california",
        "house",
        "mortgage",
        "wankara",
        "treasury"
    ]
    print(f"Found {len(data_files)} data files in {data_dir}")

    rows = []
    for dataset_name in dataset_names:
        # Connect to wandb and get specified project
        wandb_api = wandb.Api()
        wandb_project_name = f"fed-fuzzy-clust-regression-{dataset_name}"

        # Get all finished runs with validation specific tag
        try:
            runs = wandb_api.runs(wandb_project_name,
                                  filters={"state": "finished", "tags": {"$in": ["validation_best_run_repeat"]},
                                           "$and": [{"tags": {"$ne": "validation_failed_run"}}],
                                           "config.dirichlet-alpha": DIRICHLET_ALPHA,
                                           "config.clust-max-iter-global": CLUST_MAX_ITER,
                                           })
        except:
            print(f"Project {wandb_project_name} does not exist yet. Skipping.")
            continue
        # Define output paths
        save_path_test_results = f"test_results/all_test_runs_dirichlet_{DIRICHLET_ALPHA}_clust_max_iter_{CLUST_MAX_ITER}.csv"
        save_path_results_summary = f"test_results/summary_test_results_dirichlet_{DIRICHLET_ALPHA}_clust_max_iter_{CLUST_MAX_ITER}.csv"

        # For each run, get configuration and test metrics
        for run in runs:
            # Get configuration
            dirichlet_alpha = run.config["dirichlet-alpha"]
            clust_max_iter_global = run.config["clust-max-iter-global"]

            # Get hyperparameters
            q_lower = run.config["clip-lower-q"]
            q_upper = run.config["clip-upper-q"]
            ridge_alpha = run.config["ridge-alpha"]
            num_clusters = run.config["num-clusters"]
            fuzzy_m = run.config["fuzzy-m"]
            apply_preprocessing = run.config["apply-preprocessing"]

            # Get test metrics
            test_fuzzy_db = run.summary.get("test/fuzzy_db")
            test_rmse = run.summary.get("test/rmse")
            test_r2score = run.summary.get("test/r2")
            test_mae = run.summary.get("test/mae")
            test_mape = run.summary.get("test/mape")
            validation = run.summary.get("validation_rmse")

            # Print to console
            print(f"Run ID: {run.id}")
            print(f"Dataset: {dataset_name}")
            print(
                f"Hyperparameters: q_lower={q_lower}, q_upper={q_upper}, ridge_alpha={ridge_alpha}, num_clusters={num_clusters}, fuzzy_m={fuzzy_m}")
            print(
                f"Test Metrics: Fuzzy DB={test_fuzzy_db}, RMSE={test_rmse}, R2={test_r2score}, MAE={test_mae}, MAPE={test_mape}")

            # Create a dataframe row
            row = {
                "run_id": run.id,
                "dataset": dataset_name,
                "dirichlet_alpha": dirichlet_alpha,
                "clust_max_iter_global": clust_max_iter_global,
                "apply_preprocessing": apply_preprocessing,
                "q_lower": q_lower,
                "q_upper": q_upper,
                "ridge_alpha": ridge_alpha,
                "num_clusters": num_clusters,
                "fuzzy_m": fuzzy_m,
                "test_fuzzy_db": test_fuzzy_db,
                "test_rmse": test_rmse,
                "test_r2score": test_r2score,
                "test_mae": test_mae,
                "test_mape": test_mape,
                "validation_rmse": validation
            }
            rows.append(row)

    # Save all results fetched from wandb to a CSV
    results_df = pd.DataFrame(rows)
    # Save to CSV after ensuring directory exists
    os.makedirs(os.path.dirname(save_path_test_results), exist_ok=True)
    results_df.to_csv(save_path_test_results, index=False)
    print(f"Saved all test results to {save_path_test_results}")

    # Create a summary dataframe with mean and std of test metrics
    summary_df = results_df.groupby(
        ["dataset", "q_lower", "q_upper", "ridge_alpha", "num_clusters", "fuzzy_m", "apply_preprocessing"]).agg(
        test_fuzzy_db_mean=pd.NamedAgg(column="test_fuzzy_db", aggfunc="mean"),
        test_fuzzy_db_std=pd.NamedAgg(column="test_fuzzy_db", aggfunc="std"),
        test_rmse_mean=pd.NamedAgg(column="test_rmse", aggfunc="mean"),
        test_rmse_std=pd.NamedAgg(column="test_rmse", aggfunc="std"),
        test_r2score_mean=pd.NamedAgg(column="test_r2score", aggfunc="mean"),
        test_r2score_std=pd.NamedAgg(column="test_r2score", aggfunc="std"),
        test_mae_mean=pd.NamedAgg(column="test_mae", aggfunc="mean"),
        test_mae_std=pd.NamedAgg(column="test_mae", aggfunc="std"),
        test_mape_mean=pd.NamedAgg(column="test_mape", aggfunc="mean"),
        test_mape_std=pd.NamedAgg(column="test_mape", aggfunc="std"),
        validation_rmse_mean=pd.NamedAgg(column="validation_rmse", aggfunc="mean"),
        validation_rmse_std=pd.NamedAgg(column="validation_rmse", aggfunc="std"),
    ).reset_index().round(4)
    # Save summary to CSV after ensuring directory exists
    os.makedirs(os.path.dirname(save_path_results_summary), exist_ok=True)
    summary_df.to_csv(save_path_results_summary, index=False)
    print(f"Saved summary test results to {save_path_results_summary}")
