"""fed-fuzzy-clust-regression: A Flower / sklearn app."""
import os
import shutil

from flwr.app import ArrayRecord, Context
from flwr.common import ConfigRecord
from flwr.serverapp import Grid, ServerApp
import wandb

from fed_fuzzy_clust_regression.fed_fuzzy_avg import FedFuzzyAvg
from fed_fuzzy_clust_regression.task import load_data, get_federated_dataset, get_model, get_model_params, \
    set_initial_regression_params, evaluate_on_test_set, set_model_params, evaluate_clustering, get_working_dir
from fed_fuzzy_clust_regression.cluster_tasks import fed_clustering_model_from_partitions

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context):
    # Read parameters from config
    num_rounds: int = context.run_config["num-server-rounds"]
    num_clients: int = context.run_config["num-clients"]
    dataset_name: str = context.run_config["dataset-name"]
    dirichlet_alpha: float = context.run_config["dirichlet-alpha"]

    ## Clustering parameters
    num_clusters: int = context.run_config["num-clusters"]  # also number of regression models
    fuzzy_m = context.run_config["fuzzy-m"]
    clust_tol_local = context.run_config["clust-tol-local"]
    clust_tol_global = context.run_config["clust-tol-global"]
    clust_max_iter_local = context.run_config["clust-max-iter-local"]
    clust_max_iter_global = context.run_config["clust-max-iter-global"]
    # Regression parameter
    ridge_alpha = context.run_config["ridge-alpha"]
    # Data preprocessing parameters
    apply_preprocessing = context.run_config["apply-preprocessing"] == "True"
    q_lower = context.run_config["clip-lower-q"]
    q_upper = context.run_config["clip-upper-q"]

    # This is the temporary directory where the data is saved
    working_dir = get_working_dir()
    # Delete all files and working dir if exists
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)

    # Save data to disk for clustering model
    # Ensure global fds exists
    _ = load_data(0, num_clients, dataset_name, working_dir, dirichlet_alpha, q_lower, q_upper, apply_preprocessing)
    # for each client, save data to disk
    global_fds = get_federated_dataset()

    for partition_id in range(num_clients):
        global_fds.save_partition_as_csv(partition_id=partition_id)

    # Initialize wandb (makes sure to start a new run)
    wandb.finish()
    wandb_project_name = f"fed-fuzzy-clust-regression-{dataset_name}"
    wandb.init(project=wandb_project_name,
               settings=wandb.Settings(
                   x_label="Server",
                   mode="shared",
                   x_primary=True
               )
               )

    # Log parameters to wandb
    wandb.config.update(context.run_config)

    # Add run id to context to make it available to clients
    run_id = wandb.run.id
    run_config = ConfigRecord({"wandb_run_id": run_id,
                               "wandb_project_name": wandb_project_name})
    # Save train and test splits for centralized comparison
    global_fds.save_train_split_as_csv()
    global_fds.save_test_split_as_csv()

    # Train federated fuzzy clustering model
    fed_cluster_model = fed_clustering_model_from_partitions(data_dir=working_dir, num_partitions=num_clients,
                                                             num_clusters=num_clusters, m=fuzzy_m,
                                                             tol_local=clust_tol_local, tol_global=clust_tol_global,
                                                             max_iter_local=clust_max_iter_local,
                                                             max_global_iter=clust_max_iter_global
                                                             )
    print(f'Cluster model converged after {fed_cluster_model.iterations} iterations.')
    print(f'Final cluster centers: {fed_cluster_model.cluster_centers}')
    evaluate_clustering(fed_cluster_model, wandb_run_id=run_id, wandb_project_name=wandb_project_name)

    # Log number of iterations to wandb
    wandb.log({"cluster_num_iterations": fed_cluster_model.iterations})

    # Train clusterwise ridge regression model
    regr_model = get_model(fed_cluster_model.cluster_centers, m=fuzzy_m, alpha=ridge_alpha)
    # Construct ArrayRecord representation
    # Initialize regression models to construct ArrayRecord correctly
    regr_model = set_initial_regression_params(model=regr_model, n_features=fed_cluster_model.cluster_centers.shape[1])
    arrays = ArrayRecord(get_model_params(regr_model))
    wandb.finish() # finish the run, strategy will use wandb API

    # strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)
    strategy = FedFuzzyAvg(fraction_train=1.0, fraction_evaluate=1.0, wandb_run_id=run_id,
                           wandb_project_name=wandb_project_name)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=run_config,
        evaluate_config=run_config # wandb logging does not work yet
    )

    ndarrays = result.arrays.to_numpy_ndarrays()
    set_model_params(regr_model, ndarrays)
    wandb.init(project=wandb_project_name, id=run_id, resume="must")
    evaluate_on_test_set(regr_model)
    wandb.finish()
