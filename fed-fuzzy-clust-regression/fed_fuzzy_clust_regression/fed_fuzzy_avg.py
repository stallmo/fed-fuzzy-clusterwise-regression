import time
import traceback

import wandb
from flwr.serverapp.strategy import FedAvg
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, FitIns
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

def extract_metrics_from_message(entry: Any) -> Optional[Dict[str, float]]:
    """
    Accepts either (client_id, message) tuple or a Message-like object.
    Returns a plain dict of metrics or None if not found/convertible.
    """
    # Unpack tuple entries sometimes passed by Flower
    if isinstance(entry, tuple) and len(entry) == 2:
        _, msg = entry
    else:
        msg = entry

    # Get content (RecordDict) from Message
    content = getattr(msg, "content", msg)

    # Try dict-like access first
    metrics = None
    if hasattr(content, "get"):
        try:
            metrics = content.get("metrics")
        except Exception:
            metrics = None
    if metrics is None:
        try:
            metrics = content["metrics"]  # type: ignore
        except Exception:
            # maybe the whole content is the metrics dict
            metrics = content

    # Normalize to dict
    if isinstance(metrics, dict):
        return metrics
    if metrics is None:
        return None
    if hasattr(metrics, "to_dict"):
        try:
            return metrics.to_dict()
        except Exception:
            pass
    if hasattr(metrics, "value"):
        val = getattr(metrics, "value")
        if isinstance(val, dict):
            return val
    if hasattr(metrics, "items"):
        try:
            return dict(metrics)
        except Exception:
            pass
    if hasattr(metrics, "__dict__"):
        return vars(metrics)

    return None


class FedFuzzyAvg(FedAvg):

    def __init__(self, wandb_run_id: str = None, wandb_project_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self.wandb_run_id = wandb_run_id
        self.wandb_project_name = wandb_project_name

    def aggregate_fit(self, server_round: int, results: List[Tuple[int, FitRes]], failures: List[Tuple[int, FitRes]]) -> \
            Tuple[Parameters, dict]:
        if not results:
            return None, {}
        if failures:
            # Handle failures as needed (e.g., log or skip)
            pass

        # Extract parameters and metrics
        parameters_list = [fit_res.parameters for _, fit_res in results]
        metrics_list = [fit_res.metrics for _, fit_res in results]

        # Assume all clients have the same n_clusters; get from first client's parameters
        first_params = parameters_list[0].to_numpy_ndarrays()
        n_clusters = int(first_params[-1].item())  # n_clusters is stored in last parameter
        num_reg_params = n_clusters * 2  # coef + intercept per cluster

        # Initialize aggregated reg params
        aggregated_reg_params = [np.zeros_like(first_params[i]) for i in range(num_reg_params)]

        # For each cluster, collect weights and params
        for cluster_idx in range(n_clusters):
            coef_idx = 2 * cluster_idx
            inter_idx = 2 * cluster_idx + 1
            weights = [m["fuzzy_cardinalities"][cluster_idx] for m in metrics_list]
            total_weight = sum(weights)

            if total_weight == 0:
                # Fallback to equal weighting if no weights
                print("Warning: No weights found for cluster {}, falling back to equal weighting.".format(cluster_idx))
                weights = [1.0] * len(weights)
                total_weight = len(weights)

            # Weighted sum for coef and intercept
            for client_idx, params in enumerate(parameters_list):
                client_params = params.to_numpy_ndarrays()
                weight = weights[client_idx] / total_weight
                aggregated_reg_params[coef_idx] += weight * client_params[coef_idx]
                aggregated_reg_params[inter_idx] += weight * client_params[inter_idx]

        # Keep global params (centers, m, n_clusters) from first client (or server initial)
        global_params = first_params[-3:]  # centers, m, n_clusters
        aggregated_params = aggregated_reg_params + global_params

        # Return aggregated Parameters
        return ndarrays_to_parameters(aggregated_params), {}

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[int, object]],
                           failures: List[Tuple[int, object]] = None) -> Tuple[Optional[float], dict]:
        """
        Consolidate evaluation metrics from clients and log to WandB (via API).
        - Collect numeric per-client metrics as `client_{id}/{metric}`.
        - Compute averages across clients as `avg_{metric}`.
        - Update WandB run summary using wandb.Api().run(...).summary.update(...)
        Returns (None, {}) as evaluation aggregation payload (no change to server logic).
        """
        if not results:
            return None, {}

        # Collect numeric metrics
        per_client_summary = {}
        numeric_accum = {}

        for client_id, eval_res in enumerate(results):
            # eval_res may be an object with .metrics or a dict; handle both
            metrics = extract_metrics_from_message(eval_res)
            print(f"[server] Extracted metrics from client {client_id}: {metrics}")
            if metrics is None:
                raise ValueError(f"Could not extract metrics from eval result of client {client_id}: {eval_res}")
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                # Skip non-scalar values (e.g., lists like fuzzy_cardinalities)
                try:
                    val = float(v)
                except Exception:
                    continue
                per_client_summary[f"client_{client_id}/{k}"] = float(val)
                numeric_accum.setdefault(k, []).append(float(val))

        # Compute averages
        averaged = {}
        for k, vals in numeric_accum.items():
            try:
                averaged[f"avg_{k}"] = float(np.mean(vals))
            except Exception:
                averaged[f"avg_{k}"] = None

        # Merge per-client keys and averages into summary update payload
        summary_update = {}
        summary_update.update(per_client_summary)
        summary_update.update(averaged)

        # Update WandB run summary via API with retries (server-side single process)
        if self.wandb_project_name and self.wandb_run_id:
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    api = wandb.Api()
                    run = api.run(f"{self.wandb_project_name}/{self.wandb_run_id}")
                    run.summary.update(summary_update)
                    run.update()
                    print(f"[server] WandB summary updated for round {server_round} (attempt {attempt})")
                    break
                except Exception as exc:
                    print(f"[server] WandB Api update attempt {attempt} failed: {exc}")
                    traceback.print_exc()
                    time.sleep(0.5 * attempt)
        else:
            print("[server] WandB project/run not configured on strategy; skipping consolidated logging.")

        # Return no aggregated loss/metrics to Flower (preserve default behavior)
        return {}
