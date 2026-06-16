"""
src/tracking/mlflow_utils.py

Enterprise MLflow experiment tracking and model registry.

Upgrades:
  1. Custom Payload Handling: Safely logs and loads our custom AlchemistModel 
     (which contains calibration wrappers) instead of crashing on raw LightGBM flavors.
  2. Nested Run Support: Allows the tracker to seamlessly attach to the active 
     run created by our Purged Walk-Forward CV loop.
  3. Metric Type Safety: Explicitly casts metrics to floats to prevent MLflow 
     serialization crashes caused by numpy data types.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import mlflow
import pandas as pd
import yaml
from loguru import logger

# Import our custom model class to handle proper loading
from src.models.lgbm_model import AlchemistModel


class AlchemistTracker:
    def __init__(self, cfg: Dict[str, Any], symbol: str = "GLD"):
        self.cfg         = cfg
        self.exp_name    = cfg["mlflow"]["experiment_name"]
        
        # Pull the naming template from config, fallback gracefully if it doesn't exist
        template = cfg["mlflow"].get("registered_model_name_template", cfg["mlflow"]["registered_model_name"])
        
        # Dynamically inject the active asset symbol (e.g. alchemist_lgbm_GLD)
        if "{symbol}" in template:
            self.model_name = template.format(symbol=symbol)
        else:
            self.model_name = f"{template}_{symbol}"
        self.symbol      = symbol
        
        # Resolve tracking URI to an absolute path to prevent folder scattering
        root_dir = Path(__file__).resolve().parent.parent.parent
        local_path = root_dir / cfg['paths']['mlflow_uri']
        self.tracking_uri = f"file:///{local_path.as_posix()}"

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.exp_name)
    
    def start_run(self, run_name: str = None, tags: dict = None):
        """Starts and returns the MLflow run context manager."""
        import mlflow
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_config(self):
        """Log the entire config.yaml as an artifact for 100% reproducibility."""
        config_str = yaml.dump(self.cfg, default_flow_style=False)
        mlflow.log_text(config_str, "config.yaml")

    def log_cv_results(self, cv_results: List[dict]):
        """Log per-fold and aggregate walk-forward CV metrics."""
        if not cv_results:
            return
            
        auc_list   = [float(r["roc_auc"])     for r in cv_results]
        acc_list   = [float(r["accuracy"])    for r in cv_results]
        brier_list = [float(r["brier_score"]) for r in cv_results]

        mlflow.log_metric("cv_mean_auc",      round(sum(auc_list)   / len(auc_list),  4))
        mlflow.log_metric("cv_mean_accuracy", round(sum(acc_list)   / len(acc_list),  4))
        mlflow.log_metric("cv_mean_brier",    round(sum(brier_list) / len(brier_list),4))
        mlflow.log_metric("cv_n_folds",       len(cv_results))

        # Drop large arrays before saving table
        clean_results = []
        for r in cv_results:
            clean_r = {k: v for k, v in r.items() if k not in ("oos_probs", "oos_labels")}
            clean_results.append(clean_r)
            
        fold_df = pd.DataFrame(clean_results)
        mlflow.log_text(fold_df.to_csv(index=False), "cv_fold_results.csv")

    def log_backtest_metrics(self, metrics: dict):
        """Log backtest performance metrics safely."""
        numeric_keys = [
            "total_trades", "win_rate", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "total_return", "profit_factor",
            "avg_win", "avg_loss", "final_capital",
            "barrier_profit", "barrier_stop", "barrier_time",
        ]
        for k in numeric_keys:
            if k in metrics:
                try:
                    # Explicit float cast to prevent numpy type crashing MLflow
                    mlflow.log_metric(f"bt_{k}", float(metrics[k]))
                except (TypeError, ValueError):
                    pass

    def log_shap_importance(self, importance_df: pd.DataFrame):
        """Log top feature importances from SHAP."""
        mlflow.log_text(
            importance_df.to_csv(index=False),
            "shap_global_importance.csv"
        )
        for _, row in importance_df.head(10).iterrows():
            safe_name = row["feature"].replace("/", "_").replace(" ", "_")
            mlflow.log_metric(f"shap_{safe_name}", round(float(row["importance"]), 6))

    def save_and_register_model(self, model_path: Path, run_id: str, threshold_auc: float = 0.55) -> bool:
        """
        Log our CUSTOM model payload as a generic artifact, then register it.
        """
        client = mlflow.MlflowClient()
        run    = client.get_run(run_id)
        mean_auc = run.data.metrics.get("cv_mean_auc", 0)

        # Log the actual .joblib file as an artifact
        mlflow.log_artifact(local_path=str(model_path), artifact_path="alchemist_payload")

        if mean_auc < threshold_auc:
            logger.warning(
                f"Model AUC {mean_auc:.3f} below threshold {threshold_auc}. "
                f"Logged as artifact, but NOT registering to Staging."
            )
            return False

        # Register the generic artifact path to the Model Registry
        artifact_uri = f"runs:/{run_id}/alchemist_payload/{model_path.name}"
        mv = mlflow.register_model(model_uri=artifact_uri, name=self.model_name)

        # Promote to Staging
        client.transition_model_version_stage(
            name    = self.model_name,
            version = mv.version,
            stage   = "Staging",
            archive_existing_versions=False,
        )
        logger.success(f"Model v{mv.version} registered as '{self.model_name}' → Staging. AUC={mean_auc:.3f}")
        return True

    def load_production_model(self) -> Optional["AlchemistModel"]:
        """
        Finds the 'Production' (or 'Staging') model in the registry, 
        downloads the custom .joblib artifact, and rebuilds the class.
        """
        client = mlflow.MlflowClient()
        
        try:
            # Look for Production first
            versions = client.get_latest_versions(self.model_name, stages=["Production"])
            if not versions:
                logger.warning("No Production model found. Falling back to Staging...")
                versions = client.get_latest_versions(self.model_name, stages=["Staging"])
                
            if not versions:
                logger.error("No model found in Production or Staging.")
                return None
                
            latest_version = versions[0]
            run_id = latest_version.run_id
            
            # Download the artifact directory
            local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="alchemist_payload")
            
            # Find the joblib file inside
            downloaded_path = Path(local_dir) / "alchemist_model.joblib"
            
            if downloaded_path.exists():
                logger.info(f"Successfully loaded {latest_version.current_stage} model from Registry.")
                return AlchemistModel.load(downloaded_path)
            else:
                logger.error("Artifact downloaded, but joblib file missing.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            return None