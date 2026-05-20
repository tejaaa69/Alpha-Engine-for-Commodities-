"""
src/models/lgbm_model.py

LightGBM classifier for triple-barrier event prediction.
Predicts: will the asset hit the profit target before the stop loss?

Upgrades:
  1. Strict Purged Walk-Forward CV (Zero Validation/Test Leakage)
  2. Integration of Concurrency Sample Weights + Time Decay
  3. MLflow Experiment Tracking for hyperparameter and metric logging
  4. Isotonic Calibration on a strictly isolated dataset
"""

import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    log_loss
)

class AlchemistModel:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg              = cfg
        self.model_cfg        = cfg["model"]
        self.model            = None
        self.calibrated_model = None
        self.feature_cols     = None
        self.is_calibrated    = False

    def _build_lgbm_params(self, monotonic_constraints: List[int]) -> dict:
        mc = self.model_cfg
        params = {
            "objective":          "binary",
            "metric":             ["binary_logloss", "auc"],
            "n_estimators":       mc["n_estimators"],
            "learning_rate":      mc["learning_rate"],
            "max_depth":          mc["max_depth"],
            "num_leaves":         mc["num_leaves"],
            "min_child_samples":  mc["min_child_samples"],
            "subsample":          mc["subsample"],
            "colsample_bytree":   mc["colsample_bytree"],
            "reg_alpha":          mc["reg_alpha"],
            "reg_lambda":         mc["reg_lambda"],
            "class_weight":       mc.get("class_weight", "balanced"),
            "n_jobs":             -1,
            "random_state":       42,
            "verbose":            -1,
        }

        if any(c != 0 for c in monotonic_constraints):
            params["monotone_constraints"]        = monotonic_constraints
            params["monotone_constraints_method"] = "advanced"
            logger.info(
                f"Monotonic constraints applied to "
                f"{sum(c!=0 for c in monotonic_constraints)}/{len(monotonic_constraints)} features."
            )
        return params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        monotonic_constraints: Optional[List[int]] = None,
    ) -> "AlchemistModel":
        
        if monotonic_constraints is None:
            monotonic_constraints = [0] * len(X_train.columns)

        self.feature_cols = list(X_train.columns)
        params = self._build_lgbm_params(monotonic_constraints)

        self.model = lgb.LGBMClassifier(**params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0), # Silence the spammy output
        ]

        # LightGBM handles sample weights natively via the fit method
        fit_params = {
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
        if sample_weights is not None:
            fit_params["sample_weight"] = sample_weights.values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train, **fit_params)

        logger.info(f"Model trained. Optimal trees: {self.model.best_iteration_}")
        return self

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> "AlchemistModel":
        method = self.cfg["calibration"]["method"]
        self.calibrated_model = CalibratedClassifierCV(
            estimator = self.model,
            method    = method,
            cv        = "prefit",
        )
        self.calibrated_model.fit(X_cal, y_cal)
        self.is_calibrated = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.is_calibrated:
            return self.calibrated_model.predict_proba(X)
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        probs  = self.predict_proba(X)[:, 1]
        preds  = (probs >= 0.5).astype(int)
        
        # Guard against single-class arrays in small folds
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = 0.5

        metrics = {
            "accuracy":      round(accuracy_score(y, preds), 4),
            "roc_auc":       round(auc, 4),
            "brier_score":   round(brier_score_loss(y, probs), 4),
            "log_loss":      round(log_loss(y, probs), 4),
            "positive_rate": round(float(y.mean()), 4),
        }
        return metrics
    
    def save(self, path):
        """Saves the trained model and calibrator to disk."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Loads a trained model from disk."""
        import joblib
        return joblib.load(path)


#Purged Walk-Forward Cross-Validation

def purged_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list,
    cfg: dict,
    monotonic_constraints: list,
    embargo_days: int = 7,
) -> list:
    
    wf_cfg       = cfg["walk_forward"]
    train_months = wf_cfg["train_months"]
    test_months  = wf_cfg["test_months"]
    min_train    = wf_cfg["min_train_obs"]

    df = df.sort_index()
    dates = df.index
    results = []
    fold = 0

    # Ensure Target is Binary: 1 (Hit Profit) vs 0 (Hit Stop or Timeout)
    target_col = "binary_target"
    df[target_col] = (df["label"] == 1).astype(int)

    # Initialize MLflow experiment
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="Purged_WF_CV_Run",nested=True):
        mlflow.log_params(cfg["model"])

        start = dates[0]
        end   = dates[-1]
        test_start = start + pd.DateOffset(months=train_months)

        while test_start < end:
            test_end  = min(test_start + pd.DateOffset(months=test_months), end)
            
            # THE FIX: Leakage-free splitting
            # Embargo gap between train/val and test
            train_val_end = test_start - pd.Timedelta(days=embargo_days)
            
            train_val_mask = (dates >= start) & (dates < train_val_end)
            test_mask      = (dates >= test_start) & (dates <= test_end)

            train_val_df = df.loc[train_val_mask]
            test_df      = df.loc[test_mask]

            if len(train_val_df) < min_train or len(test_df) == 0:
                test_start += pd.DateOffset(months=test_months)
                fold += 1
                continue

            # Split Train/Val/Cal temporally (80% Train, 10% Val, 10% Cal)
            n_obs = len(train_val_df)
            train_idx = int(n_obs * 0.8)
            val_idx   = int(n_obs * 0.9)

            train_df = train_val_df.iloc[:train_idx]
            val_df   = train_val_df.iloc[train_idx:val_idx]
            cal_df   = train_val_df.iloc[val_idx:]

            # Extract matrices
            X_tr, y_tr = train_df[feature_cols], train_df[target_col]
            X_val, y_val = val_df[feature_cols], val_df[target_col]
            X_cal, y_cal = cal_df[feature_cols], cal_df[target_col]
            X_test, y_test = test_df[feature_cols], test_df[target_col]

            # ELITE WEIGHTS: Uniqueness Weight * Time Decay
            base_weights = train_df["sample_weight"] if "sample_weight" in train_df.columns else pd.Series(1.0, index=train_df.index)
            time_decay = np.linspace(0.5, 1.0, len(train_df))
            final_weights = base_weights * time_decay
            final_weights = final_weights / final_weights.mean() # Normalize

            # Train & Calibrate
            model = AlchemistModel(cfg)
            model.fit(
                X_tr, y_tr, X_val, y_val,
                sample_weights=final_weights,
                monotonic_constraints=monotonic_constraints
            )
            model.calibrate(X_cal, y_cal)

            # Evaluate purely out-of-sample
            metrics = model.evaluate(X_test, y_test)
            metrics["fold"] = fold
            
            logger.info(
                f"Fold {fold} | Test: {test_start.date()} to {test_end.date()} | "
                f"AUC={metrics['roc_auc']:.3f} | Acc={metrics['accuracy']:.3f}"
            )
            
            # Log metrics to MLflow per fold
            mlflow.log_metrics({f"fold_{fold}_auc": metrics["roc_auc"], f"fold_{fold}_brier": metrics["brier_score"]})

            # Store OOS predictions
            probs = model.predict_proba(X_test)[:, 1]
            metrics["oos_probs"]  = pd.Series(probs, index=X_test.index)
            metrics["oos_labels"] = y_test
            results.append(metrics)

            test_start += pd.DateOffset(months=test_months)
            fold += 1

        # Aggregate and log final metrics
        avg_auc = np.mean([r["roc_auc"] for r in results])
        avg_acc = np.mean([r["accuracy"] for r in results])
        
        mlflow.log_metrics({"mean_cv_auc": avg_auc, "mean_cv_accuracy": avg_acc})
        logger.success(f"Walk-forward CV complete. Mean AUC: {avg_auc:.3f}")

    return results