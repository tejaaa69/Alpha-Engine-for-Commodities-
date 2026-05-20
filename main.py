"""
main.py

Master Orchestrator for the Alchemist Engine.
Executes: Ingestion → Feature Store → Walk-Forward CV → Final Train → SHAP → Backtest → Registry.

Usage:
  python main.py --mode train    # Run full pipeline and register model
  python main.py --mode ingest   # Fetch fresh data only
  python main.py --mode predict  # Run live inference via MLflow Registry
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import mlflow
from dotenv import load_dotenv
from loguru import logger

# Load environment variables (API Keys)
load_dotenv()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_ingestion(cfg: dict):
    from src.ingestion.market import MarketIngestion
    from src.ingestion.macro import MacroIngestion
    from src.ingestion.news import NewsIngestion

    logger.info("=== PHASE 1: DATA INGESTION ===")
    MarketIngestion(cfg).run()
    MacroIngestion(cfg).run()
    # News ingestion matches the signature from our elite news.py
    NewsIngestion(cfg).run(assets=["gold", "silver"])


def run_full_pipeline(cfg: dict, symbol: str = None):
    from src.backtest.engine import BacktestEngine
    from src.explainability.shap_engine import SHAPExplainer
    from src.features.store import FeatureStore
    from src.models.lgbm_model import AlchemistModel, purged_walk_forward_cv
    from src.tracking.mlflow_utils import AlchemistTracker
    from src.ingestion.market import MarketIngestion

    gold_symbol = cfg["assets"]["gold"]
    if symbol is None:
        symbol = gold_symbol

    tracker = AlchemistTracker(cfg)

    # Step 1: Ingestion
    run_ingestion(cfg)

    # ── Step 2: Feature Store (Now handles targets & weights internally) ──
    logger.info("=== PHASE 2: FEATURE STORE ===")
    store = FeatureStore(cfg)
    df = store.build(symbol, force_rebuild=True)
    feature_cols = store.get_feature_columns(symbol)

    # Convert Triple Barrier Label (-1, 0, 1) to Binary Target (0, 1) for LightGBM
    df["binary_target"] = (df["label"] == 1).astype(int)

    logger.info(f"Master Dataset: {len(df)} rows × {len(feature_cols)} features")

    # ── Step 3: Monotonic Constraints
    monotonic = store.get_monotonic_constraints(feature_cols)

    # ── Step 4: Execution Block (Tracked via MLflow) 
    logger.info("=== PHASE 3: WALK-FORWARD CV & MODELING ===")
    
    with tracker.start_run(run_name=f"Alchemist_Run_{symbol}", tags={"symbol": symbol}):
        tracker.log_config()

        # Walk-forward CV
        cv_results = purged_walk_forward_cv(
            df=df,
            feature_cols=feature_cols,
            cfg=cfg,
            monotonic_constraints=monotonic,
            embargo_days=cfg["labeling"]["horizon_days"],
        )
        tracker.log_cv_results(cv_results)

        # ── Step 5: Final Model Training on Full Data 
        logger.info("=== PHASE 4: FINAL MODEL TRAINING ===")

        # Temporal Split: 80% Train, 10% Calibrate, 10% Test
        n = len(df)
        train_end = int(n * 0.80)
        cal_end   = int(n * 0.90)

        train_df = df.iloc[:train_end]
        cal_df   = df.iloc[train_end:cal_end]
        test_df  = df.iloc[cal_end:]

        X_train, y_train = train_df[feature_cols], train_df["binary_target"]
        X_cal, y_cal     = cal_df[feature_cols],   cal_df["binary_target"]
        X_test, y_test   = test_df[feature_cols],  test_df["binary_target"]

        # Reconstruct Elite Weights (Uniqueness * Time Decay)
        base_weights = train_df["sample_weight"]
        time_decay = np.linspace(0.5, 1.0, len(train_df))
        final_weights = (base_weights * time_decay)
        final_weights = final_weights / final_weights.mean()

        final_model = AlchemistModel(cfg)
        final_model.fit(
            X_train, y_train,
            X_val=X_cal,
            y_val=y_cal,
            sample_weights=final_weights,
            monotonic_constraints=monotonic,
        )
        final_model.calibrate(X_cal, y_cal)

        # Evaluate and Log Final Holdout Metrics
        test_metrics = final_model.evaluate(X_test, y_test)
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"final_test_{k}", v)

        # Save Custom Payload Locally
        model_path = Path(cfg["paths"]["models"]) / "alchemist_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(model_path)

        # ── Step 6: SHAP Explainability 
        logger.info("=== PHASE 5: SHAP EXPLAINABILITY ===")
        # Fit background on a sample of training data, compute on test data
        background_sample = X_train.sample(min(300, len(X_train)), random_state=42)

        shap_engine = SHAPExplainer(final_model.model, feature_cols)
        shap_engine.fit(background_sample)
        shap_engine.compute(X_test)

        importance = shap_engine.global_importance(top_n=20)
        tracker.log_shap_importance(importance)
        logger.info(f"Top 5 Drivers:\n{importance.head(5).to_string()}")

        # ── Step 7: Dynamic Backtest 
        logger.info("=== PHASE 6: DYNAMIC BACKTEST ===")
        # Gather out-of-sample predictions from CV folds
        all_probs = pd.concat([r["oos_probs"] for r in cv_results if "oos_probs" in r]).sort_index()
        
        # Load Raw prices for the execution engine
        price_df = MarketIngestion(cfg).load(symbol)

        bt_engine = BacktestEngine(cfg)
        bt_result = bt_engine.run(
            price_df=price_df,
            signals=all_probs,
            threshold=0.55,
            regime_series=df["regime_code"] if "regime_code" in df.columns else None,
        )

        tracker.log_backtest_metrics(bt_result.metrics)

        # ── Step 8: MLflow Registry 
        logger.info("=== PHASE 7: MODEL REGISTRY ===")
        run_id = mlflow.active_run().info.run_id
        
        # This securely uploads our custom joblib to MLflow and tags it "Staging"
        tracker.save_and_register_model(model_path, run_id, threshold_auc=0.45)

    logger.success("ALCHEMIST PIPELINE COMPLETE.")
    return {
        "cv_results":  cv_results,
        "bt_result":   bt_result,
        "model_path":  model_path,
        "shap_engine": shap_engine,
    }


def run_live_signal(cfg: dict, symbol: str = None):
    """
    Generate today's signal by pulling the Production model straight from MLflow.
    """
    from src.features.store import FeatureStore
    from src.tracking.mlflow_utils import AlchemistTracker

    gold_symbol = cfg["assets"]["gold"]
    if symbol is None:
        symbol = gold_symbol

    logger.info("Fetching latest market data...")
    run_ingestion(cfg)

    store = FeatureStore(cfg)
    df = store.build(symbol, force_rebuild=True)
    feature_cols = store.get_feature_columns(symbol)

    # Pull model from MLOps Registry
    tracker = AlchemistTracker(cfg)
    model = tracker.load_production_model()
    
    if model is None:
        logger.error("Could not load a Production or Staging model. Please train the model first.")
        return

    # Extract the absolute latest row of data
    latest_features = df[feature_cols].iloc[[-1]]
    latest_date     = latest_features.index[0]

    prob = model.predict_proba(latest_features)[0, 1]
    pred = "BUY SIGNAL (PROFIT TARGET EXPECTED)" if prob >= 0.55 else "FLAT/NO SIGNAL"

    logger.info(f"\n{'='*60}")
    logger.info(f"🔮 ALCHEMIST LIVE SIGNAL | {latest_date.date()} | {symbol}")
    logger.info(f"   Probability of hitting +2% before -1%: {prob:.2%}")
    logger.info(f"   Action: {pred}")
    logger.info(f"{'='*60}\n")

    return {"date": latest_date, "prob": prob, "signal": pred}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alchemist AI Trading System")
    parser.add_argument("--mode", choices=["train", "ingest", "predict"], default="train")
    parser.add_argument("--symbol", default=None, help="Asset symbol override")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "ingest":
        run_ingestion(cfg)
    elif args.mode == "predict":
        run_live_signal(cfg, symbol=args.symbol)
    else:
        run_full_pipeline(cfg, symbol=args.symbol)