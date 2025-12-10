"""
assess_meta_model_accuracy.py - Evaluate accuracy of meta-model predictions

This script loads folds with meta-model predictions and evaluates their accuracy
against the actual target values (prev_flight_air_time, prev_flight_taxi_time, 
prev_flight_total_duration).

Usage:
    Set VERSIONS and INPUT_SUFFIX below, then run all cells.
"""

from __future__ import annotations
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from datetime import datetime

# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to assess (e.g., ["3M", "12M", "60M"])
VERSIONS = ["3M","12M","60M"]  # <-- EDIT THIS LIST

INPUT_SUFFIX = "_with_graph_and_metamodels"  # Suffix for folds with meta-model predictions

# Meta-model IDs to assess (should match what was used in add_meta_model_features.py)
META_MODEL_IDS = ["XGB_1"]  # <-- EDIT THIS LIST

# Target columns and their corresponding prediction columns
TARGETS = {
    "prev_flight_air_time": "predicted_prev_flight_air_time_XGB_1",
    "prev_flight_taxi_time": "predicted_prev_flight_taxi_time_XGB_1",
    "prev_flight_total_duration": "predicted_prev_flight_total_duration_XGB_1"
}

VERBOSE = True


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_cv_module():
    """Load the cv module using importlib (for Databricks compatibility)."""
    import importlib.util
    import sys
    
    # Try Databricks path first, fall back to local path if needed
    cv_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/cv.py"
    try:
        spec = importlib.util.spec_from_file_location("cv", cv_path)
    except:
        # Fallback to relative path for local development
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cv_path = os.path.join(script_dir, "../cv.py")
        spec = importlib.util.spec_from_file_location("cv", cv_path)
    
    cv = importlib.util.module_from_spec(spec)
    sys.modules["cv"] = cv
    spec.loader.exec_module(cv)
    return cv


def load_folds_for_version(version: str, input_suffix: str = ""):
    """Load all available folds for a version using cv.py's FlightDelayDataLoader."""
    cv = load_cv_module()
    
    # FlightDelayDataLoader only takes suffix parameter
    data_loader = cv.FlightDelayDataLoader(suffix=input_suffix)
    data_loader.load()
    
    folds_raw = data_loader.get_version(version)
    
    # Convert to our format: (train_df, val_or_test_df, fold_type)
    folds = []
    for fold_idx, (train_df, val_or_test_df) in enumerate(folds_raw, start=1):
        # Determine fold type: last fold is TEST, others are VAL
        fold_type = "TEST" if fold_idx == len(folds_raw) else "VAL"
        folds.append((train_df, val_or_test_df, fold_type))
    
    return folds


def evaluate_meta_model_accuracy(df: DataFrame, target_col: str, pred_col: str, model_id: str):
    """
    Evaluate accuracy of a meta-model prediction.
    
    Returns:
        dict with metrics: rmse, mae, r2, count, null_predictions, null_targets
    """
    # Filter to rows where we have both target and prediction
    df_eval = df.filter(
        col(target_col).isNotNull() & 
        col(pred_col).isNotNull()
    )
    
    count = df_eval.count()
    
    if count == 0:
        return {
            "model_id": model_id,
            "target": target_col,
            "count": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "null_predictions": df.filter(col(pred_col).isNull()).count(),
            "null_targets": df.filter(col(target_col).isNull()).count()
        }
    
    # Calculate metrics
    evaluator_rmse = RegressionEvaluator(
        predictionCol=pred_col,
        labelCol=target_col,
        metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        predictionCol=pred_col,
        labelCol=target_col,
        metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        predictionCol=pred_col,
        labelCol=target_col,
        metricName="r2"
    )
    
    rmse = evaluator_rmse.evaluate(df_eval)
    mae = evaluator_mae.evaluate(df_eval)
    r2 = evaluator_r2.evaluate(df_eval)
    
    # Count nulls
    null_predictions = df.filter(col(pred_col).isNull()).count()
    null_targets = df.filter(col(target_col).isNull()).count()
    
    # Calculate mean absolute percentage error (MAPE) - useful for understanding relative error
    df_with_mape = df_eval.withColumn(
        "abs_pct_error",
        F.abs((col(pred_col) - col(target_col)) / F.greatest(F.abs(col(target_col)), F.lit(1.0))) * F.lit(100.0)
    )
    mape_result = df_with_mape.agg(F.avg("abs_pct_error").alias("mape")).collect()[0]
    mape = mape_result["mape"] if mape_result["mape"] is not None else None
    
    return {
        "model_id": model_id,
        "target": target_col,
        "count": count,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "null_predictions": null_predictions,
        "null_targets": null_targets
    }


def assess_meta_models_for_version(version: str, input_suffix: str):
    """Assess meta-model accuracy for a specific version."""
    print(f"\n{'='*80}")
    print(f"ASSESSING META-MODEL ACCURACY for {version}")
    print(f"{'='*80}")
    print(f"Input suffix: {input_suffix}")
    print(f"Meta-model IDs: {META_MODEL_IDS}")
    
    # Load folds
    folds = load_folds_for_version(version, input_suffix)
    
    if not folds:
        print(f"  ⚠ No folds found for version {version} with suffix '{input_suffix}'")
        return
    
    print(f"  Found {len(folds)} folds")
    
    all_results = []
    
    for fold_idx, (train_df, val_or_test_df, fold_type) in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(folds)} ({fold_type})")
        print(f"{'='*60}")
        
        if VERBOSE:
            train_count = train_df.count()
            val_count = val_or_test_df.count()
            print(f"  Train: {train_count:,} rows")
            print(f"  {fold_type}: {val_count:,} rows")
        
        # Evaluate on both train and validation sets
        for split_name, df in [("TRAIN", train_df), (fold_type, val_or_test_df)]:
            print(f"\n  Evaluating on {split_name} set:")
            
            # Debug: Show available prediction columns
            if VERBOSE:
                pred_cols = [c for c in df.columns if "predicted_prev_flight" in c]
                if pred_cols:
                    print(f"    Available prediction columns: {pred_cols[:5]}..." if len(pred_cols) > 5 else f"    Available prediction columns: {pred_cols}")
            
            for model_id in META_MODEL_IDS:
                for target_col, pred_col_template in TARGETS.items():
                    # Construct prediction column name with model_id
                    pred_col = pred_col_template.replace("XGB_1", model_id)
                    
                    if pred_col not in df.columns:
                        print(f"    ⚠ {pred_col} not found - skipping")
                        continue
                    
                    if target_col not in df.columns:
                        print(f"    ⚠ {target_col} not found - skipping")
                        continue
                    
                    metrics = evaluate_meta_model_accuracy(df, target_col, pred_col, model_id)
                    metrics["version"] = version
                    metrics["fold"] = fold_idx
                    metrics["split"] = split_name
                    all_results.append(metrics)
                    
                    # Print results
                    print(f"    {target_col} ({model_id}):")
                    print(f"      RMSE: {metrics['rmse']:.2f} min" if metrics['rmse'] else "      RMSE: N/A")
                    print(f"      MAE:  {metrics['mae']:.2f} min" if metrics['mae'] else "      MAE: N/A")
                    print(f"      R²:   {metrics['r2']:.3f}" if metrics['r2'] else "      R²: N/A")
                    print(f"      MAPE: {metrics['mape']:.1f}%" if metrics.get('mape') else "      MAPE: N/A")
                    print(f"      Valid rows: {metrics['count']:,}")
                    if metrics['null_predictions'] > 0:
                        print(f"      ⚠ NULL predictions: {metrics['null_predictions']:,}")
                    if metrics['null_targets'] > 0:
                        print(f"      ⚠ NULL targets: {metrics['null_targets']:,}")
    
    # Create summary DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY for {version}")
        print(f"{'='*80}")
        
        # Group by target and model_id, show average metrics
        summary = results_df.groupby(['target', 'model_id']).agg({
            'rmse': 'mean',
            'mae': 'mean',
            'r2': 'mean',
            'mape': 'mean',
            'count': 'sum'
        }).round(3)
        
        print("\nAverage Metrics Across All Folds:")
        print(summary.to_string())
        
        # Show per-fold breakdown
        print("\n\nPer-Fold Breakdown:")
        pivot = results_df.pivot_table(
            index=['fold', 'split'],
            columns='target',
            values='rmse',
            aggfunc='mean'
        )
        print(pivot.to_string())
        
        return results_df
    else:
        print("  ⚠ No results to summarize")
        return None


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    all_version_results = {}
    
    for version in VERSIONS:
        results_df = assess_meta_models_for_version(version, INPUT_SUFFIX)
        if results_df is not None:
            all_version_results[version] = results_df
    
    # Cross-version summary
    if len(all_version_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-VERSION SUMMARY")
        print(f"{'='*80}")
        
        combined_df = pd.concat(all_version_results.values(), ignore_index=True)
        
        summary = combined_df.groupby(['target', 'model_id', 'version']).agg({
            'rmse': 'mean',
            'mae': 'mean',
            'r2': 'mean',
            'mape': 'mean'
        }).round(3)
        
        print("\nAverage Metrics by Version:")
        print(summary.to_string())
    
    print(f"\n{'='*80}")
    print("✅ Assessment complete!")
    print(f"{'='*80}")

