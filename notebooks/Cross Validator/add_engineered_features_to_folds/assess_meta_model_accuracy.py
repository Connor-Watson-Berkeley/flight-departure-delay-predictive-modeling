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
VERSIONS = ["60M"]  # <-- EDIT THIS LIST

SOURCE = "OTPW"  # Change to "CUSTOM" for CUSTOM data, "OTPW" for OTPW data
INPUT_SUFFIX = "_with_graph_and_metamodels"  # Suffix for folds with meta-model predictions

# Meta-model IDs to assess (should match what was used in add_meta_model_features.py)
META_MODEL_IDS = ["XGB_1"]  # <-- EDIT THIS LIST

# Target columns and their corresponding prediction columns
# NOTE: The prediction column for taxi_time is actually named "turnover_time" because
# we're predicting turnover time (current dep - prev arr), not just taxi time
TARGETS = {
    "prev_flight_air_time": "predicted_prev_flight_air_time_XGB_1",
    "prev_flight_taxi_time": "predicted_prev_flight_turnover_time_XGB_1",  # Note: column name is "turnover_time" not "taxi_time"
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


def load_folds_for_version(version: str, input_suffix: str = "", source: str = "OTPW"):
    """Load all available folds for a version using cv.py's FlightDelayDataLoader."""
    cv = load_cv_module()
    
    # Create data loader with suffix and source (use local SOURCE, not cv.py's global)
    data_loader = cv.FlightDelayDataLoader(suffix=input_suffix, source=source)
    data_loader.load()
    
    folds_raw = data_loader.get_version(version)
    
    # Convert to our format: (train_df, val_or_test_df, fold_type)
    folds = []
    for fold_idx, (train_df, val_or_test_df) in enumerate(folds_raw, start=1):
        # Determine fold type: last fold is TEST, others are VAL
        fold_type = "TEST" if fold_idx == len(folds_raw) else "VAL"
        folds.append((train_df, val_or_test_df, fold_type))
    
    return folds


def evaluate_meta_model_accuracy(df: DataFrame, target_col: str, pred_col: str, model_id: str, 
                                  include_stratified: bool = True):
    """
    Evaluate accuracy of a meta-model prediction.
    
    Args:
        df: DataFrame with target and prediction columns
        target_col: Name of target column
        pred_col: Name of prediction column
        model_id: Model identifier
        include_stratified: If True, calculate stratified metrics by duration buckets
    
    Returns:
        dict with metrics: rmse, mae, r2, count, null_predictions, null_targets, and optionally stratified metrics
    """
    # Filter to rows where we have both target and prediction
    df_eval = df.filter(
        col(target_col).isNotNull() & 
        col(pred_col).isNotNull()
    )
    
    count = df_eval.count()
    
    if count == 0:
        result = {
            "model_id": model_id,
            "target": target_col,
            "count": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "null_predictions": df.filter(col(pred_col).isNull()).count(),
            "null_targets": df.filter(col(target_col).isNull()).count()
        }
        if include_stratified:
            # Add empty stratified metrics
            result.update({
                "rmse_small": None, "mae_small": None, "r2_small": None, "count_small": 0,
                "rmse_large": None, "mae_large": None, "r2_large": None, "count_large": 0
            })
        return result
    
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
    
    result = {
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
    
    # Calculate stratified metrics by duration buckets
    if include_stratified:
        # Define duration thresholds based on target type
        very_small_threshold = 90  # 1.5 hours = 90 minutes (for all targets)
        
        if "turnover_time" in target_col.lower() or "taxi_time" in target_col.lower():
            # Turnover time: very small < 1.5h, small 1.5h-4h, large > 4h
            threshold_minutes = 240  # 4 hours
            very_small_label = "<1.5h"
            small_label = "1.5h-4h"
            large_label = ">4h"
        elif "total_duration" in target_col.lower():
            # Total duration (rotation time): very small < 1.5h, small 1.5h-5h, large > 5h
            threshold_minutes = 300  # 5 hours
            very_small_label = "<1.5h"
            small_label = "1.5h-5h"
            large_label = ">5h"
        elif "air_time" in target_col.lower():
            # Air time: very small < 1.5h, small 1.5h-3h, large > 3h
            threshold_minutes = 180  # 3 hours
            very_small_label = "<1.5h"
            small_label = "1.5h-3h"
            large_label = ">3h"
        else:
            # Default: use 4 hours
            threshold_minutes = 240
            very_small_label = "<1.5h"
            small_label = "1.5h-4h"
            large_label = ">4h"
        
        # Split into three duration buckets: very small, small, and large
        df_very_small = df_eval.filter(col(target_col) < very_small_threshold)
        df_small = df_eval.filter((col(target_col) >= very_small_threshold) & (col(target_col) <= threshold_minutes))
        df_large = df_eval.filter(col(target_col) > threshold_minutes)
        
        count_very_small = df_very_small.count()
        count_small = df_small.count()
        count_large = df_large.count()
        
        # Calculate metrics for very small durations
        if count_very_small > 0:
            rmse_very_small = evaluator_rmse.evaluate(df_very_small)
            mae_very_small = evaluator_mae.evaluate(df_very_small)
            r2_very_small = evaluator_r2.evaluate(df_very_small)
        else:
            rmse_very_small = None
            mae_very_small = None
            r2_very_small = None
        
        # Calculate metrics for small durations
        if count_small > 0:
            rmse_small = evaluator_rmse.evaluate(df_small)
            mae_small = evaluator_mae.evaluate(df_small)
            r2_small = evaluator_r2.evaluate(df_small)
        else:
            rmse_small = None
            mae_small = None
            r2_small = None
        
        # Calculate metrics for large durations
        if count_large > 0:
            rmse_large = evaluator_rmse.evaluate(df_large)
            mae_large = evaluator_mae.evaluate(df_large)
            r2_large = evaluator_r2.evaluate(df_large)
        else:
            rmse_large = None
            mae_large = None
            r2_large = None
        
        result.update({
            f"rmse_{very_small_label}": rmse_very_small,
            f"mae_{very_small_label}": mae_very_small,
            f"r2_{very_small_label}": r2_very_small,
            f"count_{very_small_label}": count_very_small,
            f"rmse_{small_label}": rmse_small,
            f"mae_{small_label}": mae_small,
            f"r2_{small_label}": r2_small,
            f"count_{small_label}": count_small,
            f"rmse_{large_label}": rmse_large,
            f"mae_{large_label}": mae_large,
            f"r2_{large_label}": r2_large,
            f"count_{large_label}": count_large,
            "very_small_threshold_minutes": very_small_threshold,
            "threshold_minutes": threshold_minutes
        })
    
    return result


def assess_meta_models_for_version(version: str, input_suffix: str, source: str = "OTPW"):
    """Assess meta-model accuracy for a specific version."""
    print(f"\n{'='*80}")
    print(f"ASSESSING META-MODEL ACCURACY for {version}")
    print(f"{'='*80}")
    print(f"Source: {source}")
    print(f"Input suffix: {input_suffix}")
    print(f"Meta-model IDs: {META_MODEL_IDS}")
    
    # Load folds
    folds = load_folds_for_version(version, input_suffix, source=source)
    
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
                    
                    # Print stratified metrics if available
                    if 'threshold_minutes' in metrics:
                        threshold = metrics['threshold_minutes']
                        very_small_threshold = metrics.get('very_small_threshold_minutes', 90)
                        # Find all bucket labels
                        very_small_label = None
                        small_label = None
                        large_label = None
                        for key in metrics.keys():
                            if key.startswith('rmse_') and key != 'rmse':
                                label = key.replace('rmse_', '')
                                if '<1.5h' in label or '<' in label and '1.5' in label:
                                    very_small_label = label
                                elif '-' in label:  # e.g., "1.5h-4h"
                                    small_label = label
                                elif '>' in label:  # e.g., ">4h"
                                    large_label = label
                        
                        if very_small_label or small_label or large_label:
                            print(f"      Stratified by duration (very small: <{very_small_threshold} min = {very_small_threshold/60:.1f}h, threshold: {threshold} min = {threshold/60:.1f}h):")
                            
                            if very_small_label:
                                print(f"        Very Small ({very_small_label}):")
                                print(f"          RMSE: {metrics.get(f'rmse_{very_small_label}', 'N/A'):.2f} min" if metrics.get(f'rmse_{very_small_label}') else "          RMSE: N/A")
                                print(f"          MAE:  {metrics.get(f'mae_{very_small_label}', 'N/A'):.2f} min" if metrics.get(f'mae_{very_small_label}') else "          MAE: N/A")
                                print(f"          R²:   {metrics.get(f'r2_{very_small_label}', 'N/A'):.3f}" if metrics.get(f'r2_{very_small_label}') is not None else "          R²: N/A")
                                print(f"          Count: {metrics.get(f'count_{very_small_label}', 0):,}")
                            
                            if small_label:
                                print(f"        Small ({small_label}):")
                                print(f"          RMSE: {metrics.get(f'rmse_{small_label}', 'N/A'):.2f} min" if metrics.get(f'rmse_{small_label}') else "          RMSE: N/A")
                                print(f"          MAE:  {metrics.get(f'mae_{small_label}', 'N/A'):.2f} min" if metrics.get(f'mae_{small_label}') else "          MAE: N/A")
                                print(f"          R²:   {metrics.get(f'r2_{small_label}', 'N/A'):.3f}" if metrics.get(f'r2_{small_label}') is not None else "          R²: N/A")
                                print(f"          Count: {metrics.get(f'count_{small_label}', 0):,}")
                            
                            if large_label:
                                print(f"        Large ({large_label}):")
                                print(f"          RMSE: {metrics.get(f'rmse_{large_label}', 'N/A'):.2f} min" if metrics.get(f'rmse_{large_label}') else "          RMSE: N/A")
                                print(f"          MAE:  {metrics.get(f'mae_{large_label}', 'N/A'):.2f} min" if metrics.get(f'mae_{large_label}') else "          MAE: N/A")
                                print(f"          R²:   {metrics.get(f'r2_{large_label}', 'N/A'):.3f}" if metrics.get(f'r2_{large_label}') is not None else "          R²: N/A")
                                print(f"          Count: {metrics.get(f'count_{large_label}', 0):,}")
                    
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
        results_df = assess_meta_models_for_version(version, INPUT_SUFFIX, source=SOURCE)
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