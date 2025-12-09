#!/usr/bin/env python3
"""
validate_precomputed_features.py - Validate precomputed graph and meta-model features

Checks:
1. Table structure - ensures new fields are present
2. Null/zero values - checks for missing or zero predictions
3. Meta-model error metrics - evaluates RMSE, MAE for meta-model predictions
"""

import sys
import importlib.util
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator

# Import cv module for data loading
cv_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/cv.py"
spec = importlib.util.spec_from_file_location("cv", cv_path)
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def validate_graph_features(df, fold_name):
    """Validate graph features are present and not all zeros/null."""
    print(f"\n{'='*60}")
    print(f"Validating Graph Features: {fold_name}")
    print(f"{'='*60}")
    
    required_graph_features = [
        "origin_pagerank_weighted",
        "origin_pagerank_unweighted",
        "dest_pagerank_weighted",
        "dest_pagerank_unweighted"
    ]
    
    # Check for prev_flight graph features (optional, for meta models)
    optional_graph_features = [
        "prev_flight_origin_pagerank_weighted",
        "prev_flight_origin_pagerank_unweighted",
        "prev_flight_dest_pagerank_weighted",
        "prev_flight_dest_pagerank_unweighted"
    ]
    
    all_graph_features = required_graph_features + optional_graph_features
    
    # Check presence
    available_features = [f for f in all_graph_features if f in df.columns]
    missing_features = [f for f in required_graph_features if f not in df.columns]
    
    print(f"\n✓ Graph Features Present: {len(available_features)}/{len(all_graph_features)}")
    if missing_features:
        print(f"  ⚠ Missing required features: {missing_features}")
    else:
        print(f"  ✓ All required features present")
    
    # Check for nulls and zeros
    total_rows = df.count()
    print(f"\n  Total rows: {total_rows:,}")
    
    for feature in available_features:
        null_count = df.filter(col(feature).isNull()).count()
        zero_count = df.filter(col(feature) == 0.0).count()
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        zero_pct = (zero_count / total_rows * 100) if total_rows > 0 else 0
        
        print(f"\n  {feature}:")
        print(f"    Nulls: {null_count:,} ({null_pct:.2f}%)")
        print(f"    Zeros: {zero_count:,} ({zero_pct:.2f}%)")
        
        if null_pct > 1.0:
            print(f"    ⚠ WARNING: High null percentage!")
        if zero_pct > 10.0:
            print(f"    ⚠ WARNING: High zero percentage (may indicate missing airports)")
    
    return len(missing_features) == 0


def validate_meta_model_predictions(df, fold_name, model_ids):
    """Validate meta-model predictions and compute error metrics."""
    print(f"\n{'='*60}")
    print(f"Validating Meta-Model Predictions: {fold_name}")
    print(f"{'='*60}")
    
    prediction_cols = [
        "predicted_prev_flight_air_time",
        "predicted_prev_flight_taxi_time",
        "predicted_prev_flight_total_duration"
    ]
    
    # Check for predictions with model_id suffix
    all_pred_cols = []
    for model_id in model_ids:
        for pred_col in prediction_cols:
            full_col = f"{pred_col}_{model_id}"
            if full_col in df.columns:
                all_pred_cols.append((full_col, pred_col, model_id))
    
    print(f"\n✓ Meta-Model Predictions Found: {len(all_pred_cols)} columns")
    for full_col, base_col, model_id in all_pred_cols:
        print(f"  - {full_col} ({model_id})")
    
    if len(all_pred_cols) == 0:
        print(f"  ⚠ WARNING: No meta-model predictions found!")
        return False
    
    # Check for nulls and zeros
    total_rows = df.count()
    print(f"\n  Total rows: {total_rows:,}")
    
    results = {}
    
    for full_col, base_col, model_id in all_pred_cols:
        print(f"\n  {full_col} ({model_id}):")
        
        null_count = df.filter(col(full_col).isNull()).count()
        zero_count = df.filter(col(full_col) == 0.0).count()
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        zero_pct = (zero_count / total_rows * 100) if total_rows > 0 else 0
        
        print(f"    Nulls: {null_count:,} ({null_pct:.2f}%)")
        print(f"    Zeros: {zero_count:,} ({zero_pct:.2f}%)")
        
        if null_pct > 0:
            print(f"    ⚠ WARNING: Null values found (should be imputed)!")
        if zero_pct == 100:
            print(f"    ⚠ ERROR: All predictions are zero!")
        
        # Get actual target column name
        if base_col == "predicted_prev_flight_air_time":
            target_col = "prev_flight_air_time"
        elif base_col == "predicted_prev_flight_taxi_time":
            target_col = "prev_flight_taxi_time"
        elif base_col == "predicted_prev_flight_total_duration":
            target_col = "prev_flight_actual_elapsed_time"
        else:
            target_col = None
        
        # Compute error metrics if target exists
        if target_col and target_col in df.columns:
            # Filter to rows with both prediction and target
            valid_df = df.filter(
                col(full_col).isNotNull() & 
                col(target_col).isNotNull()
            )
            valid_count = valid_df.count()
            
            if valid_count > 0:
                # Compute RMSE and MAE
                rmse_evaluator = RegressionEvaluator(
                    predictionCol=full_col,
                    labelCol=target_col,
                    metricName="rmse"
                )
                mae_evaluator = RegressionEvaluator(
                    predictionCol=full_col,
                    labelCol=target_col,
                    metricName="mae"
                )
                
                rmse = rmse_evaluator.evaluate(valid_df)
                mae = mae_evaluator.evaluate(valid_df)
                
                # Compute mean absolute percentage error
                mape_df = valid_df.withColumn(
                    "abs_error",
                    F.abs(col(full_col) - col(target_col))
                ).withColumn(
                    "pct_error",
                    F.when(col(target_col) != 0, 
                           col("abs_error") / F.abs(col(target_col)) * 100)
                    .otherwise(F.when(col("abs_error") > 0, 999.0).otherwise(0.0))
                )
                mape = mape_df.select(F.avg("pct_error")).first()[0] or 0.0
                
                print(f"    ✓ Error Metrics (on {valid_count:,} valid rows):")
                print(f"      RMSE: {rmse:.2f}")
                print(f"      MAE:  {mae:.2f}")
                print(f"      MAPE: {mape:.2f}%")
                
                results[f"{base_col}_{model_id}"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "valid_rows": valid_count
                }
            else:
                print(f"    ⚠ No valid rows for error computation")
        else:
            print(f"    ⚠ Target column '{target_col}' not found for error computation")
    
    return results


def validate_fold(train_df, val_df, fold_num, model_ids):
    """Validate a single fold."""
    print(f"\n{'#'*60}")
    print(f"# VALIDATING FOLD {fold_num}")
    print(f"{'#'*60}")
    
    # Validate train
    train_graph_ok = validate_graph_features(train_df, f"Fold {fold_num} - TRAIN")
    train_meta_results = validate_meta_model_predictions(train_df, f"Fold {fold_num} - TRAIN", model_ids)
    
    # Validate val
    val_graph_ok = validate_graph_features(val_df, f"Fold {fold_num} - VAL")
    val_meta_results = validate_meta_model_predictions(val_df, f"Fold {fold_num} - VAL", model_ids)
    
    return {
        "fold": fold_num,
        "train_graph_ok": train_graph_ok,
        "val_graph_ok": val_graph_ok,
        "train_meta_results": train_meta_results,
        "val_meta_results": val_meta_results
    }


def main():
    """Main validation function."""
    spark = SparkSession.builder.getOrCreate()
    
    # Configuration
    VERSION = "3M"  # Change to "12M" or "60M" as needed
    MODEL_IDS = ["RF_1"]  # Update if using multiple models
    
    print("="*60)
    print("VALIDATING PRECOMPUTED FEATURES")
    print("="*60)
    print(f"Version: {VERSION}")
    print(f"Model IDs: {MODEL_IDS}")
    
    # Load data
    print(f"\n📥 Loading folds for {VERSION}...")
    loader = cv.FlightDelayDataLoader()
    loader.load()
    folds = loader.get_version(VERSION)
    
    print(f"✓ Loaded {len(folds)} folds")
    
    # Validate each fold
    all_results = []
    for fold_idx, (train_df, val_df) in enumerate(folds, start=1):
        result = validate_fold(train_df, val_df, fold_idx, MODEL_IDS)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    all_graph_ok = all(r["train_graph_ok"] and r["val_graph_ok"] for r in all_results)
    print(f"\n✓ Graph Features: {'PASS' if all_graph_ok else 'FAIL'}")
    
    # Meta-model summary
    print(f"\n✓ Meta-Model Predictions:")
    for result in all_results:
        fold = result["fold"]
        train_meta = result["train_meta_results"]
        val_meta = result["val_meta_results"]
        
        if train_meta or val_meta:
            print(f"\n  Fold {fold}:")
            if val_meta:
                print(f"    Validation Set Metrics:")
                for col_name, metrics in val_meta.items():
                    print(f"      {col_name}:")
                    print(f"        RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2f}%")
    
    print(f"\n{'='*60}")
    print("Validation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()