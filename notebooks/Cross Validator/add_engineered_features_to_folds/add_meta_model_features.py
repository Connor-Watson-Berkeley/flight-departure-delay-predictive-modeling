"""
add_meta_model_features.py - Add meta-model predictions to existing folds

This notebook reads folds with graph features (from add_graph_features.py) and adds
meta-model predictions for previous flight components, saving to a new path with 
suffix _with_graph_and_metamodels.

Usage:
    Set VERSIONS list below, then run all cells.
"""

from __future__ import annotations
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, coalesce, lit
from pyspark.ml.base import Estimator, Model
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from datetime import datetime
import importlib.util
import sys
import time
import logging

# XGBoost support (much faster than Random Forest - 5-10x speedup)
from xgboost.spark import SparkXGBRegressor

# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M", "XM"])
VERSIONS = ["60M"]  # <-- EDIT THIS LIST

INPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "OTPW"  # Change to "CUSTOM" for CUSTOM data, "OTPW" for OTPW data
WRITE_MODE = "overwrite"
VERBOSE = True
OUTPUT_SUFFIX = "_with_graph_and_metamodels"
INPUT_SUFFIX = "_with_graph"  # Input suffix: '_with_graph' (base) or '_with_graph_and_metamodels' (to add more models)
SKIP_EXISTING_FOLDS = True  # If True, skip folds where all 3 meta-model predictions already exist 
                             # AND have >97% non-null values (catches bugs where columns were created but not populated).
                             # If any are missing or have too many NULLs, the fold will be re-processed to add/fix the missing models.

# Retry configuration for robust handling of cluster failures
MAX_RETRIES = 10  # Maximum number of retry attempts
INITIAL_RETRY_DELAY = 60  # Initial delay in seconds (exponential backoff)
RETRYABLE_ERRORS = [
    "ExecutorLostFailure",
    "worker lost",
    "heartbeat timeout",
    "barrier ResultStage",
    "Stage failed",
    "Could not recover",
    "BarrierJobSlotsNumberCheckFailed",  # XGBoost requesting more workers than available
    "does not allow run a barrier stage that requires more slots",
    "SPARK_JOB_CANCELLED",  # Job cancelled (often due to timeout or resource constraints - retryable)
    "Job.*cancelled",  # Pattern match for job cancellation
    "cancelled as part of cancellation"  # Job cancellation message
]

# NOTE: To add a new meta-model version (e.g., XGB_2) without losing existing ones:
#   1. Set INPUT_SUFFIX = "_with_graph_and_metamodels" (reads existing meta-model predictions)
#   2. Add new model to META_MODEL_IDS (e.g., "XGB_2": {...})
#   3. Run script - it will preserve existing _XGB_1 columns and add new _XGB_2 columns

# Performance optimization: Sample training data for very large datasets (60M)
# Set to None to use all data, or a fraction (e.g., 0.5 = 50% of training data)
# NOTE: Only applies to meta-model training, not inference
TRAINING_SAMPLE_FRACTION = None  # None = use all data, 0.5 = 50% sample, etc.

# Meta-model configuration
# Using XGBoost for meta-models (5-10x faster than Random Forest)
# Hyperparameters optimized for speed while maintaining good accuracy:
# - num_boost_round=75: Faster than 100, still good accuracy for meta-models
# - max_depth=6: Good balance (default is 6)
# - learning_rate=0.15: Slightly higher for faster convergence
# - min_child_weight=1: Default, prevents overfitting
META_MODEL_IDS = {
    "XGB_1": {
        "description": "XGBoost v1 (75 rounds, max_depth=6, learning_rate=0.15) - Optimized for speed",
        "model_type": "xgboost",
        "num_boost_round": 75,  # Reduced from 100 for faster training (meta-models don't need perfection)
        "max_depth": 6,  # Good default, prevents overfitting
        "learning_rate": 0.15,  # Slightly higher for faster convergence
        "min_child_weight": 1,  # Default, prevents overfitting
        "use_preprocessed_features": False
    }
}


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_cv_module():
    """Load the cv module using importlib (for Databricks compatibility)."""
    cv_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/cv.py"
    # Try Databricks path first, fall back to local path if needed
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
    """
    Load all available folds for a version using cv.py's FlightDelayDataLoader.
    Returns list of (train_df, val_or_test_df, fold_type) tuples, where fold_type is either
    "VAL" (for CV folds) or "TEST" (for test fold).
    
    Note: Handles schema mismatches by using mergeSchema option when reading parquet.
    """
    cv = load_cv_module()
    
    # Create data loader with suffix and source (use local SOURCE, not cv.py's global)
    data_loader = cv.FlightDelayDataLoader(suffix=input_suffix, source=SOURCE)
    
    # Temporarily patch _load_parquet to use mergeSchema for schema evolution
    original_load_parquet = data_loader._load_parquet
    def _load_parquet_with_merge_schema(name):
        spark = SparkSession.builder.getOrCreate()
        # Use mergeSchema to handle type mismatches (e.g., INT64 vs double)
        df = spark.read.option("mergeSchema", "true").parquet(f"{cv.FOLDER_PATH}/{name}.parquet")
        df = data_loader._cast_numerics(df)
        return df
    
    # Patch the method
    data_loader._load_parquet = _load_parquet_with_merge_schema
    
    try:
        # Load folds for this specific version (uses suffix automatically)
        folds_raw = data_loader._load_version(version)
    finally:
        # Restore original method
        data_loader._load_parquet = original_load_parquet
    
    # Convert to our format: (train_df, val_or_test_df, fold_type)
    folds = []
    for fold_idx, (train_df, val_or_test_df) in enumerate(folds_raw, start=1):
        # Determine fold type: last fold is TEST, others are VAL
        fold_type = "TEST" if fold_idx == len(folds_raw) else "VAL"
        folds.append((train_df, val_or_test_df, fold_type))
        
        if VERBOSE:
            base_name = f"OTPW_{SOURCE}_{version}{input_suffix}"
            if fold_type == "VAL":
                print(f"  ✓ Loaded fold {fold_idx}: {base_name}_FOLD_{fold_idx}_TRAIN + {base_name}_FOLD_{fold_idx}_VAL")
            else:
                print(f"  ✓ Loaded fold {fold_idx}: {base_name}_FOLD_{fold_idx}_TRAIN + {base_name}_FOLD_{fold_idx}_TEST")
    
    if VERBOSE:
        expected_folds = 4  # 3 CV + 1 test
        if len(folds) != expected_folds:
            print(f"  ⚠ WARNING: Found {len(folds)} folds, expected {expected_folds} (3 CV + 1 test)")
        else:
            print(f"  ✓ Found {len(folds)} folds as expected")
    
    return folds


def fold_exists(version: str, fold_idx: int, output_suffix: str, fold_type: str,
                expected_model_ids: list = None):
    """
    Check if a fold already exists in the output folder AND has all expected meta-model predictions.
    
    Args:
        version: Data version (e.g., "12M")
        fold_idx: Fold index (1-4)
        output_suffix: Output suffix (e.g., "_with_graph_and_metamodels")
        fold_type: "VAL" or "TEST"
        expected_model_ids: List of model IDs that should have predictions (e.g., ["XGB_1"])
    
    Returns:
        True if fold exists AND has all expected prediction columns with >97% non-null values, False otherwise
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    spark = SparkSession.builder.getOrCreate()
    
    base_name = f"OTPW_{SOURCE}_{version}{output_suffix}"
    train_name = f"{base_name}_FOLD_{fold_idx}_TRAIN"
    train_path = f"{OUTPUT_FOLDER}/{train_name}.parquet"
    
    if fold_type == "VAL":
        val_name = f"{base_name}_FOLD_{fold_idx}_VAL"
        val_path = f"{OUTPUT_FOLDER}/{val_name}.parquet"
    else:
        test_name = f"{base_name}_FOLD_{fold_idx}_TEST"
        val_path = f"{OUTPUT_FOLDER}/{test_name}.parquet"
    
    # Check if both train and val/test files exist
    # Use Spark's file system to check existence (faster than trying to read)
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark.sparkContext._jvm.java.net.URI(train_path), hadoop_conf
        )
        train_exists = fs.exists(spark.sparkContext._jvm.org.apache.hadoop.fs.Path(train_path))
        val_exists = fs.exists(spark.sparkContext._jvm.org.apache.hadoop.fs.Path(val_path))
        
        if not (train_exists and val_exists):
            return False
        
        # If expected_model_ids provided, check for all expected prediction columns
        if expected_model_ids:
            try:
                # Read DataFrames to check columns and non-null percentages
                # Use mergeSchema=True to handle schema evolution, and allow type mismatches
                train_df = spark.read.option("mergeSchema", "true").parquet(train_path)
                val_df = spark.read.option("mergeSchema", "true").parquet(val_path)
                train_cols = set(train_df.columns)
                val_cols = set(val_df.columns)
                
                # Expected prediction columns for each model ID
                expected_cols = []
                for model_id in expected_model_ids:
                    expected_cols.extend([
                        f"predicted_prev_flight_air_time_{model_id}",
                        f"predicted_prev_flight_turnover_time_{model_id}",  # Note: This is turnover time, not raw taxi time
                        f"predicted_prev_flight_total_duration_{model_id}"
                    ])
                
                # Check if all expected columns exist in both train and val
                missing_cols_train = [col for col in expected_cols if col not in train_cols]
                missing_cols_val = [col for col in expected_cols if col not in val_cols]
                
                if missing_cols_train or missing_cols_val:
                    # Fold exists but is incomplete - return False so it gets reprocessed
                    return False
                
                # Check non-null percentage for each column (must be >97% non-null)
                # This catches bugs where columns were created but not populated
                MIN_NON_NULL_PERCENTAGE = 0.97  # 97% threshold
                
                train_total = train_df.count()
                val_total = val_df.count()
                
                # Check each expected column
                for col_name in expected_cols:
                    # Check train DataFrame
                    train_non_null = train_df.filter(col(col_name).isNotNull()).count()
                    train_non_null_pct = train_non_null / train_total if train_total > 0 else 0.0
                    
                    if train_non_null_pct < MIN_NON_NULL_PERCENTAGE:
                        # Column exists but has too many NULLs - fold needs reprocessing
                        return False
                    
                    # Check val/test DataFrame
                    val_non_null = val_df.filter(col(col_name).isNotNull()).count()
                    val_non_null_pct = val_non_null / val_total if val_total > 0 else 0.0
                    
                    if val_non_null_pct < MIN_NON_NULL_PERCENTAGE:
                        # Column exists but has too many NULLs - fold needs reprocessing
                        return False
                
                # All columns exist and have >97% non-null values
                return True
            except Exception:
                # If we can't read or check the data, assume incomplete
                return False
        
        return True
    except Exception as e:
        # Fallback: try to read schema (slower but more reliable)
        try:
            # Use mergeSchema=True to handle schema evolution
            train_df = spark.read.option("mergeSchema", "true").parquet(train_path)
            val_df = spark.read.option("mergeSchema", "true").parquet(val_path)
            _ = train_df.schema
            _ = val_df.schema
            return True
        except:
            return False


def save_fold_with_suffix(version: str, fold_idx: int, train_df: DataFrame, val_or_test_df: DataFrame, 
                          output_suffix: str, fold_type: str):
    """Save fold with output suffix."""
    base_name = f"OTPW_{SOURCE}_{version}{output_suffix}"
    
    train_name = f"{base_name}_FOLD_{fold_idx}_TRAIN"
    train_path = f"{OUTPUT_FOLDER}/{train_name}.parquet"
    train_df.write.mode(WRITE_MODE).parquet(train_path)
    
    if fold_type == "VAL":
        val_name = f"{base_name}_FOLD_{fold_idx}_VAL"
        val_path = f"{OUTPUT_FOLDER}/{val_name}.parquet"
        val_or_test_df.write.mode(WRITE_MODE).parquet(val_path)
    else:
        test_name = f"{base_name}_FOLD_{fold_idx}_TEST"
        val_path = f"{OUTPUT_FOLDER}/{test_name}.parquet"
        val_or_test_df.write.mode(WRITE_MODE).parquet(val_path)


# -------------------------
# META-MODEL LOGIC (simplified, inlined)
# -------------------------
def compute_meta_model_predictions(train_df, val_df, model_id="RF_1", verbose=True):
    """
    Train meta-models on training data and apply predictions to both train and val DataFrames.
    
    Uses shared preprocessing pipeline for efficiency - all three models use the same features,
    so we preprocess once and reuse for all models.
    
    Returns:
        tuple: (train_df_with_predictions, val_df_with_predictions) both with predicted columns
    """
    if model_id not in META_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}. Available: {list(META_MODEL_IDS.keys())}")
    
    model_config = META_MODEL_IDS[model_id]
    
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing meta-model predictions ({model_id})...")
        print(f"  Config: {model_config['description']}")
    
    train_with_meta = train_df
    val_with_meta = val_df
    
    # Check which prediction columns already exist (for incremental training)
    expected_pred_cols = {
        "prev_flight_air_time": f"predicted_prev_flight_air_time_{model_id}",
        "prev_flight_taxi_time": f"predicted_prev_flight_turnover_time_{model_id}",
        "prev_flight_total_duration": f"predicted_prev_flight_total_duration_{model_id}"
    }
    
    existing_pred_cols = {}
    for target, pred_col in expected_pred_cols.items():
        if pred_col in train_with_meta.columns and pred_col in val_with_meta.columns:
            existing_pred_cols[target] = pred_col
    
    if verbose and existing_pred_cols:
        print(f"\n  🔍 Found {len(existing_pred_cols)}/{len(expected_pred_cols)} existing prediction columns:")
        for target, pred_col in existing_pred_cols.items():
            print(f"    ✓ {pred_col} (will skip training)")
        missing = [target for target in expected_pred_cols.keys() if target not in existing_pred_cols]
        if missing:
            print(f"  Will train models for: {', '.join(missing)}")
    
    # Add special case flags
    if "is_first_flight" not in train_with_meta.columns:
        train_with_meta = train_with_meta.withColumn("is_first_flight", (col("lineage_rank") == 1).cast("int"))
        val_with_meta = val_with_meta.withColumn("is_first_flight", (col("lineage_rank") == 1).cast("int"))
    if "is_jump" not in train_with_meta.columns:
        train_with_meta = train_with_meta.withColumn("is_jump", col("lineage_is_jump").cast("int"))
        val_with_meta = val_with_meta.withColumn("is_jump", col("lineage_is_jump").cast("int"))
    
    # Compute derived target columns if needed
    # Note: These are derived from available columns if the target columns don't exist
    # This allows the script to work with datasets that may not have all flight lineage features
    
    # Derive prev_flight_air_time from prev_flight_crs_elapsed_time if needed
    if "prev_flight_air_time" not in train_with_meta.columns:
        if "prev_flight_crs_elapsed_time" in train_with_meta.columns:
            train_with_meta = train_with_meta.withColumn(
                "prev_flight_air_time", col("prev_flight_crs_elapsed_time")
            )
            val_with_meta = val_with_meta.withColumn(
                "prev_flight_air_time", col("prev_flight_crs_elapsed_time")
            )
            if verbose:
                print(f"  ✓ Derived prev_flight_air_time from prev_flight_crs_elapsed_time")
    else:
        if verbose:
            print(f"  🔍✓ Using existing prev_flight_air_time column")
    
    # Derive prev_flight_taxi_time from turnover time (current departure - previous arrival)
    # This is the ground time between previous flight arrival and current flight departure
    # ALWAYS use turnover time, not raw taxi_in + taxi_out
    # Priority: existing prev_flight_taxi_time > lineage_actual_turnover_time_minutes > scheduled_lineage_turnover_time_minutes > taxi_in + taxi_out
    if "prev_flight_taxi_time" in train_with_meta.columns:
        # Use existing prev_flight_taxi_time if it already exists (preferred - already computed)
        if verbose:
            print(f"  🔍✓ Using existing prev_flight_taxi_time column")
    elif "lineage_actual_turnover_time_minutes" in train_with_meta.columns:
        # Use actual turnover time (fallback - uses actual times)
        train_with_meta = train_with_meta.withColumn(
            "prev_flight_taxi_time",
            col("lineage_actual_turnover_time_minutes")
        )
        val_with_meta = val_with_meta.withColumn(
            "prev_flight_taxi_time",
            col("lineage_actual_turnover_time_minutes")
        )
        if verbose:
            # Verify the column was actually created
            has_col = "prev_flight_taxi_time" in train_with_meta.columns
            print(f"  ✓ Derived prev_flight_taxi_time from lineage_actual_turnover_time_minutes (verified: {has_col})")
    elif "scheduled_lineage_turnover_time_minutes" in train_with_meta.columns:
        # Use scheduled turnover time (fallback - uses scheduled times)
        train_with_meta = train_with_meta.withColumn(
            "prev_flight_taxi_time",
            col("scheduled_lineage_turnover_time_minutes")
        )
        val_with_meta = val_with_meta.withColumn(
            "prev_flight_taxi_time",
            col("scheduled_lineage_turnover_time_minutes")
        )
        if verbose:
            print(f"  ✓ Derived prev_flight_taxi_time from scheduled_lineage_turnover_time_minutes")
    else:
        # Last resort: use raw taxi_in + taxi_out (not ideal, but better than nothing)
        # This should rarely happen if flight lineage features are properly applied
        if "prev_flight_taxi_in" in train_with_meta.columns or "prev_flight_taxi_out" in train_with_meta.columns:
            train_with_meta = train_with_meta.withColumn(
                "prev_flight_taxi_time",
                coalesce(col("prev_flight_taxi_in"), lit(0.0)) + 
                coalesce(col("prev_flight_taxi_out"), lit(0.0))
            )
            val_with_meta = val_with_meta.withColumn(
                "prev_flight_taxi_time",
                coalesce(col("prev_flight_taxi_in"), lit(0.0)) + 
                coalesce(col("prev_flight_taxi_out"), lit(0.0))
            )
            if verbose:
                print(f"  ⚠ Derived prev_flight_taxi_time from prev_flight_taxi_in + prev_flight_taxi_out (fallback)")
    
    # Derive prev_flight_total_duration from actual rotation time (current actual dep - prev actual dep)
    # This is the total time from previous flight's actual departure to current flight's actual departure
    # Rotation time = Air Time + Turnover Time (aviation terminology)
    if "prev_flight_total_duration" not in train_with_meta.columns:
        # First, ensure we have the time components in minutes
        from pyspark.sql.functions import when, floor, lit
        
        # Compute actual_dep_time_minutes if not already present
        if "actual_dep_time_minutes" not in train_with_meta.columns:
            train_with_meta = train_with_meta.withColumn(
                "actual_dep_time_minutes",
                when(
                    col("dep_time").isNotNull(),
                    floor(col("dep_time") / 100) * 60 + (col("dep_time") % 100)
                ).otherwise(None)
            )
            val_with_meta = val_with_meta.withColumn(
                "actual_dep_time_minutes",
                when(
                    col("dep_time").isNotNull(),
                    floor(col("dep_time") / 100) * 60 + (col("dep_time") % 100)
                ).otherwise(None)
            )
        
        # Compute prev_flight_actual_dep_time_minutes if not already present
        if "prev_flight_actual_dep_time_minutes" not in train_with_meta.columns:
            if "prev_flight_actual_dep_time" in train_with_meta.columns:
                train_with_meta = train_with_meta.withColumn(
                    "prev_flight_actual_dep_time_minutes",
                    when(
                        col("prev_flight_actual_dep_time").isNotNull(),
                        floor(col("prev_flight_actual_dep_time") / 100) * 60 + (col("prev_flight_actual_dep_time") % 100)
                    ).otherwise(None)
                )
                val_with_meta = val_with_meta.withColumn(
                    "prev_flight_actual_dep_time_minutes",
                    when(
                        col("prev_flight_actual_dep_time").isNotNull(),
                        floor(col("prev_flight_actual_dep_time") / 100) * 60 + (col("prev_flight_actual_dep_time") % 100)
                    ).otherwise(None)
                )
        
        # Compute actual rotation time = current actual dep - previous actual dep
        # This is the total duration from previous departure to current departure
        if "actual_dep_time_minutes" in train_with_meta.columns and "prev_flight_actual_dep_time_minutes" in train_with_meta.columns:
            train_with_meta = train_with_meta.withColumn(
                "prev_flight_total_duration",
                when(
                    (col("actual_dep_time_minutes").isNotNull()) & (col("prev_flight_actual_dep_time_minutes").isNotNull()),
                    when(
                        col("actual_dep_time_minutes") >= col("prev_flight_actual_dep_time_minutes"),
                        col("actual_dep_time_minutes") - col("prev_flight_actual_dep_time_minutes")
                    ).otherwise(col("actual_dep_time_minutes") + 1440 - col("prev_flight_actual_dep_time_minutes"))
                ).otherwise(None)
            )
            val_with_meta = val_with_meta.withColumn(
                "prev_flight_total_duration",
                when(
                    (col("actual_dep_time_minutes").isNotNull()) & (col("prev_flight_actual_dep_time_minutes").isNotNull()),
                    when(
                        col("actual_dep_time_minutes") >= col("prev_flight_actual_dep_time_minutes"),
                        col("actual_dep_time_minutes") - col("prev_flight_actual_dep_time_minutes")
                    ).otherwise(col("actual_dep_time_minutes") + 1440 - col("prev_flight_actual_dep_time_minutes"))
                ).otherwise(None)
            )
            if verbose:
                print(f"  ✓ Derived prev_flight_total_duration from actual rotation time (current actual dep - prev actual dep)")
        elif "prev_flight_actual_elapsed_time" in train_with_meta.columns:
            # Fallback to previous flight's elapsed time if rotation time components not available
            train_with_meta = train_with_meta.withColumn(
                "prev_flight_total_duration", col("prev_flight_actual_elapsed_time")
            )
            val_with_meta = val_with_meta.withColumn(
                "prev_flight_total_duration", col("prev_flight_actual_elapsed_time")
            )
            if verbose:
                print(f"  ⚠ Derived prev_flight_total_duration from prev_flight_actual_elapsed_time (fallback - not ideal)")
    else:
        if verbose:
            print(f"  🔍✓ Using existing prev_flight_total_duration column")
    
    # Get shared feature lists (all models use the same features)
    categorical_features, numerical_features = _get_shared_features(train_with_meta)
    
    # CRITICAL: Filter out categorical features that are all NULL
    # StringIndexer cannot handle columns where all values are NULL
    valid_categorical_features = []
    invalid_categorical_features = []
    
    for cat_col in categorical_features:
        if cat_col in train_with_meta.columns:
            # Check if column has any non-null values
            non_null_count = train_with_meta.filter(col(cat_col).isNotNull()).count()
            total_count = train_with_meta.count()
            
            if non_null_count > 0:
                valid_categorical_features.append(cat_col)
            else:
                invalid_categorical_features.append(cat_col)
                if verbose:
                    print(f"  ⚠ Excluding categorical feature '{cat_col}' (all values are NULL)")
        else:
            invalid_categorical_features.append(cat_col)
            if verbose:
                print(f"  ⚠ Excluding categorical feature '{cat_col}' (column not found)")
    
    categorical_features = valid_categorical_features
    
    if verbose:
        print(f"  Shared features: {len(categorical_features)} categorical, {len(numerical_features)} numerical")
        if invalid_categorical_features:
            print(f"  Excluded {len(invalid_categorical_features)} categorical features (all NULL or missing)")
    
    # CRITICAL: Cast numerical features to numeric types before preprocessing
    # Some features (especially weather features) may be loaded as strings, but Imputer requires numeric types
    NULL_PAT = r'^(NA|N/A|NULL|null|None|none|\\N|\\s*|\\.|M|T)$'
    
    # Get current column types
    train_dtypes = dict(train_with_meta.dtypes)
    val_dtypes = dict(val_with_meta.dtypes)
    
    features_to_cast = []
    for col_name in numerical_features:
        if col_name in train_with_meta.columns:
            current_type = train_dtypes[col_name]
            # Check if column is string type (needs casting)
            if current_type in ['string', 'StringType']:
                features_to_cast.append(col_name)
    
    if features_to_cast:
        if verbose:
            print(f"  Casting {len(features_to_cast)} numerical features from string to double:")
            for col_name in features_to_cast[:5]:  # Show first 5
                print(f"    - {col_name}")
            if len(features_to_cast) > 5:
                print(f"    ... and {len(features_to_cast) - 5} more")
        
        # Cast all string numerical features to double
        for col_name in features_to_cast:
            # Cast to double, handling NULL patterns
            train_with_meta = train_with_meta.withColumn(
                col_name,
                F.regexp_replace(F.col(col_name).cast("string"), NULL_PAT, "")
                .cast("double")
            )
            val_with_meta = val_with_meta.withColumn(
                col_name,
                F.regexp_replace(F.col(col_name).cast("string"), NULL_PAT, "")
                .cast("double")
            )
    
    if verbose:
        print(f"  Building shared preprocessing pipeline...")
    
    # Add stable row identifiers BEFORE preprocessing (for joining predictions back)
    # Use monotonically_increasing_id() which is stable within a single DataFrame
    from pyspark.sql.functions import monotonically_increasing_id
    
    train_with_meta = train_with_meta.withColumn("_row_id", monotonically_increasing_id())
    val_with_meta = val_with_meta.withColumn("_row_id", monotonically_increasing_id())
    
    # Build shared preprocessing pipeline (fit once, use for all models)
    preprocessing_pipeline = _build_shared_preprocessing_pipeline(categorical_features, numerical_features)
    
    # Fit preprocessing on training data (with retry for cluster failures)
    if verbose:
        print("  Fitting preprocessing pipeline (this may take a moment on large datasets)...")
    
    def _fit_preprocessing():
        return preprocessing_pipeline.fit(train_with_meta)
    
    fitted_preprocessing = _retry_with_backoff(_fit_preprocessing, verbose=verbose)
    
    # Apply preprocessing to both train and val once (with retry)
    if verbose:
        print("  Applying preprocessing to train and validation sets...")
    
    def _transform_train():
        return fitted_preprocessing.transform(train_with_meta)
    
    def _transform_val():
        return fitted_preprocessing.transform(val_with_meta)
    
    train_preprocessed = _retry_with_backoff(_transform_train, verbose=verbose)
    val_preprocessed = _retry_with_backoff(_transform_val, verbose=verbose)
    
    # Verify row_id is preserved through preprocessing
    if "_row_id" not in train_preprocessed.columns:
        raise RuntimeError("CRITICAL: _row_id was dropped during preprocessing! Cannot join predictions back.")
    
    # IMPORTANT: Verify target columns are preserved through preprocessing
    # Spark ML transformers should preserve all columns not in their inputCols, but let's verify
    target_columns = ["prev_flight_air_time", "prev_flight_taxi_time", "prev_flight_total_duration"]
    missing_in_preprocessed = []
    for target in target_columns:
        if target in train_with_meta.columns and target not in train_preprocessed.columns:
            missing_in_preprocessed.append(target)
    
    if missing_in_preprocessed:
        raise RuntimeError(
            f"CRITICAL: Target columns were dropped during preprocessing: {missing_in_preprocessed}. "
            f"This should never happen - Spark ML transformers preserve all columns not in inputCols. "
            f"Please check the preprocessing pipeline."
        )
    
    # Cache preprocessed DataFrames to avoid recomputation during model training
    if verbose:
        print("  Caching preprocessed DataFrames...")
    train_preprocessed.cache()
    val_preprocessed.cache()
    
    if verbose:
        train_count = train_preprocessed.count()
        val_count = val_preprocessed.count()
        print(f"  ✓ Preprocessing complete (applied to {train_count:,} train, {val_count:,} val rows)")
        print("  ✓ Preprocessed DataFrames cached")
        print("  ✓ Verified: All target columns preserved through preprocessing")
        
        # Debug: Check which target columns are available
        print(f"\n  🔍 Checking target columns in preprocessed DataFrames:")
        for target in target_columns:
            in_original = target in train_with_meta.columns
            in_preprocessed = target in train_preprocessed.columns
            status = "✓" if (in_original and in_preprocessed) else "✗"
            print(f"    {status} {target}: original={in_original}, preprocessed={in_preprocessed}")
            
            # If missing, show why
            if not in_original:
                if target == "prev_flight_taxi_time":
                    has_turnover = "lineage_actual_turnover_time_minutes" in train_with_meta.columns
                    print(f"      ⚠ Missing in original - lineage_actual_turnover_time_minutes exists: {has_turnover}")
                    if has_turnover:
                        print(f"      ⚠ BUG: Derivation should have created prev_flight_taxi_time but didn't!")
                elif target == "prev_flight_air_time":
                    has_crs = "prev_flight_crs_elapsed_time" in train_with_meta.columns
                    print(f"      ⚠ Missing in original - prev_flight_crs_elapsed_time exists: {has_crs}")
                    if has_crs:
                        print(f"      ⚠ BUG: Derivation should have created prev_flight_air_time but didn't!")
                elif target == "prev_flight_total_duration":
                    has_actual_dep = "actual_dep_time_minutes" in train_with_meta.columns
                    has_prev_actual_dep = "prev_flight_actual_dep_time_minutes" in train_with_meta.columns
                    has_actual_elapsed = "prev_flight_actual_elapsed_time" in train_with_meta.columns
                    print(f"      ⚠ Missing in original - actual_dep_time_minutes exists: {has_actual_dep}")
                    print(f"      ⚠ Missing in original - prev_flight_actual_dep_time_minutes exists: {has_prev_actual_dep}")
                    print(f"      ⚠ Missing in original - prev_flight_actual_elapsed_time exists: {has_actual_elapsed}")
                    if (has_actual_dep and has_prev_actual_dep) or has_actual_elapsed:
                        print(f"      ⚠ BUG: Derivation should have created prev_flight_total_duration but didn't!")
            elif in_original and not in_preprocessed:
                print(f"      ⚠ CRITICAL ERROR: Column exists in original but missing after preprocessing!")
                print(f"      ⚠ This indicates a bug in the preprocessing pipeline!")
    
    # Helper function to check if target column is valid for training
    def _check_target_column(df, target_col, col_name):
        """Check if target column exists and has non-null values."""
        if target_col not in df.columns:
            if verbose:
                print(f"  ⚠ Skipping {col_name} model (column '{target_col}' not found)")
            return False
        
        # Check if column has non-null values
        non_null_count = df.filter(col(target_col).isNotNull()).count()
        total_count = df.count()
        
        if non_null_count == 0:
            if verbose:
                print(f"  ⚠ Skipping {col_name} model (column '{target_col}' has no non-null values)")
            return False
        
        if verbose:
            print(f"  ✓ {col_name} model: {non_null_count:,}/{total_count:,} rows have non-null target")
        
        return True
    
    # Train and apply models in order: total_duration, taxi_time, air_time (air_time last to avoid positional issues)
    # CRITICAL: Pass train_with_meta/val_with_meta to preserve predictions from previous models
    # Train and apply total_duration model (only if prediction doesn't already exist)
    if "prev_flight_total_duration" not in existing_pred_cols:
        if _check_target_column(train_preprocessed, "prev_flight_total_duration", "prev_flight_total_duration"):
            train_with_meta, val_with_meta = _train_and_apply_model_with_preprocessed(
                train_preprocessed, val_preprocessed,
                "prev_flight_total_duration", "predicted_prev_flight_total_duration",
                model_config, verbose,
                train_with_meta=train_with_meta, val_with_meta=val_with_meta
            )
        else:
            if verbose:
                print(f"  ⏭ Skipping prev_flight_total_duration model (target column invalid)")
    else:
        if verbose:
            print(f"  ⏭ Skipping prev_flight_total_duration model (prediction already exists: {existing_pred_cols['prev_flight_total_duration']})")
    
    # Train and apply taxi_time model (actually predicting turnover time: current dep - prev arr)
    # Only if prediction doesn't already exist
    if "prev_flight_taxi_time" not in existing_pred_cols:
        if _check_target_column(train_preprocessed, "prev_flight_taxi_time", "prev_flight_turnover_time"):
            train_with_meta, val_with_meta = _train_and_apply_model_with_preprocessed(
                train_preprocessed, val_preprocessed,
                "prev_flight_taxi_time", "predicted_prev_flight_turnover_time",
                model_config, verbose,
                train_with_meta=train_with_meta, val_with_meta=val_with_meta
            )
        else:
            if verbose:
                print(f"  ⏭ Skipping prev_flight_turnover_time model (target column invalid)")
    else:
        if verbose:
            print(f"  ⏭ Skipping prev_flight_turnover_time model (prediction already exists: {existing_pred_cols['prev_flight_taxi_time']})")
    
    # Train and apply air_time model LAST (only if prediction doesn't already exist)
    # Swapped order: air_time is now last to avoid potential positional issues
    if "prev_flight_air_time" not in existing_pred_cols:
        if _check_target_column(train_preprocessed, "prev_flight_air_time", "prev_flight_air_time"):
            train_with_meta, val_with_meta = _train_and_apply_model_with_preprocessed(
                train_preprocessed, val_preprocessed,
                "prev_flight_air_time", "predicted_prev_flight_air_time",
                model_config, verbose,
                train_with_meta=train_with_meta, val_with_meta=val_with_meta
            )
        else:
            if verbose:
                print(f"  ⏭ Skipping prev_flight_air_time model (target column invalid)")
    else:
        if verbose:
            print(f"  ⏭ Skipping prev_flight_air_time model (prediction already exists: {existing_pred_cols['prev_flight_air_time']})")
    
    # Rename prediction columns to include model_id (only if they don't already have it)
    prediction_cols = [
        ("predicted_prev_flight_air_time", f"predicted_prev_flight_air_time_{model_id}"),
        ("predicted_prev_flight_turnover_time", f"predicted_prev_flight_turnover_time_{model_id}"),  # Note: This is turnover time (current dep - prev arr), not raw taxi time
        ("predicted_prev_flight_total_duration", f"predicted_prev_flight_total_duration_{model_id}")
    ]
    
    if verbose:
        print(f"\n  🔍 Renaming prediction columns to include model_id '{model_id}':")
    
    renamed_cols = []
    for col_name, final_name in prediction_cols:
        # Check if column already has the model_id suffix (from existing predictions)
        if final_name in train_with_meta.columns and final_name in val_with_meta.columns:
            renamed_cols.append(final_name)
            if verbose:
                print(f"    ✓ {final_name} (already has model_id suffix - keeping as-is)")
        elif col_name in train_with_meta.columns:
            # Rename to add model_id suffix
            train_with_meta = train_with_meta.withColumnRenamed(col_name, final_name)
            val_with_meta = val_with_meta.withColumnRenamed(col_name, final_name)
            renamed_cols.append(final_name)
            if verbose:
                print(f"    ✓ {col_name} → {final_name}")
        else:
            if verbose:
                print(f"    ✗ {col_name} (not found - model may not have trained)")
    
    if verbose:
        print(f"\n  🔍 Final prediction columns in DataFrame:")
        for col_name in renamed_cols:
            in_train = col_name in train_with_meta.columns
            in_val = col_name in val_with_meta.columns
            status = "✓" if (in_train and in_val) else "✗"
            print(f"    {status} {col_name} (train={in_train}, val={in_val})")
        
        if len(renamed_cols) == 0:
            print(f"    ⚠ WARNING: No prediction columns were created!")
        elif len(renamed_cols) < len(prediction_cols):
            print(f"    ⚠ WARNING: Only {len(renamed_cols)}/{len(prediction_cols)} prediction columns created!")
        else:
            print(f"    ✅ SUCCESS: All {len(renamed_cols)} prediction columns created and renamed!")
    
    # Unpersist cached preprocessed DataFrames to free memory
    if train_preprocessed.is_cached:
        train_preprocessed.unpersist()
    if val_preprocessed.is_cached:
        val_preprocessed.unpersist()
    
    if verbose:
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ✓ Meta-model predictions complete! (took {duration})")
        
        # Final summary of prediction columns
        print(f"\n{'='*80}")
        print("PREDICTION COLUMN CREATION SUMMARY:")
        print(f"{'='*80}")
        expected_final_cols = [
            f"predicted_prev_flight_air_time_{model_id}",
            f"predicted_prev_flight_turnover_time_{model_id}",
            f"predicted_prev_flight_total_duration_{model_id}"
        ]
        
        for col_name in expected_final_cols:
            in_train = col_name in train_with_meta.columns
            in_val = col_name in val_with_meta.columns
            status = "✅" if (in_train and in_val) else "❌"
            print(f"  {status} {col_name}")
            if in_train and in_val:
                # Check non-null counts
                train_non_null = train_with_meta.filter(col(col_name).isNotNull()).count()
                val_non_null = val_with_meta.filter(col(col_name).isNotNull()).count()
                train_total = train_with_meta.count()
                val_total = val_with_meta.count()
                print(f"      Train: {train_non_null:,}/{train_total:,} non-null ({train_non_null/train_total*100:.1f}%)")
                print(f"      Val: {val_non_null:,}/{val_total:,} non-null ({val_non_null/val_total*100:.1f}%)")
            else:
                print(f"      ⚠ Missing in {'train' if not in_train else ''}{' and ' if (not in_train and not in_val) else ''}{'val' if not in_val else ''}")
        
        success_count = sum(1 for col_name in expected_final_cols 
                          if col_name in train_with_meta.columns and col_name in val_with_meta.columns)
        print(f"\n  Result: {success_count}/{len(expected_final_cols)} prediction columns successfully created")
        if success_count == len(expected_final_cols):
            print(f"  ✅ SUCCESS: All prediction columns are ready to save!")
        else:
            print(f"  ⚠ WARNING: {len(expected_final_cols) - success_count} prediction column(s) missing!")
        print(f"{'='*80}\n")
    
    return train_with_meta, val_with_meta


def _retry_with_backoff(func, max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY, verbose=True):
    """
    Retry a function with exponential backoff for transient cluster failures.
    
    Args:
        func: Function to retry (should be a callable that takes no args)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)
        verbose: Whether to print retry messages
    
    Returns:
        Result of func()
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()  # Case-insensitive matching
            # Also check error message/description if available
            error_msg = ""
            if hasattr(e, 'java_exception'):
                try:
                    error_msg = str(e.java_exception.getMessage()).lower()
                except:
                    pass
            if hasattr(e, 'msg'):
                error_msg = str(e.msg).lower()
            
            # Check if any retryable error pattern matches (case-insensitive)
            is_retryable = any(
                err.lower() in error_str or err.lower() in error_msg
                for err in RETRYABLE_ERRORS
            )
            
            if attempt < max_retries and is_retryable:
                delay = initial_delay * (2 ** attempt)
                if verbose:
                    print(f"  ⚠ Retryable error (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}")
                    print(f"  ⏳ Waiting {delay}s before retry...")
                time.sleep(delay)
                last_exception = e
            else:
                # Not retryable or out of retries
                if not is_retryable and verbose:
                    print(f"  ❌ Non-retryable error: {type(e).__name__}")
                raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


def _get_shared_features(df):
    """Get shared feature lists used by all meta-models."""
    categorical_features = [
        'is_first_flight', 'is_jump',
        'prev_flight_origin', 'prev_flight_dest',
        'prev_flight_month', 'prev_flight_op_carrier',
        'prev_flight_day_of_week',
        # Hour features - extract hour from departure time if available
        # These help capture time-of-day patterns (e.g., morning vs evening operations)
        'dep_time_blk',  # Hour block of current flight departure (if available)
        # NOTE: prev_flight_hourlypresentweathertype, prev_flight_hourlyskyconditions, and
        # prev_flight_dailyweather were categorical weather features that were incorrectly
        # cast to numeric in previous versions. Removed to avoid confusion.
    ]
    
    numerical_features = [
        'prev_flight_distance', 'prev_flight_crs_elapsed_time',
        'prev_flight_origin_hourlywindspeed', 'prev_flight_dest_hourlywindspeed',
        'prev_flight_origin_elevation', 'prev_flight_dest_elevation',
        # Scheduled duration features - these are data leakage-free and provide baseline expectations
        # Scheduled rotation time: scheduled prev departure to scheduled current departure
        'scheduled_lineage_rotation_time_minutes',
        # Scheduled turnover time: scheduled prev arrival to scheduled current departure
        'scheduled_lineage_turnover_time_minutes',
        # Scheduled flight time for previous flight (scheduled arrival - scheduled departure)
        'prev_flight_scheduled_flight_time_minutes',
    ]
    
    # Add graph features if available
    graph_features = [
        'prev_flight_origin_pagerank_weighted', 'prev_flight_origin_pagerank_unweighted',
        'prev_flight_dest_pagerank_weighted', 'prev_flight_dest_pagerank_unweighted',
        'origin_pagerank_weighted', 'origin_pagerank_unweighted'
    ]
    numerical_features.extend([f for f in graph_features if f in df.columns])
    
    # Previous Flight Weather Features
    # See docs/previous_flight_weather_features.md for detailed documentation
    # 
    # Summary: Uses recommended subset of 14 most predictive weather features for
    # prev_flight_air_time and prev_flight_total_duration models. NOT used for
    # prev_flight_turnover_time (happens at current airport, not previous).
    
    # Previous flight weather features - RECOMMENDED SUBSET (14 most predictive features)
    # These are created by flight_lineage_features.py when weather data is available (e.g., OTPW)
    # 
    # USAGE BY MODEL:
    #   - prev_flight_air_time: USE weather features (weather affects flight duration)
    #   - prev_flight_total_duration: USE weather features (weather affects overall flight time)
    #   - prev_flight_turnover_time: DO NOT USE weather features (turnover happens at current airport, not previous)
    #
    # NOTE: Currently all models share the same feature set. To exclude weather from turnover_time,
    # you would need to create separate feature lists per model. For now, weather features are included
    # in the shared feature set, but turnover_time model may not benefit from them (which is fine).
    prev_flight_weather_features_recommended = [
        # WIND (Highest Impact - 40-60% of weather-related delay variance)
        'prev_flight_hourlywindspeed',           # Headwinds/tailwinds directly affect flight time
        'prev_flight_hourlywinddirection',       # Combined with route determines headwind component
        'prev_flight_hourlywindgustspeed',       # Turbulence, approach safety, route deviations
        'prev_flight_dailypeakwindspeed',        # Extreme wind conditions
        
        # PRECIPITATION & VISIBILITY (High Impact - 20-30% of weather-related delay variance)
        'prev_flight_hourlyprecipitation',       # Flight path deviations, approach restrictions
        'prev_flight_hourlyvisibility',          # Approach procedures, landing restrictions
        # NOTE: prev_flight_hourlypresentweathertype and prev_flight_hourlyskyconditions are categorical
        # and should NOT be in numerical_features. They are excluded from this list.
        'prev_flight_dailyprecipitation',        # Aggregate precipitation impact
        
        # TEMPERATURE & PRESSURE (Moderate Impact - 10-20% of weather-related delay variance)
        'prev_flight_hourlydrybulbtemperature',  # Air density, engine performance
        'prev_flight_hourlysealevelpressure',    # Air density, takeoff/landing performance
        'prev_flight_hourlyaltimetersetting',    # Critical for aviation, affects flight levels
        'prev_flight_dailymaximumdrybulbtemperature',  # Extreme heat effects
        'prev_flight_dailyminimumdrybulbtemperature',  # Extreme cold effects
        
        # HUMIDITY (Lower but still predictive - 5-10% of weather-related delay variance)
        'prev_flight_hourlyrelativehumidity',    # Engine performance, air density
        'prev_flight_hourlydewpointtemperature', # Condensation, icing risk
    ]
    
    # Safe fallback: only include weather features that exist in the DataFrame
    # This allows the function to work with or without weather data
    # NOTE: These features are used for air_time and total_duration models, but NOT for turnover_time
    # (turnover happens at current airport, not previous airport)
    available_weather_features = [f for f in prev_flight_weather_features_recommended if f in df.columns]
    missing_weather_features = [f for f in prev_flight_weather_features_recommended if f not in df.columns]
    
    if available_weather_features:
        numerical_features.extend(available_weather_features)
        if VERBOSE:
            print(f"  ✓ Added {len(available_weather_features)} previous flight weather features to meta-model inputs")
            print(f"    Using recommended subset: Wind (4), Precip/Visibility (5), Temp/Pressure (5), Humidity (2)")
            print(f"    NOTE: Weather features used for air_time and total_duration models only")
            print(f"    NOTE: Turnover_time model does NOT use previous flight weather (happens at current airport)")
    else:
        if VERBOSE:
            print(f"  ⚠ Previous flight weather features not available (expected for CUSTOM dataset without weather)")
            print(f"    Missing {len(missing_weather_features)} recommended weather features - will use other features only")
            print(f"    Weather features will be available for OTPW dataset (has weather data joined)")
            if missing_weather_features:
                print(f"    Missing features: {missing_weather_features[:5]}{'...' if len(missing_weather_features) > 5 else ''}")
    
    # Filter to only features that exist
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    return categorical_features, numerical_features


def _build_shared_preprocessing_pipeline(categorical_features, numerical_features):
    """Build shared preprocessing pipeline (fit once, used for all models)."""
    stages = []
    
    # Impute numerical features
    if numerical_features:
        imputer = Imputer(
            inputCols=numerical_features,
            outputCols=[f"{col}_IMPUTED" for col in numerical_features],
            strategy="mean"
        )
        stages.append(imputer)
    
    # Index and encode categorical features
    if categorical_features:
        indexer = StringIndexer(
            inputCols=categorical_features,
            outputCols=[f"{col}_INDEX" for col in categorical_features],
            handleInvalid="keep"
        )
        stages.append(indexer)
        
        encoder = OneHotEncoder(
            inputCols=[f"{col}_INDEX" for col in categorical_features],
            outputCols=[f"{col}_VEC" for col in categorical_features],
            dropLast=False
        )
        stages.append(encoder)
    
    return Pipeline(stages=stages)


def _train_and_apply_model_with_preprocessed(train_preprocessed, val_preprocessed, 
                                              target_col, pred_col_name, model_config, verbose,
                                              train_with_meta=None, val_with_meta=None):
    """
    Train a single meta-model using preprocessed data and apply to both train and val.
    
    Args:
        train_preprocessed: Preprocessed training DataFrame (for training the model)
        val_preprocessed: Preprocessed validation DataFrame (for training the model)
        target_col: Target column name
        pred_col_name: Final prediction column name
        model_config: Model configuration dict
        verbose: Whether to print verbose output
        train_with_meta: Optional DataFrame with existing predictions (will be joined with new predictions)
        val_with_meta: Optional DataFrame with existing predictions (will be joined with new predictions)
    
    Returns:
        tuple: (train_df_with_predictions, val_df_with_predictions)
    """
    # Filter to rows with valid target (use preprocessed for training)
    train_filtered = train_preprocessed.filter(col(target_col).isNotNull())
    
    # Optional: Sample training data for very large datasets (60M)
    if TRAINING_SAMPLE_FRACTION is not None and TRAINING_SAMPLE_FRACTION < 1.0:
        if verbose:
            print(f"  Sampling {TRAINING_SAMPLE_FRACTION*100:.0f}% of training data for faster training...")
        train_filtered = train_filtered.sample(fraction=TRAINING_SAMPLE_FRACTION, seed=42)
    
    count = train_filtered.count()
    
    if count < 100:
        if verbose:
            print(f"  ⚠ Skipping {target_col} model (only {count} rows)")
        # Return with NULL predictions
        train_result = train_preprocessed.withColumn(pred_col_name, lit(None).cast("double"))
        val_result = val_preprocessed.withColumn(pred_col_name, lit(None).cast("double"))
        return train_result, val_result
    
    # Get processed feature column names
    processed_categorical = [c for c in train_filtered.columns if "_VEC" in c]
    processed_numerical = [c for c in train_filtered.columns if "_IMPUTED" in c]
    
    if verbose:
        print(f"  Training {target_col} model ({count:,} rows, {len(processed_categorical)} cat, {len(processed_numerical)} num)")
    
    # Adaptive complexity: Reduce model complexity for very large datasets to avoid resource issues
    # Fold 4 (test fold) has ~15.9M training rows - needs lighter model
    # Balance: With more data, simpler models can still achieve good accuracy
    base_num_rounds = model_config.get("num_boost_round", 75)
    base_max_depth = model_config.get("max_depth", 6)
    
    if count > 10_000_000:  # >10M rows: moderate reduction (more data compensates for simpler model)
        num_rounds = max(50, int(base_num_rounds * 0.67))  # ~67% of base (50 rounds), at least 50
        max_depth = max(5, base_max_depth - 1)  # Depth 5 (1 less than base), at least 5
        if verbose:
            print(f"    ⚠ Large dataset detected ({count:,} rows) - moderate complexity reduction:")
            print(f"      num_boost_round: {base_num_rounds} → {num_rounds} (more data compensates)")
            print(f"      max_depth: {base_max_depth} → {max_depth}")
    elif count > 5_000_000:  # 5-10M rows: slight reduction
        num_rounds = max(60, int(base_num_rounds * 0.8))  # 80% of base (60 rounds), at least 60
        max_depth = base_max_depth  # Keep full depth
        if verbose:
            print(f"    ⚠ Medium-large dataset ({count:,} rows) - slight complexity reduction:")
            print(f"      num_boost_round: {base_num_rounds} → {num_rounds}")
            print(f"      max_depth: {base_max_depth} (unchanged)")
    else:
        num_rounds = base_num_rounds
        max_depth = base_max_depth
    
    # Cache the training DataFrame to avoid recomputation during Random Forest training
    if not train_filtered.is_cached:
        if verbose:
            print("    Caching training DataFrame for XGBoost...")
        train_filtered.cache()
        _ = train_filtered.count()  # Materialize cache
        if verbose:
            print("    ✓ Cache materialized")
    
    # Assemble preprocessed features
    assembler = VectorAssembler(
        inputCols=processed_categorical + processed_numerical,
        outputCol="features",
        handleInvalid="skip"
    )
    
    # Build XGBoost model (only model type supported now)
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    
    # Dynamically determine number of workers (cap at available executors to avoid barrier errors)
    # XGBoost requires num_workers <= available executor slots
    requested_workers = sc.defaultParallelism
    # num_rounds and max_depth are now set adaptively above based on dataset size
    
    # Get actual number of executors (fallback to requested if unavailable)
    try:
        # Try to get executor count from Spark context
        executor_infos = sc.statusTracker().getExecutorInfos()
        executor_count = len([e for e in executor_infos if e.isActive])
        # XGBoost typically needs num_workers <= executor_count
        # Use min of requested and available, but at least 1
        num_workers = max(1, min(requested_workers, executor_count))
        if num_workers < requested_workers and verbose:
            print(f"      ⚠ Adjusted num_workers from {requested_workers} to {num_workers} (available executors: {executor_count})")
    except Exception as e:
        # Fallback: use requested workers, but cap at reasonable number
        num_workers = min(requested_workers, 30)  # Cap at 30 to avoid barrier errors and timeouts (reduced from 50)
        if verbose:
            print(f"      Using num_workers={num_workers} (could not detect executor count: {str(e)[:50]})")
    
    # XGBoost Regressor (much faster than Random Forest)
    # Use unique temporary prediction column name to avoid overwriting between models
    temp_pred_col = f"_temp_prediction_{target_col.replace('prev_flight_', '')}"
    model = SparkXGBRegressor(
        num_workers=num_workers,
        label_col=target_col,
        features_col="features",
        prediction_col=temp_pred_col,  # Use unique temp name to avoid overwriting
        missing=0.0,
        num_boost_round=num_rounds,
        max_depth=max_depth,
        learning_rate=model_config.get("learning_rate", 0.15),
        min_child_weight=model_config.get("min_child_weight", 1),
        seed=42
    )
    
    # Log actual parameter being used for debugging
    if verbose:
        print(f"      Using num_boost_round={num_rounds}, num_workers={num_workers}")
    
    # Build model pipeline (preprocessing already done)
    model_pipeline = Pipeline(stages=[assembler, model])
    
    # Add timestamp before training
    if verbose:
        fit_start = datetime.now()
        fit_timestamp = fit_start.strftime("%Y-%m-%d %H:%M:%S")
        print(f"    [{fit_timestamp}] Starting XGBoost training (this may take 2-10 minutes)...")
        print(f"      Config: {num_rounds} rounds, max_depth={max_depth}, learning_rate={model_config.get('learning_rate', 0.15)}")
    
    # Suppress verbose XGBoost logging (booster params, train_call_kwargs, etc.)
    xgboost_logger = logging.getLogger("XGBoost-PySpark")
    original_level = xgboost_logger.level if xgboost_logger.level != logging.NOTSET else logging.INFO
    xgboost_logger.setLevel(logging.WARNING)  # Only show warnings and errors, suppress INFO logs
    
    # Train (with retry for cluster failures)
    def _fit_model():
        return model_pipeline.fit(train_filtered)
    
    fitted_model = _retry_with_backoff(_fit_model, verbose=verbose)
    
    # Restore original logging level
    xgboost_logger.setLevel(original_level)
    
    # Log duration
    if verbose:
        fit_end = datetime.now()
        fit_duration = (fit_end - fit_start).total_seconds()
        print(f"    ✓ {target_col} model trained (took {fit_duration:.1f}s / {fit_duration/60:.1f}min)")
    
    # Unpersist cached DataFrame to free memory
    if train_filtered.is_cached:
        train_filtered.unpersist()
    
    # Apply to both train and val (inference step - can take time on large datasets, with retry)
    if verbose:
        inf_start = datetime.now()
        inf_timestamp = inf_start.strftime("%Y-%m-%d %H:%M:%S")
        print(f"    [{inf_timestamp}] Applying predictions to train and validation sets...")
    
    def _transform_train():
        return fitted_model.transform(train_preprocessed)
    
    def _transform_val():
        return fitted_model.transform(val_preprocessed)
    
    train_with_pred = _retry_with_backoff(_transform_train, verbose=verbose)
    val_with_pred = _retry_with_backoff(_transform_val, verbose=verbose)
    
    if verbose:
        inf_end = datetime.now()
        inf_duration = (inf_end - inf_start).total_seconds()
        print(f"    ✓ Inference complete (took {inf_duration:.1f}s / {inf_duration/60:.1f}min)")
    
    # Extract prediction column and add to train_with_meta/val_with_meta (preserving existing predictions)
    if temp_pred_col in train_with_pred.columns:
        # Rename prediction column from unique temp name to final name
        train_with_pred = train_with_pred.withColumnRenamed(temp_pred_col, pred_col_name)
        val_with_pred = val_with_pred.withColumnRenamed(temp_pred_col, pred_col_name)
        
        # Debug: Check prediction column before join
        if verbose:
            train_pred_count = train_with_pred.filter(col(pred_col_name).isNotNull()).count()
            train_total = train_with_pred.count()
            val_pred_count = val_with_pred.filter(col(pred_col_name).isNotNull()).count()
            val_total = val_with_pred.count()
            print(f"    🔍 Before join: {train_pred_count:,}/{train_total:,} train predictions non-null, {val_pred_count:,}/{val_total:,} val predictions non-null")
        
        # Impute NULLs with fallback
        if target_col == "prev_flight_air_time":
            fallback = coalesce(col("prev_flight_crs_elapsed_time"), lit(120.0))
        elif target_col == "prev_flight_taxi_time":
            fallback = lit(25.0)
        else:
            fallback = coalesce(col("prev_flight_crs_elapsed_time"), lit(120.0))
        
        train_with_pred = train_with_pred.withColumn(
            pred_col_name, coalesce(col(pred_col_name), fallback)
        )
        val_with_pred = val_with_pred.withColumn(
            pred_col_name, coalesce(col(pred_col_name), fallback)
        )
        
        # Add prediction column to train_with_meta/val_with_meta (preserving existing predictions)
        if train_with_meta is not None and val_with_meta is not None:
            # Use _row_id that was added before preprocessing for reliable joining
            if "_row_id" in train_with_pred.columns and "_row_id" in train_with_meta.columns:
                if verbose:
                    print(f"    🔍 Joining predictions back using _row_id (stable row identifier)")
                
                # Join on _row_id (most reliable method)
                train_with_meta = train_with_meta.join(
                    train_with_pred.select("_row_id", pred_col_name),
                    on="_row_id",
                    how="left"
                )
                val_with_meta = val_with_meta.join(
                    val_with_pred.select("_row_id", pred_col_name),
                    on="_row_id",
                    how="left"
                )
                
                # Debug: Check prediction column after join
                if verbose:
                    train_pred_count_after = train_with_meta.filter(col(pred_col_name).isNotNull()).count()
                    train_total_after = train_with_meta.count()
                    val_pred_count_after = val_with_meta.filter(col(pred_col_name).isNotNull()).count()
                    val_total_after = val_with_meta.count()
                    print(f"    🔍 After join: {train_pred_count_after:,}/{train_total_after:,} train predictions non-null, {val_pred_count_after:,}/{val_total_after:,} val predictions non-null")
                    
                    if train_pred_count_after == 0 and train_pred_count > 0:
                        print(f"    ⚠⚠⚠ WARNING: All predictions became NULL after join! Join may have failed.")
                        print(f"    ⚠ This suggests _row_id doesn't match between DataFrames.")
            else:
                # Fallback: use common columns (less reliable)
                if verbose:
                    print(f"    ⚠ _row_id not found, falling back to common columns join")
                
                common_cols = [c for c in train_preprocessed.columns 
                              if c in train_with_meta.columns and c not in ["features", pred_col_name, "_row_id"]]
                
                if len(common_cols) > 0:
                    if verbose:
                        print(f"    🔍 Joining on {len(common_cols)} common columns: {common_cols[:5]}..." if len(common_cols) > 5 else f"    🔍 Joining on {len(common_cols)} common columns: {common_cols}")
                    
                    train_with_meta = train_with_meta.join(
                        train_with_pred.select(common_cols + [pred_col_name]),
                        on=common_cols,
                        how="left"
                    )
                    val_with_meta = val_with_meta.join(
                        val_with_pred.select(common_cols + [pred_col_name]),
                        on=common_cols,
                        how="left"
                    )
                else:
                    raise RuntimeError("Cannot join predictions back: no _row_id and no common columns found!")
        else:
            # First model - use transformed DataFrames but drop features and keep _row_id
            train_with_meta = train_with_pred.drop("features") if "features" in train_with_pred.columns else train_with_pred
            val_with_meta = val_with_pred.drop("features") if "features" in val_with_pred.columns else val_with_pred
    
    # Note: Completion message already printed above after training
    return train_with_meta, val_with_meta


# -------------------------
# MAIN FUNCTION
# -------------------------
def add_meta_model_predictions_to_folds(version: str, input_suffix: str = "_with_graph"):
    """Add meta-model predictions to all folds."""
    spark = SparkSession.builder.getOrCreate()
    
    print(f"\n{'='*80}")
    print(f"ADDING META-MODEL PREDICTIONS to {version}")
    print(f"{'='*80}")
    print(f"Input suffix: {input_suffix}")
    print(f"Output suffix: {OUTPUT_SUFFIX}")
    print(f"Meta-model IDs: {list(META_MODEL_IDS.keys())}")
    print(f"Retry configuration: {MAX_RETRIES} max retries, {INITIAL_RETRY_DELAY}s initial delay (exponential backoff)")
    if SKIP_EXISTING_FOLDS:
        print(f"  ✓ Resume mode: ENABLED (will skip completed folds)")
    else:
        print(f"  ⚠ Resume mode: DISABLED (will re-run all folds)")
        print(f"  💡 Tip: Set SKIP_EXISTING_FOLDS=True for overnight runs to auto-resume after failures")
    
    # Load all available folds (same pattern as cv.py dataloader)
    folds = load_folds_for_version(version, input_suffix)
    
    if not folds:
        print(f"  ⚠ No folds found for version {version} with suffix '{input_suffix}'")
        print(f"  Expected pattern: OTPW_{SOURCE}_{version}{input_suffix}_FOLD_*_TRAIN.parquet")
        return
    
    print(f"  Found {len(folds)} folds")
    
    if SKIP_EXISTING_FOLDS:
        print(f"  ⏭ Skip existing folds: ENABLED (will skip folds that already exist)")
    else:
        print(f"  ⏭ Skip existing folds: DISABLED (will re-run all folds)")
    
    processed_count = 0
    skipped_count = 0
    
    for fold_idx, (train_df, val_or_test_df, fold_type) in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}/{len(folds)} ({fold_type})")
        print(f"{'='*60}")
        
        # Check if fold already exists (resume mode)
        # Pass expected_model_ids to verify all meta-model predictions are present
        if SKIP_EXISTING_FOLDS and fold_exists(version, fold_idx, OUTPUT_SUFFIX, fold_type, 
                                                expected_model_ids=list(META_MODEL_IDS.keys())):
            if VERBOSE:
                print(f"  ⏭ Skipping fold {fold_idx} (already exists with all expected meta-model predictions)")
            print(f"✅ Fold {fold_idx} skipped (already complete)")
            skipped_count += 1
            continue
        
        if VERBOSE:
            train_count = train_df.count()
            val_or_test_count = val_or_test_df.count()
            print(f"  Train: {train_count:,} rows")
            print(f"  {fold_type}: {val_or_test_count:,} rows")
        
        # Add meta-model predictions for each model ID
        train_with_meta = train_df
        val_or_test_with_meta = val_or_test_df
        
        for model_id in META_MODEL_IDS:
            if VERBOSE:
                print(f"\n  Computing meta-model predictions: {model_id}")
            
            # Retry meta-model computation with exponential backoff
            def _compute_meta_models():
                return compute_meta_model_predictions(
                    train_with_meta, val_or_test_with_meta, 
                    model_id=model_id, 
                    verbose=VERBOSE
                )
            
            try:
                train_with_meta, val_or_test_with_meta = _retry_with_backoff(
                    _compute_meta_models, 
                    verbose=VERBOSE
                )
            except Exception as e:
                error_str = str(e)
                is_retryable = any(err in error_str for err in RETRYABLE_ERRORS)
                
                if is_retryable:
                    print(f"  ❌ ERROR: Meta-model {model_id} failed after {MAX_RETRIES} retries")
                    print(f"  💡 Tip: Enable SKIP_EXISTING_FOLDS=True and re-run to resume from completed folds")
                else:
                    print(f"  ❌ ERROR: Meta-model {model_id} failed with non-retryable error")
                
                raise RuntimeError(f"Meta-model precomputation failed for {model_id}. Error: {str(e)}") from e
        
        # Drop temporary _row_id column before saving (it was only used for joining)
        if "_row_id" in train_with_meta.columns:
            train_with_meta = train_with_meta.drop("_row_id")
        if "_row_id" in val_or_test_with_meta.columns:
            val_or_test_with_meta = val_or_test_with_meta.drop("_row_id")
        
        # Save (checkpoint after each fold - allows resume if interrupted)
        if VERBOSE:
            save_start = datetime.now()
            save_timestamp = save_start.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n💾 [{save_timestamp}] Saving fold {fold_idx} to {OUTPUT_SUFFIX}...")
        
        save_fold_with_suffix(version, fold_idx, train_with_meta, val_or_test_with_meta, 
                             OUTPUT_SUFFIX, fold_type)
        
        if VERBOSE:
            save_end = datetime.now()
            save_duration = (save_end - save_start).total_seconds()
            print(f"  ✓ Saved in {save_duration:.1f}s / {save_duration/60:.1f}min")
        
        print(f"✅ Fold {fold_idx} complete (saved - safe to interrupt)")
        processed_count += 1
    
    print(f"\n{'='*80}")
    print(f"✅ Complete! All folds processed")
    print(f"   Processed: {processed_count} folds")
    if SKIP_EXISTING_FOLDS:
        print(f"   Skipped: {skipped_count} folds (already existed)")
    print(f"   Output suffix: {OUTPUT_SUFFIX}")
    print(f"   Use version='{version}' with suffix='{OUTPUT_SUFFIX}' in cv.py")
    print(f"{'='*80}")


# -------------------------
# MAIN
# -------------------------
# Process all versions in VERSIONS list
for version in VERSIONS:
    add_meta_model_predictions_to_folds(version, INPUT_SUFFIX)

print("\n" + "="*80)
print("✅ All versions processed successfully!")
print("="*80)
