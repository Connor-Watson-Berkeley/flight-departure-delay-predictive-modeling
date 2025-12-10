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

# XGBoost support (much faster than Random Forest - 5-10x speedup)
from xgboost.spark import SparkXGBRegressor

# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M", "XM"])
VERSIONS = ["60M","3M","12M"]  # <-- EDIT THIS LIST

INPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "CUSTOM"
WRITE_MODE = "overwrite"
VERBOSE = True
OUTPUT_SUFFIX = "_with_graph_and_metamodels"
INPUT_SUFFIX = "_with_graph"  # Input suffix: '_with_graph' (base) or '_with_graph_and_metamodels' (to add more models)
SKIP_EXISTING_FOLDS = False  # If True, skip folds that already exist in output (resume mode)

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
    """
    cv = load_cv_module()
    
    # Create data loader with suffix
    data_loader = cv.FlightDelayDataLoader(suffix=input_suffix)
    
    # Load folds for this specific version (uses suffix automatically)
    folds_raw = data_loader._load_version(version)
    
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


def fold_exists(version: str, fold_idx: int, output_suffix: str, fold_type: str):
    """Check if a fold already exists in the output folder."""
    from pyspark.sql import SparkSession
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
        return train_exists and val_exists
    except Exception as e:
        # Fallback: try to read schema (slower but more reliable)
        try:
            train_df = spark.read.parquet(train_path)
            val_df = spark.read.parquet(val_path)
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
    
    # Add special case flags
    if "is_first_flight" not in train_with_meta.columns:
        train_with_meta = train_with_meta.withColumn("is_first_flight", (col("lineage_rank") == 1).cast("int"))
        val_with_meta = val_with_meta.withColumn("is_first_flight", (col("lineage_rank") == 1).cast("int"))
    if "is_jump" not in train_with_meta.columns:
        train_with_meta = train_with_meta.withColumn("is_jump", col("lineage_is_jump").cast("int"))
        val_with_meta = val_with_meta.withColumn("is_jump", col("lineage_is_jump").cast("int"))
    
    # Compute derived target columns if needed
    if "prev_flight_taxi_time" not in train_with_meta.columns:
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
    
    if "prev_flight_total_duration" not in train_with_meta.columns:
        if "prev_flight_actual_elapsed_time" in train_with_meta.columns:
            train_with_meta = train_with_meta.withColumn(
                "prev_flight_total_duration", col("prev_flight_actual_elapsed_time")
            )
            val_with_meta = val_with_meta.withColumn(
                "prev_flight_total_duration", col("prev_flight_actual_elapsed_time")
            )
    
    # Get shared feature lists (all models use the same features)
    categorical_features, numerical_features = _get_shared_features(train_with_meta)
    
    if verbose:
        print(f"  Shared features: {len(categorical_features)} categorical, {len(numerical_features)} numerical")
        print(f"  Building shared preprocessing pipeline...")
    
    # Build shared preprocessing pipeline (fit once, use for all models)
    preprocessing_pipeline = _build_shared_preprocessing_pipeline(categorical_features, numerical_features)
    
    # Fit preprocessing on training data (includes all rows, not filtered by target)
    fitted_preprocessing = preprocessing_pipeline.fit(train_with_meta)
    
    # Apply preprocessing to both train and val once
    train_preprocessed = fitted_preprocessing.transform(train_with_meta)
    val_preprocessed = fitted_preprocessing.transform(val_with_meta)
    
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
    
    # Train and apply air_time model
    if "prev_flight_air_time" in train_with_meta.columns:
        train_with_meta, val_with_meta = _train_and_apply_model_with_preprocessed(
            train_preprocessed, val_preprocessed,
            "prev_flight_air_time", "predicted_prev_flight_air_time",
            model_config, verbose
        )
    
    # Train and apply taxi_time model
    if "prev_flight_taxi_time" in train_with_meta.columns:
        train_with_meta, val_with_meta = _train_and_apply_model_with_preprocessed(
            train_preprocessed, val_preprocessed,
            "prev_flight_taxi_time", "predicted_prev_flight_taxi_time",
            model_config, verbose
        )
    
    # Train and apply total_duration model
    if "prev_flight_total_duration" in train_with_meta.columns:
        train_with_meta, val_with_meta = _train_and_apply_model_with_preprocessed(
            train_preprocessed, val_preprocessed,
            "prev_flight_total_duration", "predicted_prev_flight_total_duration",
            model_config, verbose
        )
    
    # Rename prediction columns to include model_id
    prediction_cols = [
        "predicted_prev_flight_air_time",
        "predicted_prev_flight_taxi_time",
        "predicted_prev_flight_total_duration"
    ]
    
    for col_name in prediction_cols:
        if col_name in train_with_meta.columns:
            new_name = f"{col_name}_{model_id}"
            train_with_meta = train_with_meta.withColumnRenamed(col_name, new_name)
            val_with_meta = val_with_meta.withColumnRenamed(col_name, new_name)
    
    # Unpersist cached preprocessed DataFrames to free memory
    if train_preprocessed.is_cached:
        train_preprocessed.unpersist()
    if val_preprocessed.is_cached:
        val_preprocessed.unpersist()
    
    if verbose:
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Meta-model predictions complete! (took {duration})")
    
    return train_with_meta, val_with_meta


def _get_shared_features(df):
    """Get shared feature lists used by all meta-models."""
    categorical_features = [
        'is_first_flight', 'is_jump',
        'prev_flight_origin', 'prev_flight_dest',
        'prev_flight_month', 'prev_flight_op_carrier',
        'prev_flight_day_of_week'
    ]
    
    numerical_features = [
        'prev_flight_distance', 'prev_flight_crs_elapsed_time',
        'prev_flight_origin_hourlywindspeed', 'prev_flight_dest_hourlywindspeed',
        'prev_flight_origin_elevation', 'prev_flight_dest_elevation'
    ]
    
    # Add graph features if available
    graph_features = [
        'prev_flight_origin_pagerank_weighted', 'prev_flight_origin_pagerank_unweighted',
        'prev_flight_dest_pagerank_weighted', 'prev_flight_dest_pagerank_unweighted',
        'origin_pagerank_weighted', 'origin_pagerank_unweighted'
    ]
    numerical_features.extend([f for f in graph_features if f in df.columns])
    
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
                                              target_col, pred_col_name, model_config, verbose):
    """Train a single meta-model using preprocessed data and apply to both train and val."""
    # Filter to rows with valid target
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
    
    # XGBoost Regressor (much faster than Random Forest)
    # Note: SparkXGBRegressor may use 'n_estimators' instead of 'num_boost_round'
    # If num_boost_round doesn't work, try n_estimators
    num_rounds = model_config.get("num_boost_round", 75)
    model = SparkXGBRegressor(
        num_workers=sc.defaultParallelism,
        label_col=target_col,
        features_col="features",
        prediction_col="prediction",
        missing=0.0,
        num_boost_round=num_rounds,  # Try this first
        max_depth=model_config.get("max_depth", 6),
        learning_rate=model_config.get("learning_rate", 0.15),
        min_child_weight=model_config.get("min_child_weight", 1),
        seed=42
    )
    
    # Log actual parameter being used for debugging
    if verbose:
        print(f"      Using num_boost_round={num_rounds} (check XGBoost logs to confirm)")
    
    # Build model pipeline (preprocessing already done)
    model_pipeline = Pipeline(stages=[assembler, model])
    
    # Add timestamp before training
    if verbose:
        fit_start = datetime.now()
        fit_timestamp = fit_start.strftime("%Y-%m-%d %H:%M:%S")
        print(f"    [{fit_timestamp}] Starting XGBoost training (this may take 2-10 minutes)...")
        print(f"      Config: {model_config.get('num_boost_round', 75)} rounds, max_depth={model_config.get('max_depth', 6)}, learning_rate={model_config.get('learning_rate', 0.15)}")
    
    # Train
    fitted_model = model_pipeline.fit(train_filtered)
    
    # Log duration
    if verbose:
        fit_end = datetime.now()
        fit_duration = (fit_end - fit_start).total_seconds()
        print(f"    ✓ {target_col} model trained (took {fit_duration:.1f}s / {fit_duration/60:.1f}min)")
    
    # Unpersist cached DataFrame to free memory
    if train_filtered.is_cached:
        train_filtered.unpersist()
    
    # Apply to both train and val (inference step - can take time on large datasets)
    if verbose:
        inf_start = datetime.now()
        inf_timestamp = inf_start.strftime("%Y-%m-%d %H:%M:%S")
        print(f"    [{inf_timestamp}] Applying predictions to train and validation sets...")
    
    train_with_pred = fitted_model.transform(train_preprocessed)
    val_with_pred = fitted_model.transform(val_preprocessed)
    
    if verbose:
        inf_end = datetime.now()
        inf_duration = (inf_end - inf_start).total_seconds()
        print(f"    ✓ Inference complete (took {inf_duration:.1f}s / {inf_duration/60:.1f}min)")
    
    # Rename prediction column
    if "prediction" in train_with_pred.columns:
        train_with_pred = train_with_pred.withColumnRenamed("prediction", pred_col_name)
        val_with_pred = val_with_pred.withColumnRenamed("prediction", pred_col_name)
        
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
    
    # Drop intermediate columns (keep preprocessed columns for other models, but drop features vector)
    if "features" in train_with_pred.columns:
        train_with_pred = train_with_pred.drop("features")
        val_with_pred = val_with_pred.drop("features")
    
    # Note: Completion message already printed above after training
    return train_with_pred, val_with_pred


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
        if SKIP_EXISTING_FOLDS and fold_exists(version, fold_idx, OUTPUT_SUFFIX, fold_type):
            if VERBOSE:
                print(f"  ⏭ Skipping fold {fold_idx} (already exists in output)")
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
            
            try:
                train_with_meta, val_or_test_with_meta = compute_meta_model_predictions(
                    train_with_meta, val_or_test_with_meta, 
                    model_id=model_id, 
                    verbose=VERBOSE
                )
            except Exception as e:
                print(f"  ❌ ERROR: Meta-model {model_id} failed: {str(e)}")
                raise RuntimeError(f"Meta-model precomputation failed for {model_id}. Error: {str(e)}") from e
        
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

