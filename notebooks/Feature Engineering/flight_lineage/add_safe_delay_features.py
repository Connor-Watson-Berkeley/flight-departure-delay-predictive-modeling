"""
add_safe_departure_delay.py - Add safe delay features to existing folds

This script reads existing folds (base, _with_graph, or _with_graph_and_metamodels) and adds
safe delay features, saving back to the same location (in-place update).

Features added:
1. safe_prev_departure_delay: Data leakage-safe previous flight departure delay
2. safe_prev_arrival_delay: Data leakage-safe previous flight arrival delay
3. safe_time_since_prev_arrival: Time between previous flight's arrival and prediction cutoff

All features use intelligent data leakage handling based on the 2-hour prediction cutoff window.

Usage:
    Set VERSIONS and INPUT_SUFFIXES lists below, then run all cells.
"""

from __future__ import annotations
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from datetime import datetime
import importlib.util
import sys


# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M"])
VERSIONS = ["3M", "12M", "60M"]  # <-- EDIT THIS LIST

# List of input suffixes to process (empty string for base folds)
# This script will process the specified stages: base, _with_graph, and/or _with_graph_and_metamodels
# NOTE: Only process base and _with_graph if metamodel backfill is not yet complete
INPUT_SUFFIXES = ["", "_with_graph"]  # <-- EDIT THIS LIST (currently: base and _with_graph only)

INPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "CUSTOM"
WRITE_MODE = "overwrite"
VERBOSE = True
SKIP_EXISTING_FOLDS = True  # Set to True to skip folds that already have all safe delay features


# -------------------------
# SAFE DELAY COMPUTATION
# -------------------------
def compute_safe_delay_features(df, verbose=True):
    """
    Compute safe_prev_departure_delay and safe_prev_arrival_delay features for previous flight.
    
    Logic for both features:
    (1) If previous flight already occurred (actual <= cutoff): use actual delay
    (2) If previous flight hasn't occurred yet (actual > cutoff):
        - If scheduled < cutoff: use (cutoff - scheduled) in minutes (at least delayed until now)
        - If scheduled >= cutoff: use 0 (scheduled is in future, no delay yet)
    (3) Otherwise: fallback to 0
    
    Returns:
        DataFrame with safe_prev_departure_delay and safe_prev_arrival_delay columns added
    """
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing safe delay features (departure and arrival)...")
    
    # Check if required columns exist
    required_cols = [
        'prediction_cutoff_timestamp',
        'prev_flight_dep_delay',
        'prev_flight_arr_delay',
        'has_leakage_prev_flight_actual_dep_time',
        'has_leakage_prev_flight_actual_arr_time',
        'prev_flight_sched_dep_timestamp'
    ]
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for safe delay computation: {missing_cols}. "
            f"Make sure the folds have flight lineage features (from split.py)."
        )
    
    # Convert previous flight's scheduled arrival to timestamp if not already present
    if 'prev_flight_sched_arr_timestamp' not in df.columns:
        df = df.withColumn(
            'prev_flight_sched_arr_timestamp',
            F.when(
                (col('prev_flight_crs_arr_time').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
                F.to_timestamp(
                    F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_crs_arr_time').cast('string'), 4, '0')),
                    'yyyy-MM-ddHHmm'
                )
            ).otherwise(None)
        )
    
    # Compute safe previous flight departure delay
    df = df.withColumn(
        'safe_prev_departure_delay',
        F.when(
            # Case 1: Previous flight already departed (actual_dep <= cutoff) - use actual delay
            (col('prev_flight_dep_delay').isNotNull()) & 
            (~col('has_leakage_prev_flight_actual_dep_time')),  # No leakage = already departed
            col('prev_flight_dep_delay')
        ).otherwise(
            # Case 2: Previous flight hasn't departed yet (actual_dep > cutoff or NULL)
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & 
                (col('prev_flight_sched_dep_timestamp').isNotNull()),
                F.when(
                    # Scheduled departure is before cutoff: compute delay as (cutoff - scheduled_dep) in minutes
                    col('prev_flight_sched_dep_timestamp') < col('prediction_cutoff_timestamp'),
                    # Calculate difference in minutes: (cutoff - scheduled_dep)
                    (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
                     F.unix_timestamp(col('prev_flight_sched_dep_timestamp'))) / 60.0
                ).otherwise(
                    # Scheduled departure is after cutoff: no delay yet (scheduled is in future)
                    lit(0.0)
                )
            ).otherwise(
                # Fallback: if we can't compute, use 0 (no delay information available)
                lit(0.0)
            )
        )
    )
    
    # Compute safe previous flight arrival delay
    df = df.withColumn(
        'safe_prev_arrival_delay',
        F.when(
            # Case 1: Previous flight already arrived (actual_arr <= cutoff) - use actual delay
            (col('prev_flight_arr_delay').isNotNull()) & 
            (~col('has_leakage_prev_flight_actual_arr_time')),  # No leakage = already arrived
            col('prev_flight_arr_delay')
        ).otherwise(
            # Case 2: Previous flight hasn't arrived yet (actual_arr > cutoff or NULL)
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & 
                (col('prev_flight_sched_arr_timestamp').isNotNull()),
                F.when(
                    # Scheduled arrival is before cutoff: compute delay as (cutoff - scheduled_arr) in minutes
                    col('prev_flight_sched_arr_timestamp') < col('prediction_cutoff_timestamp'),
                    # Calculate difference in minutes: (cutoff - scheduled_arr)
                    (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
                     F.unix_timestamp(col('prev_flight_sched_arr_timestamp'))) / 60.0
                ).otherwise(
                    # Scheduled arrival is after cutoff: no delay yet (scheduled is in future)
                    lit(0.0)
                )
            ).otherwise(
                # Fallback: if we can't compute, use 0 (no delay information available)
                lit(0.0)
            )
        )
    )
    
    # Compute safe_time_since_prev_arrival
    # This measures the time between previous flight's arrival and the prediction cutoff
    # Logic:
    # (1) If actual prev arrival is before cutoff: use (cutoff - actual_arrival) in minutes
    #     This is the actual time that has passed since arrival
    # (2) If actual arrival is after cutoff (hasn't arrived yet):
    #     - If scheduled arrival is before cutoff: use 0
    #       Reason: Flight is delayed, earliest it can arrive is right now (at cutoff)
    #       So time since arrival = 0 (it arrives right now, no time has passed yet)
    #     - If scheduled arrival is after cutoff: use (cutoff - scheduled_arrival) in minutes (negative)
    #       This is time until scheduled arrival (negative value)
    df = df.withColumn(
        'safe_time_since_prev_arrival',
        F.when(
            # Case 1: Previous flight already arrived (actual_arr <= cutoff)
            (col('prev_flight_actual_arr_time').isNotNull()) & 
            (col('prev_flight_fl_date').isNotNull()) &
            (~col('has_leakage_prev_flight_actual_arr_time')),  # No leakage = already arrived
            # Use (cutoff - actual_arrival) in minutes
            (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
             F.unix_timestamp(
                 F.to_timestamp(
                     F.concat(col('prev_flight_fl_date'), 
                             F.lpad(col('prev_flight_actual_arr_time').cast('string'), 4, '0')),
                     'yyyy-MM-ddHHmm'
                 )
             )) / 60.0
        ).otherwise(
            # Case 2: Previous flight hasn't arrived yet (actual_arr > cutoff or NULL)
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & 
                (col('prev_flight_sched_arr_timestamp').isNotNull()),
                F.when(
                    # Scheduled arrival is before cutoff: use 0 (assume it arrives right now at cutoff)
                    col('prev_flight_sched_arr_timestamp') < col('prediction_cutoff_timestamp'),
                    lit(0.0)
                ).otherwise(
                    # Scheduled arrival is after cutoff: use (cutoff - scheduled_arrival) in minutes (negative)
                    (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
                     F.unix_timestamp(col('prev_flight_sched_arr_timestamp'))) / 60.0
                )
            ).otherwise(
                # Fallback: if we can't compute, use 0
                lit(0.0)
            )
        )
    )
    
    # Impute NULLs to 0 (for first flights with no previous flight)
    df = df.withColumn('safe_prev_departure_delay',
                       F.coalesce(col('safe_prev_departure_delay'), lit(0.0)))
    df = df.withColumn('safe_prev_arrival_delay',
                       F.coalesce(col('safe_prev_arrival_delay'), lit(0.0)))
    df = df.withColumn('safe_time_since_prev_arrival',
                       F.coalesce(col('safe_time_since_prev_arrival'), lit(0.0)))
    
    if verbose:
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Count non-zero values for validation
        total = df.count()
        dep_non_zero = df.filter(col('safe_prev_departure_delay') != 0.0).count()
        arr_non_zero = df.filter(col('safe_prev_arrival_delay') != 0.0).count()
        time_since_non_zero = df.filter(col('safe_time_since_prev_arrival') != 0.0).count()
        dep_null_count = df.filter(col('safe_prev_departure_delay').isNull()).count()
        arr_null_count = df.filter(col('safe_prev_arrival_delay').isNull()).count()
        time_since_null_count = df.filter(col('safe_time_since_prev_arrival').isNull()).count()
        
        print(f"[{timestamp}] ✓ Safe delay features computed! (took {duration})")
        print(f"  Total rows: {total:,}")
        print(f"  safe_prev_departure_delay - Non-zero: {dep_non_zero:,} ({dep_non_zero/total*100:.1f}%)")
        print(f"  safe_prev_arrival_delay - Non-zero: {arr_non_zero:,} ({arr_non_zero/total*100:.1f}%)")
        print(f"  safe_time_since_prev_arrival - Non-zero: {time_since_non_zero:,} ({time_since_non_zero/total*100:.1f}%)")
        if dep_null_count > 0 or arr_null_count > 0 or time_since_null_count > 0:
            print(f"  ⚠ Warning: {dep_null_count:,} dep NULLs, {arr_null_count:,} arr NULLs, {time_since_null_count:,} time_since NULLs (should be 0 after imputation)")
    
    return df


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


def fold_has_feature(version: str, fold_idx: int, input_suffix: str, fold_type: str):
    """Check if a fold already has all safe delay features."""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    base_name = f"OTPW_{SOURCE}_{version}{input_suffix}"
    train_name = f"{base_name}_FOLD_{fold_idx}_TRAIN"
    train_path = f"{INPUT_FOLDER}/{train_name}.parquet"
    
    try:
        train_df = spark.read.parquet(train_path)
        return ("safe_prev_departure_delay" in train_df.columns and 
                "safe_prev_arrival_delay" in train_df.columns and
                "safe_time_since_prev_arrival" in train_df.columns)
    except:
        return False


def save_fold_with_suffix(version: str, fold_idx: int, train_df: DataFrame, val_or_test_df: DataFrame, 
                          output_suffix: str, fold_type: str):
    """Save fold with output suffix (same as input suffix for in-place update)."""
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
# MAIN FUNCTION
# -------------------------
def add_safe_departure_delay_to_folds(version: str, input_suffix: str = ""):
    """Add safe_prev_departure_delay and safe_prev_arrival_delay features to all folds (in-place update)."""
    spark = SparkSession.builder.getOrCreate()
    
    print(f"\n{'='*80}")
    print(f"ADDING safe delay features to {version} (suffix: '{input_suffix or '(none - base folds)'}')")
    print(f"{'='*80}")
    
    # Load all available folds (same pattern as cv.py dataloader)
    folds = load_folds_for_version(version, input_suffix)
    
    if not folds:
        print(f"  ⚠ No folds found for version {version} with suffix '{input_suffix}'")
        print(f"  Expected pattern: OTPW_{SOURCE}_{version}{input_suffix}_FOLD_*_TRAIN.parquet")
        return
    
    print(f"  Found {len(folds)} folds")
    
    if SKIP_EXISTING_FOLDS:
        print(f"  ⏭ Skip existing folds: ENABLED (will skip folds that already have all safe delay features)")
    else:
        print(f"  ⏭ Skip existing folds: DISABLED (will re-run all folds)")
    
    processed_count = 0
    skipped_count = 0
    
    for fold_idx, (train_df, val_or_test_df, fold_type) in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}/{len(folds)} ({fold_type})")
        print(f"{'='*60}")
        
        # Check if fold already has the features (resume mode)
        if SKIP_EXISTING_FOLDS and fold_has_feature(version, fold_idx, input_suffix, fold_type):
            if VERBOSE:
                print(f"  ⏭ Skipping fold {fold_idx} (already has safe delay features)")
            print(f"✅ Fold {fold_idx} skipped (already complete)")
            skipped_count += 1
            continue
        
        if VERBOSE:
            train_count = train_df.count()
            val_or_test_count = val_or_test_df.count()
            print(f"  Train: {train_count:,} rows")
            print(f"  {fold_type}: {val_or_test_count:,} rows")
        
        # Add safe delay features
        train_with_feature = compute_safe_delay_features(train_df, verbose=VERBOSE)
        val_or_test_with_feature = compute_safe_delay_features(val_or_test_df, verbose=VERBOSE)
        
        # Save (in-place update - same suffix as input)
        if VERBOSE:
            save_start = datetime.now()
            save_timestamp = save_start.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n💾 [{save_timestamp}] Saving fold {fold_idx} (in-place update)...")
        
        save_fold_with_suffix(version, fold_idx, train_with_feature, val_or_test_with_feature, 
                             input_suffix, fold_type)
        
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
        print(f"   Skipped: {skipped_count} folds (already had safe delay features)")
    print(f"   Suffix: '{input_suffix or '(none - base folds)'}'")
    print(f"   Use version='{version}' with suffix='{input_suffix}' in cv.py")
    print(f"{'='*80}")


# -------------------------
# MAIN
# -------------------------
# Process all versions and suffixes
total_versions = len(VERSIONS)
total_suffixes = len(INPUT_SUFFIXES)
total_tasks = total_versions * total_suffixes
current_task = 0

print("\n" + "="*80)
print(f"SAFE DELAY FEATURES BACKFILL")
print("="*80)
print(f"Versions: {VERSIONS}")
print(f"Suffixes: {INPUT_SUFFIXES}")
print(f"Total tasks: {total_tasks} (versions × suffixes)")
print("="*80)

for version in VERSIONS:
    for input_suffix in INPUT_SUFFIXES:
        current_task += 1
        print(f"\n{'#'*80}")
        print(f"TASK {current_task}/{total_tasks}: {version} with suffix '{input_suffix or '(none)'}'")
        print(f"{'#'*80}")
        
        try:
            add_safe_departure_delay_to_folds(version, input_suffix)
        except Exception as e:
            print(f"\n❌ ERROR processing {version} with suffix '{input_suffix}': {str(e)}")
            print(f"   Continuing with next task...")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print("✅ All versions and suffixes processed!")
print("="*80)

