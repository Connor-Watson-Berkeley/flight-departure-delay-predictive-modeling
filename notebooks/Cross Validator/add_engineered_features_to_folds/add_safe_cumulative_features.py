"""
add_safe_cumulative_features.py - Update cumulative features to exclude immediate previous flight

This script reads existing folds (base, _with_graph, or _with_graph_and_metamodels) and updates
cumulative features to use the safer implementation that excludes the immediate previous flight,
saving back to the same location (in-place update).

Features updated:
1. lineage_cumulative_delay: Sum of delays from flights BEFORE immediate previous flight
2. lineage_num_previous_flights: Count of flights BEFORE immediate previous flight
3. lineage_avg_delay_previous_flights: Average delay of flights BEFORE immediate previous flight
4. lineage_max_delay_previous_flights: Maximum delay of flights BEFORE immediate previous flight

The new implementation uses rowsBetween(Window.unboundedPreceding, -2) instead of -1,
which excludes the immediate previous flight (n-1) and only looks at flights 1, 2, 3, etc.
This dramatically reduces data leakage risk since flights before n-1 have almost certainly
completed by the time we're making predictions.

Usage:
    Set VERSIONS and INPUT_SUFFIXES lists below, then run all cells.
"""

from __future__ import annotations
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from datetime import datetime
import importlib.util
import sys


# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M"])
VERSIONS = ["12M", "60M", "3M"]  # <-- EDIT THIS LIST

# List of input suffixes to process (empty string for base folds)
# This script will process the specified stages: "", "_with_graph", and/or "_with_graph_and_metamodels"
INPUT_SUFFIXES = ["_with_graph_and_metamodels"]  # <-- EDIT THIS LIST

INPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "CUSTOM"
WRITE_MODE = "overwrite"
VERBOSE = True
SKIP_EXISTING_FOLDS = True  # Set to True to skip folds that already have the safe cumulative features


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def _find_tail_num_column(df):
    """Find the tail_num column (aircraft identifier) in the DataFrame."""
    tail_num_candidates = ['tail_num', 'TAIL_NUM', 'tail_number', 'TAIL_NUMBER', 'op_unique_carrier_tail_num']
    
    for candidate in tail_num_candidates:
        if candidate in df.columns:
            return candidate
    
    # Try pattern matching
    tail_cols = [c for c in df.columns if 'tail' in c.lower() and 'num' in c.lower()]
    if tail_cols:
        return tail_cols[0]
    
    raise ValueError(f"Could not find tail_num column. Available columns: {df.columns[:20]}...")


# -------------------------
# SAFE CUMULATIVE FEATURES COMPUTATION
# -------------------------
def compute_safe_cumulative_features(df, verbose=True):
    """
    Recompute cumulative features to exclude the immediate previous flight (n-1).
    
    Uses rowsBetween(Window.unboundedPreceding, -2) instead of -1 to exclude flight n-1.
    This means:
    - Flight 1: No previous flights → NULL/0 (window is empty)
    - Flight 2: Only flight 1 exists, but -2 excludes it → NULL/0 (window is empty)
    - Flight 3: Flights 1 and 2 exist, -2 excludes flight 2 → Only flight 1 is included
    - Flight 4: Flights 1, 2, 3 exist, -2 excludes flight 3 → Flights 1 and 2 are included
    - etc.
    
    This dramatically reduces data leakage since flights before n-1 have almost certainly
    completed by prediction time (it would be insane if a flight made 2 full trips in 2 hours).
    
    Returns:
        DataFrame with updated cumulative features
    """
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Recomputing cumulative features (excluding immediate previous flight)...")
    
    # Check if required columns exist
    # Note: Folds may have DEP_DELAY (uppercase) as the label, but we need dep_delay (lowercase) for cumulative features
    # Check for both possible column names
    dep_delay_col = None
    if 'dep_delay' in df.columns:
        dep_delay_col = 'dep_delay'
    elif 'DEP_DELAY' in df.columns:
        dep_delay_col = 'DEP_DELAY'
    else:
        raise ValueError(
            f"Missing required column: 'dep_delay' or 'DEP_DELAY'. "
            f"Available columns: {df.columns[:20]}... "
            f"Make sure the folds have flight lineage features (from split.py)."
        )
    
    if 'arrival_timestamp' not in df.columns:
        raise ValueError(
            f"Missing required column: 'arrival_timestamp'. "
            f"Available columns: {df.columns[:20]}... "
            f"Make sure the folds have flight lineage features (from split.py)."
        )
    
    # Find tail_num column
    tail_num_col = _find_tail_num_column(df)
    if verbose:
        print(f"  Using tail_num column: {tail_num_col}")
        print(f"  Using delay column: {dep_delay_col}")
    
    # Create window specification: partition by tail_num, order by arrival_timestamp
    # Use -2 instead of -1 to exclude the immediate previous flight
    window_spec_cumulative = Window.partitionBy(tail_num_col).orderBy(
        F.col('arrival_timestamp').asc_nulls_last()
    ).rowsBetween(Window.unboundedPreceding, -2)
    
    # Recompute cumulative features
    # NOTE: withColumn REPLACES existing columns if they exist (no risk of duplication)
    # Use the detected column name (dep_delay or DEP_DELAY)
    df = df.withColumn('lineage_cumulative_delay', F.sum(dep_delay_col).over(window_spec_cumulative))
    df = df.withColumn('lineage_num_previous_flights', F.count('*').over(window_spec_cumulative))
    df = df.withColumn('lineage_avg_delay_previous_flights', F.avg(dep_delay_col).over(window_spec_cumulative))
    df = df.withColumn('lineage_max_delay_previous_flights', F.max(dep_delay_col).over(window_spec_cumulative))
    
    # Impute NULLs (for first/second flights with no flights before n-1)
    # When window is empty (no flights 2 lags away), window functions return NULL
    # - Cumulative delay: 0 (no previous flights = no cumulative delay)
    # - Num previous flights: 0 (no previous flights = 0 count)
    # - Avg delay: Use mean delay (9.37) as default when no previous flights exist
    # - Max delay: 0 (no previous flights = no max delay)
    MEAN_DELAY = 9.37  # Mean departure delay across all flights
    df = df.withColumn('lineage_cumulative_delay',
                       F.coalesce(col('lineage_cumulative_delay'), lit(0.0)))
    df = df.withColumn('lineage_num_previous_flights',
                       F.coalesce(col('lineage_num_previous_flights'), lit(0)))
    df = df.withColumn('lineage_avg_delay_previous_flights',
                       F.coalesce(col('lineage_avg_delay_previous_flights'), lit(MEAN_DELAY)))
    df = df.withColumn('lineage_max_delay_previous_flights',
                       F.coalesce(col('lineage_max_delay_previous_flights'), lit(0.0)))
    
    if verbose:
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Count non-zero values for validation
        total = df.count()
        cum_delay_non_zero = df.filter(col('lineage_cumulative_delay') != 0.0).count()
        num_flights_non_zero = df.filter(col('lineage_num_previous_flights') != 0).count()
        avg_delay_non_zero = df.filter(col('lineage_avg_delay_previous_flights') != 0.0).count()
        max_delay_non_zero = df.filter(col('lineage_max_delay_previous_flights') != 0.0).count()
        
        # Check for NULLs (should be 0 after imputation)
        cum_delay_null = df.filter(col('lineage_cumulative_delay').isNull()).count()
        num_flights_null = df.filter(col('lineage_num_previous_flights').isNull()).count()
        avg_delay_null = df.filter(col('lineage_avg_delay_previous_flights').isNull()).count()
        max_delay_null = df.filter(col('lineage_max_delay_previous_flights').isNull()).count()
        
        print(f"[{timestamp}] ✓ Cumulative features recomputed! (took {duration})")
        print(f"  Total rows: {total:,}")
        print(f"  lineage_cumulative_delay - Non-zero: {cum_delay_non_zero:,} ({cum_delay_non_zero/total*100:.1f}%)")
        print(f"  lineage_num_previous_flights - Non-zero: {num_flights_non_zero:,} ({num_flights_non_zero/total*100:.1f}%)")
        print(f"  lineage_avg_delay_previous_flights - Non-zero: {avg_delay_non_zero:,} ({avg_delay_non_zero/total*100:.1f}%)")
        print(f"  lineage_max_delay_previous_flights - Non-zero: {max_delay_non_zero:,} ({max_delay_non_zero/total*100:.1f}%)")
        if cum_delay_null > 0 or num_flights_null > 0 or avg_delay_null > 0 or max_delay_null > 0:
            print(f"  ⚠ Warning: {cum_delay_null:,} cum_delay NULLs, {num_flights_null:,} num_flights NULLs, "
                  f"{avg_delay_null:,} avg_delay NULLs, {max_delay_null:,} max_delay NULLs (should be 0 after imputation)")
    
    return df


# -------------------------
# LOAD/SAVE FUNCTIONS
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


def fold_has_safe_cumulative_features(version: str, fold_idx: int, input_suffix: str, fold_type: str):
    """
    Check if a fold already has the safe cumulative features.
    
    We can't easily detect if the features were computed with -2 vs -1 from the data alone,
    so we'll check if the fold has been processed by checking a marker or by re-running.
    For now, we'll just check if the features exist (they should always exist from split.py).
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    base_name = f"OTPW_{SOURCE}_{version}{input_suffix}"
    train_name = f"{base_name}_FOLD_{fold_idx}_TRAIN"
    train_path = f"{INPUT_FOLDER}/{train_name}.parquet"
    
    try:
        train_df = spark.read.parquet(train_path)
        # Check if all cumulative features exist
        required_features = [
            'lineage_cumulative_delay',
            'lineage_num_previous_flights',
            'lineage_avg_delay_previous_flights',
            'lineage_max_delay_previous_flights'
        ]
        return all(feat in train_df.columns for feat in required_features)
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
def update_cumulative_features_to_safe(version: str, input_suffix: str = ""):
    """Update cumulative features to exclude immediate previous flight (in-place update)."""
    spark = SparkSession.builder.getOrCreate()
    
    print(f"\n{'='*80}")
    print(f"UPDATING cumulative features to safe implementation for {version} (suffix: '{input_suffix or '(none - base folds)'}')")
    print(f"{'='*80}")
    
    # Load all available folds (same pattern as cv.py dataloader)
    folds = load_folds_for_version(version, input_suffix)
    
    if not folds:
        print(f"  ⚠ No folds found for version {version} with suffix '{input_suffix}'")
        print(f"  Expected pattern: OTPW_{SOURCE}_{version}{input_suffix}_FOLD_*_TRAIN.parquet")
        return
    
    print(f"  Found {len(folds)} folds")
    
    if SKIP_EXISTING_FOLDS:
        print(f"  ⏭ Skip existing folds: ENABLED (will skip folds that already have cumulative features)")
        print(f"     Note: This checks if features exist, but cannot verify if they use -2 vs -1 window.")
        print(f"     To force re-computation, set SKIP_EXISTING_FOLDS = False")
    else:
        print(f"  ⏭ Skip existing folds: DISABLED (will re-run all folds)")
    
    processed_count = 0
    skipped_count = 0
    
    for fold_idx, (train_df, val_or_test_df, fold_type) in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}/{len(folds)} ({fold_type})")
        print(f"{'='*60}")
        
        # Check if fold already has the features (resume mode)
        # Note: We can't easily detect if features use -2 vs -1, so we'll always recompute
        # unless SKIP_EXISTING_FOLDS is True and features don't exist
        if SKIP_EXISTING_FOLDS and not fold_has_safe_cumulative_features(version, fold_idx, input_suffix, fold_type):
            if VERBOSE:
                print(f"  ⏭ Skipping fold {fold_idx} (cumulative features don't exist - may need to run split.py first)")
            print(f"⏭ Fold {fold_idx} skipped (features don't exist)")
            skipped_count += 1
            continue
        
        if VERBOSE:
            train_count = train_df.count()
            val_or_test_count = val_or_test_df.count()
            print(f"  Train: {train_count:,} rows")
            print(f"  {fold_type}: {val_or_test_count:,} rows")
        
        # Recompute cumulative features with safe implementation
        train_with_feature = compute_safe_cumulative_features(train_df, verbose=VERBOSE)
        val_or_test_with_feature = compute_safe_cumulative_features(val_or_test_df, verbose=VERBOSE)
        
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
        print(f"   Skipped: {skipped_count} folds (cumulative features didn't exist)")
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
print(f"SAFE CUMULATIVE FEATURES BACKFILL")
print("="*80)
print(f"Versions: {VERSIONS}")
print(f"Suffixes: {INPUT_SUFFIXES}")
print(f"Total tasks: {total_tasks} (versions × suffixes)")
print("="*80)
print("\nThis script updates cumulative features to exclude the immediate previous flight (n-1).")
print("Features now look at flights 1, 2, 3, etc. but NOT flight n-1, dramatically reducing")
print("data leakage risk since flights before n-1 have almost certainly completed.")
print("="*80)

for version in VERSIONS:
    for input_suffix in INPUT_SUFFIXES:
        current_task += 1
        print(f"\n{'#'*80}")
        print(f"TASK {current_task}/{total_tasks}: {version} with suffix '{input_suffix or '(none)'}'")
        print(f"{'#'*80}")
        
        try:
            update_cumulative_features_to_safe(version, input_suffix)
        except Exception as e:
            print(f"\n❌ ERROR processing {version} with suffix '{input_suffix}': {str(e)}")
            print(f"   Continuing with next task...")
            import traceback
            traceback.print_exc()

print("\n" + "="*80)
print("✅ All versions and suffixes processed!")
print("="*80)

