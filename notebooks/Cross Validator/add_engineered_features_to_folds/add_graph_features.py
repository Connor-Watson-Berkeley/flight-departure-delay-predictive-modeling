"""
add_graph_features.py - Add graph features (PageRank) to existing folds

This notebook reads folds created by split.py (which includes flight lineage features) and adds
graph features (PageRank scores for airports), saving to a new path with suffix _with_graph.

Usage:
    Set VERSIONS list below, then run all cells.
"""

from __future__ import annotations
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, broadcast
from datetime import datetime
from graphframes import GraphFrame
import importlib.util
import sys


# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M", "XM"])
VERSIONS = ["12M"]  # <-- EDIT THIS LIST

INPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "OTPW"  # Change to "CUSTOM" for CUSTOM data, "OTPW" for OTPW data
WRITE_MODE = "overwrite"
VERBOSE = True
OUTPUT_SUFFIX = "_with_graph"
INPUT_SUFFIX = ""  # Input suffix (default: '' for base folds)
SKIP_EXISTING_FOLDS = True  # If True, skip folds where all graph features already exist 
                             # AND have >97% non-null values (catches bugs where columns were created but not populated).
                             # If any are missing or have too many NULLs, the fold will be re-processed to add/fix the missing features.


# -------------------------
# GRAPH FEATURES COMPUTATION
# -------------------------


def compute_pagerank_features(train_df, val_df, 
                              origin_col="origin", dest_col="dest",
                              reset_probability=0.15, max_iter=10,
                              checkpoint_dir="dbfs:/tmp/graphframes_checkpoint",
                              verbose=True):
    """
    Compute PageRank features on training data and join to both train and val DataFrames.
    
    Uses GraphFrames for both weighted and unweighted PageRank.
    
    Returns:
        tuple: (train_df_with_features, val_df_with_features)
    """
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing graph features (train: {train_df.count():,} rows)...")
    
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    
    # Build graph from training data only (CV-safe)
    edges = (
        train_df
        .select(origin_col, dest_col)
        .filter(
            col(origin_col).isNotNull() & 
            col(dest_col).isNotNull()
        )
        .groupBy(origin_col, dest_col)
        .count()
        .withColumnRenamed(origin_col, "src")
        .withColumnRenamed(dest_col, "dst")
        .withColumnRenamed("count", "weight")
    )
    
    # Set checkpoint directory
    sc.setCheckpointDir(checkpoint_dir)
    
    # Get vertices (all unique airports)
    src_airports = edges.select(col("src").alias("id")).distinct()
    dst_airports = edges.select(col("dst").alias("id")).distinct()
    vertices = src_airports.union(dst_airports).distinct()
    
    # Compute unweighted PageRank using GraphFrames
    if verbose:
        print("  Computing unweighted PageRank (GraphFrames)...")
    
    edges_unweighted = edges.select("src", "dst")
    g_unweighted = GraphFrame(vertices, edges_unweighted)
    pagerank_unweighted_result = g_unweighted.pageRank(
        resetProbability=reset_probability,
        maxIter=max_iter
    )
    
    pagerank_unweighted_df = pagerank_unweighted_result.vertices.select(
        col("id").alias("airport"),
        col("pagerank").alias("pagerank_unweighted")
    )
    
    # Compute weighted PageRank using GraphFrames (by duplicating edges based on weight)
    if verbose:
        print("  Computing weighted PageRank (GraphFrames with edge duplication)...")
    
    # Duplicate edges based on weight: create array [0, 1, ..., weight-1] and explode
    edges_weighted = (
        edges
        .withColumn("seq", F.sequence(F.lit(0), col("weight").cast("int") - 1))
        .select("src", "dst", F.explode("seq").alias("_"))
        .select("src", "dst")
    )
    
    g_weighted = GraphFrame(vertices, edges_weighted)
    pagerank_weighted_result = g_weighted.pageRank(
        resetProbability=reset_probability,
        maxIter=max_iter
    )
    
    pagerank_weighted_df = pagerank_weighted_result.vertices.select(
        col("id").alias("airport"),
        col("pagerank").alias("pagerank_weighted")
    )
    
    # Combine PageRank scores
    pagerank_scores = pagerank_unweighted_df.join(pagerank_weighted_df, "airport", "outer")
    broadcast_scores = broadcast(pagerank_scores)
    
    # Join origin and dest features
    train_with_base = (
        train_df
        .join(broadcast_scores, col(origin_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
        .drop("airport")
        .join(broadcast_scores, col(dest_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
        .drop("airport")
        .cache()
    )
    
    val_with_base = (
        val_df
        .join(broadcast_scores, col(origin_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
        .drop("airport")
        .join(broadcast_scores, col(dest_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
        .drop("airport")
        .cache()
    )
    
    # Trigger cache materialization
    _ = train_with_base.count()
    _ = val_with_base.count()
    
    # Add prev_flight graph features if prev_flight columns exist
    train_with_features = train_with_base
    val_with_features = val_with_base
    
    # Check if prev_flight_origin exists in both train and val
    if "prev_flight_origin" in train_with_base.columns and "prev_flight_origin" in val_with_base.columns:
        train_with_features = (
            train_with_base
            .join(broadcast_scores, col("prev_flight_origin") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
            .drop("airport")
        )
        val_with_features = (
            val_with_base
            .join(broadcast_scores, col("prev_flight_origin") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
            .drop("airport")
        )
    
    # Check if prev_flight_dest exists in both train and val
    if "prev_flight_dest" in train_with_features.columns and "prev_flight_dest" in val_with_features.columns:
        train_with_features = (
            train_with_features
            .join(broadcast_scores, col("prev_flight_dest") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_dest_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_dest_pagerank_unweighted")
            .drop("airport")
        )
        val_with_features = (
            val_with_features
            .join(broadcast_scores, col("prev_flight_dest") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_dest_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_dest_pagerank_unweighted")
            .drop("airport")
        )
    
    # Fill NULL PageRank values with 0
    pagerank_cols = [
        "origin_pagerank_weighted", "origin_pagerank_unweighted",
        "dest_pagerank_weighted", "dest_pagerank_unweighted"
    ]
    if "prev_flight_origin_pagerank_weighted" in train_with_features.columns:
        pagerank_cols.extend([
            "prev_flight_origin_pagerank_weighted", "prev_flight_origin_pagerank_unweighted"
        ])
    if "prev_flight_dest_pagerank_weighted" in train_with_features.columns:
        pagerank_cols.extend([
            "prev_flight_dest_pagerank_weighted", "prev_flight_dest_pagerank_unweighted"
        ])
    
    fill_dict = {col_name: 0.0 for col_name in pagerank_cols}
    train_with_features = train_with_features.fillna(fill_dict)
    val_with_features = val_with_features.fillna(fill_dict)
    
    # Cache final DataFrames
    train_with_features = train_with_features.cache()
    val_with_features = val_with_features.cache()
    
    # Safety check and validation
    if verbose:
        train_total = train_with_features.count()
        val_total = val_with_features.count()
        train_missing = train_with_features.filter(
            col("origin_pagerank_weighted") == 0.0
        ).count()
        val_missing = val_with_features.filter(
            col("origin_pagerank_weighted") == 0.0
        ).count()
        
        if train_missing > 0 or val_missing > 0:
            print(f"  ⚠ Warning: {train_missing:,}/{train_total:,} train rows and {val_missing:,}/{val_total:,} val rows have missing airports (filled with 0.0)")
            if val_missing > val_total * 0.1:
                print(f"  ⚠ WARNING: High percentage of missing airports in validation data!")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Graph features computed! (took {duration})")
    
    return train_with_features, val_with_features


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def fold_exists(version: str, fold_idx: int, output_suffix: str, fold_type: str):
    """
    Check if a fold already exists in the output folder AND has all expected graph features.
    
    Args:
        version: Data version (e.g., "12M")
        fold_idx: Fold index (1-4)
        output_suffix: Output suffix (e.g., "_with_graph")
        fold_type: "VAL" or "TEST"
    
    Returns:
        True if fold exists AND has all expected graph feature columns with >97% non-null values, False otherwise
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
        
        # Read DataFrames to check columns and non-null percentages
        try:
            train_df = spark.read.option("mergeSchema", "true").parquet(train_path)
            val_df = spark.read.option("mergeSchema", "true").parquet(val_path)
            train_cols = set(train_df.columns)
            val_cols = set(val_df.columns)
            
            # Expected graph feature columns (always present)
            expected_cols = [
                "origin_pagerank_weighted",
                "origin_pagerank_unweighted",
                "dest_pagerank_weighted",
                "dest_pagerank_unweighted"
            ]
            
            # Optional: prev_flight graph features (if prev_flight columns exist)
            if "prev_flight_origin" in train_cols and "prev_flight_origin" in val_cols:
                expected_cols.extend([
                    "prev_flight_origin_pagerank_weighted",
                    "prev_flight_origin_pagerank_unweighted"
                ])
            
            if "prev_flight_dest" in train_cols and "prev_flight_dest" in val_cols:
                expected_cols.extend([
                    "prev_flight_dest_pagerank_weighted",
                    "prev_flight_dest_pagerank_unweighted"
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
    except Exception as e:
        # Fallback: try to read schema (slower but more reliable)
        try:
            train_df = spark.read.option("mergeSchema", "true").parquet(train_path)
            val_df = spark.read.option("mergeSchema", "true").parquet(val_path)
            _ = train_df.schema
            _ = val_df.schema
            return True
        except:
            return False


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
    
    # Temporarily modify VERSIONS to load only the version we need
    original_versions = cv.VERSIONS
    cv.VERSIONS = [version]
    
    try:
        # Create a loader instance with the input suffix and source (use local SOURCE, not cv.py's global)
        data_loader = cv.FlightDelayDataLoader(suffix=input_suffix, source=SOURCE)
        
        # Load all versions (in this case, just the one we need)
        data_loader.load()
        
        # Get folds for the requested version
        folds_raw = data_loader.get_version(version)
    finally:
        # Restore original VERSIONS
        cv.VERSIONS = original_versions
    
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


def save_fold_with_suffix(version: str, fold_idx: int, train_df: DataFrame, val_or_test_df: DataFrame, 
                          output_suffix: str, fold_type: str):
    """Save fold with output suffix."""
    base_name = f"OTPW_{SOURCE}_{version}{output_suffix}"
    
    train_name = f"{base_name}_FOLD_{fold_idx}_TRAIN"
    train_df.write.mode(WRITE_MODE).parquet(f"{OUTPUT_FOLDER}/{train_name}.parquet")
    
    if fold_type == "VAL":
        val_name = f"{base_name}_FOLD_{fold_idx}_VAL"
        val_or_test_df.write.mode(WRITE_MODE).parquet(f"{OUTPUT_FOLDER}/{val_name}.parquet")
    else:
        test_name = f"{base_name}_FOLD_{fold_idx}_TEST"
        val_or_test_df.write.mode(WRITE_MODE).parquet(f"{OUTPUT_FOLDER}/{test_name}.parquet")


# -------------------------
# MAIN FUNCTION
# -------------------------
def add_graph_features_to_folds(version: str, input_suffix: str = ""):
    """Add graph features (PageRank) to all folds."""
    spark = SparkSession.builder.getOrCreate()
    
    print(f"\n{'='*80}")
    print(f"ADDING GRAPH FEATURES to {version}")
    print(f"{'='*80}")
    print(f"Input suffix: {input_suffix or '(none - base folds)'}")
    print(f"Output suffix: {OUTPUT_SUFFIX}")
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
        print(f"  ⏭ Skip existing folds: ENABLED (will skip folds that already exist with valid graph features)")
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
                print(f"  ⏭ Skipping fold {fold_idx} (already exists with all expected graph features)")
            print(f"✅ Fold {fold_idx} skipped (already complete)")
            skipped_count += 1
            continue
        
        if VERBOSE:
            train_count = train_df.count()
            val_or_test_count = val_or_test_df.count()
            print(f"  Train: {train_count:,} rows")
            print(f"  {fold_type}: {val_or_test_count:,} rows")
        
        # Add graph features
        train_with_graph, val_or_test_with_graph = compute_pagerank_features(
            train_df, val_or_test_df, verbose=VERBOSE
        )
        
        # Save
        if VERBOSE:
            print(f"💾 Saving to {OUTPUT_SUFFIX}...")
        save_fold_with_suffix(version, fold_idx, train_with_graph, val_or_test_with_graph, 
                             OUTPUT_SUFFIX, fold_type)
        
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
    add_graph_features_to_folds(version, INPUT_SUFFIX)

print("\n" + "="*80)
print("✅ All versions processed successfully!")
print("="*80)

