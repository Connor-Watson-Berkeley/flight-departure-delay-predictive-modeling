#!/usr/bin/env python3
"""
split.py (CUSTOM only, real paths, processes both 3M and 12M)

Reads:
  3M  from dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m
  12M from dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015

Writes folds to:
  dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed

Output format:
  OTPW_CUSTOM_{VERSION}_FOLD_{i}_TRAIN.parquet
  OTPW_CUSTOM_{VERSION}_FOLD_{i}_VAL.parquet   (for i=1..3)
  OTPW_CUSTOM_{VERSION}_FOLD_4_TEST.parquet    (last fold)
"""

from __future__ import annotations
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Load modules from our Databricks repo
import importlib.util

flight_lineage_features_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/flight_lineage_features.py"
spec = importlib.util.spec_from_file_location("flight_lineage_features", flight_lineage_features_path)
flight_lineage_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flight_lineage_features)

precompute_features_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/precompute_features.py"
spec = importlib.util.spec_from_file_location("precompute_features", precompute_features_path)
precompute_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute_features)
# Note: precompute_features.py now imports graph_features and meta_model_estimator directly


# -------------------------
# HARD-CODED SETTINGS
# -------------------------
DATASETS = {
    "3M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m",
    "12M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015",
    "60M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_/"
}

OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "CUSTOM"
DATE_COL = "FL_DATE"

N_FOLDS = 3              # CV folds
CREATE_TEST_FOLD = True  # adds one final test period
WRITE_MODE = "overwrite"
VERBOSE = True

# Feature precomputation settings
PRECOMPUTE_GRAPH_FEATURES = True  # Precompute graph features (PageRank)
PRECOMPUTE_META_MODELS = True     # Precompute meta-model predictions
META_MODEL_IDS = ["RF_1"]         # List of meta-model identifiers (e.g., ["RF_1", "RF_2", "XGB_1"])


# -------------------------
# FOLD CREATION
# -------------------------
def create_sliding_window_folds(
    df: DataFrame,
    date_col: str = "FL_DATE",
    n_folds: int = 3,
    test_fold: bool = True,
    verbose: bool = True,
):
    """Create sliding-window CV folds after removing null DEP_DELAY."""

    # Normalize FL_DATE to DateType
    df = df.withColumn(date_col, F.to_date(F.col(date_col)))

    # Drop null labels
    df_clean = df.filter(F.col("DEP_DELAY").isNotNull())

    if verbose:
        initial = df.count()
        cleaned = df_clean.count()
        removed = initial - cleaned
        print(
            f"⚠ Removed {removed:,} rows with null DEP_DELAY "
            f"({removed/initial*100:.2f}%). Remaining: {cleaned:,}"
        )

    # Cast label to double for Spark ML
    df_clean = df_clean.withColumn("DEP_DELAY", F.col("DEP_DELAY").cast("double"))

    # Binary labels
    df_clean = df_clean.withColumn(
        "SEVERE_DEL60",
        F.when(F.col("DEP_DELAY") >= 60, 1).otherwise(0)
    )
    df_clean = df_clean.withColumn(
        "DEP_DEL15",
        F.when(F.col("DEP_DELAY") >= 15, 1).otherwise(0)
    )

    # Auto-detect date range
    r = df_clean.select(
        F.min(F.col(date_col)).alias("start"),
        F.max(F.col(date_col)).alias("end")
    ).first()

    start_date = r["start"].strftime("%Y-%m-%d")
    end_date = r["end"].strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_dt - start_dt).days
    n_periods = n_folds + 1 if test_fold else n_folds
    period_days = total_days // n_periods

    if period_days <= 0:
        raise ValueError("Not enough days to split into folds.")

    if verbose:
        print(f"Date range: {start_date} → {end_date}")
        print(f"Total days: {total_days}, period_days: {period_days}, periods: {n_periods}")

    # Build periods, last period takes remainder
    periods = []
    for i in range(n_periods):
        p_start = start_dt + timedelta(days=i * period_days)
        p_end = end_dt if i == n_periods - 1 else start_dt + timedelta(days=(i + 1) * period_days)
        periods.append((p_start, p_end))

    folds = []

    # CV folds
    for f in range(n_folds):
        t_start, t_end = periods[f]
        v_start, v_end = periods[f + 1]

        train_df = df_clean.filter(
            (F.col(date_col) >= F.lit(t_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <  F.lit(t_end.strftime("%Y-%m-%d")))
        )
        val_df = df_clean.filter(
            (F.col(date_col) >= F.lit(v_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <  F.lit(v_end.strftime("%Y-%m-%d")))
        )
        
        # CRITICAL: Cache filtered DataFrames before adding features
        # This prevents recomputing the lineage join chain on every operation
        # The filter operation creates a new DataFrame, so we need to cache it
        train_df = train_df.cache()
        val_df = val_df.cache()
        
        # Add precomputed features (graph + meta-models) if enabled
        # Use warm start: previous fold's PageRank scores to speed up convergence
        warm_start_scores = None
        if PRECOMPUTE_GRAPH_FEATURES or PRECOMPUTE_META_MODELS:
            if verbose:
                # Trigger cache materialization and get counts for logging (single count, not double)
                train_count = train_df.count()
                val_count = val_df.count()
                print(f"\n{'='*60}")
                print(f"Computing engineered features for {version} Fold {f+1}")
                print(f"  Train: [{t_start:%Y-%m-%d} to {t_end:%Y-%m-%d}), {train_count:,} rows")
                print(f"  Val:   [{v_start:%Y-%m-%d} to {v_end:%Y-%m-%d}), {val_count:,} rows")
                print(f"{'='*60}")
            train_df, val_df, warm_start_scores = precompute_features.add_precomputed_features(
                train_df, val_df,
                model_ids=META_MODEL_IDS,
                compute_graph=PRECOMPUTE_GRAPH_FEATURES,
                compute_meta_models=PRECOMPUTE_META_MODELS,
                verbose=verbose,
                warm_start_scores=warm_start_scores  # Use previous fold's scores (None for first fold)
            )
        
        folds.append((train_df, val_df))

        if verbose:
            print(
                f"Fold {f+1}: Train [{t_start:%Y-%m-%d} to {t_end:%Y-%m-%d}), "
                f"Val [{v_start:%Y-%m-%d} to {v_end:%Y-%m-%d}) | "
                f"Train={train_df.count():,}, Val={val_df.count():,}"
            )

    # Test fold
    if test_fold:
        test_start, test_end = periods[-1]

        combined_train = df_clean.filter(
            (F.col(date_col) >= F.lit(start_date)) &
            (F.col(date_col) <  F.lit(test_start.strftime("%Y-%m-%d")))
        )
        test_df = df_clean.filter(
            (F.col(date_col) >= F.lit(test_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <= F.lit(test_end.strftime("%Y-%m-%d")))
        )
        
        # CRITICAL: Cache filtered DataFrames before adding features
        # This prevents recomputing the lineage join chain on every operation
        combined_train = combined_train.cache()
        test_df = test_df.cache()

        # Add precomputed features (graph + meta-models) if enabled
        # Use warm start from last CV fold (or None if no CV folds)
        if PRECOMPUTE_GRAPH_FEATURES or PRECOMPUTE_META_MODELS:
            if verbose:
                # Trigger cache materialization and get counts for logging (single count, not double)
                train_count = combined_train.count()
                test_count = test_df.count()
                print(f"\n{'='*60}")
                print(f"Computing engineered features for {version} Test Fold")
                print(f"  Train: [{start_date} to {test_start:%Y-%m-%d}), {train_count:,} rows")
                print(f"  Test:  [{test_start:%Y-%m-%d} to {test_end:%Y-%m-%d}), {test_count:,} rows")
                print(f"{'='*60}")
            combined_train, test_df, _ = precompute_features.add_precomputed_features(
                combined_train, test_df,
                model_ids=META_MODEL_IDS,
                compute_graph=PRECOMPUTE_GRAPH_FEATURES,
                compute_meta_models=PRECOMPUTE_META_MODELS,
                verbose=verbose,
                warm_start_scores=warm_start_scores  # Use last CV fold's scores
            )
        
        folds.append((combined_train, test_df))

        if verbose:
            print(
                f"Test Fold: Train [{start_date} to {test_start:%Y-%m-%d}), "
                f"Test [{test_start:%Y-%m-%d} to {test_end:%Y-%m-%d}) | "
                f"Train={combined_train.count():,}, Test={test_df.count():,}"
            )

    return folds


# -------------------------
# SAVE FOLDS
# -------------------------
def save_folds_to_parquet(folds, version: str):
    """Save folds in your loader's naming convention."""
    n_total = len(folds)
    n_cv = n_total - 1  # last is test

    for i, (train_df, val_df) in enumerate(folds, start=1):
        train_name = f"OTPW_{SOURCE}_{version}_FOLD_{i}_TRAIN"
        train_df.write.mode(WRITE_MODE).parquet(f"{OUTPUT_FOLDER}/{train_name}.parquet")

        if i <= n_cv:
            val_name = f"OTPW_{SOURCE}_{version}_FOLD_{i}_VAL"
            val_df.write.mode(WRITE_MODE).parquet(f"{OUTPUT_FOLDER}/{val_name}.parquet")
        else:
            test_name = f"OTPW_{SOURCE}_{version}_FOLD_{i}_TEST"
            val_df.write.mode(WRITE_MODE).parquet(f"{OUTPUT_FOLDER}/{test_name}.parquet")
        
        # Cleanup: Unpersist cached DataFrames after writing to disk to free memory
        # This prevents memory buildup when processing multiple folds
        if train_df.is_cached:
            train_df.unpersist()
        if val_df.is_cached:
            val_df.unpersist()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    for version, path in DATASETS.items():
        print(f"\n================= SPLITTING {version} =================")
        print(f"📥 Reading: {path}")

        df = spark.read.parquet(path)

        # Joins in flight lineage features
        df = flight_lineage_features.add_flight_lineage_features(df)
        
        # CRITICAL: Cache the lineage features to avoid recomputing the entire join chain
        # The lineage join involves many window functions and LAG operations - very expensive to recompute
        # Without caching, every operation on train_df/val_df would recompute the entire lineage join
        df = df.cache()
        cached_count = df.count()  # Trigger cache materialization
        if VERBOSE:
            print(f"✓ Cached flight lineage features ({cached_count:,} rows)")

        folds = create_sliding_window_folds(
            df=df, 
            date_col=DATE_COL,
            n_folds=N_FOLDS,
            test_fold=CREATE_TEST_FOLD,
            verbose=VERBOSE
        )

        print(f"💾 Writing {len(folds)} folds for {version} to {OUTPUT_FOLDER}")
        save_folds_to_parquet(folds, version)
        
        # Unpersist full dataset cache after all folds are written (free memory)
        if df.is_cached:
            df.unpersist()
            if VERBOSE:
                print(f"✓ Unpersisted cached lineage features for {version}")

        print(f"✅ Done writing {version} folds.")
    