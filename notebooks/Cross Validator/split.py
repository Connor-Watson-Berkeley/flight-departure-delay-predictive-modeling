#!/usr/bin/env python3
"""
split.py (CUSTOM only, real paths, processes 3M, 12M, and 60M)

Reads:
  3M  from dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m
  12M from dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015
  60M from dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60m

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

# Load module from our Databricks repo
import importlib.util

flight_lineage_features_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/flight_lineage_features.py"
spec = importlib.util.spec_from_file_location("flight_lineage_features", flight_lineage_features_path)
flight_lineage_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flight_lineage_features)


# -------------------------
# HARD-CODED SETTINGS
# -------------------------
OUTPUT_FOLDER = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "OTPW"  # Change to "OTPW" when processing OTPW data
DATE_COL = "FL_DATE"  # Default/preference - auto-detects actual column (FL_DATE for CUSTOM, fl_date for OTPW)

# List of versions to assess (e.g., ["3M", "12M", "60M"])
VERSIONS = ["60M"]  # <-- EDIT THIS LIST

# Dataset paths - depends on SOURCE
# When SOURCE="CUSTOM": Uses Custom Join outputs
# When SOURCE="OTPW": Uses OTPW processed outputs (with _otpw suffix)
if SOURCE == "OTPW":
    DATASETS = {
        "3M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m_otpw",
        "12M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015_otpw",
        "60M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M_otpw"
    }
else:  # SOURCE == "CUSTOM"
    DATASETS = {
        "3M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m",
        "12M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015",
        # 60M: Created by Custom Join (full).ipynb with data_version="60m" for 2015-2019 data
        # The Custom Join notebook filters to 2015-2019 at the beginning (Cell 13) for efficiency
        # "60M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60m"
        "60M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M"
    }

N_FOLDS = 3              # CV folds
CREATE_TEST_FOLD = True  # adds one final test period
WRITE_MODE = "overwrite"
VERBOSE = True


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
    
    # Note: Date column should already be normalized before calling this function
    # This normalization is kept for backwards compatibility
    if df.schema[date_col].dataType.typeName() != "date":
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


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    # Process only versions specified in VERSIONS list
    for version in VERSIONS:
        if version not in DATASETS:
            print(f"⚠️  Warning: Version '{version}' not found in DATASETS. Skipping...")
            print(f"   Available versions: {list(DATASETS.keys())}")
            continue
        
        path = DATASETS[version]
        print(f"\n================= SPLITTING {version} =================")
        print(f"📥 Reading: {path}")

        df = spark.read.parquet(path)
        
        # Normalize date column for all versions (consistent processing)
        # Handle both FL_DATE (CUSTOM) and fl_date (OTPW after column mapping)
        date_col_actual = None
        if DATE_COL in df.columns:
            date_col_actual = DATE_COL
        elif DATE_COL.lower() in df.columns:
            date_col_actual = DATE_COL.lower()
        elif "fl_date" in df.columns:
            date_col_actual = "fl_date"
        elif "FL_DATE" in df.columns:
            date_col_actual = "FL_DATE"
        else:
            raise ValueError(f"Date column not found. Expected '{DATE_COL}' or 'fl_date'. Available columns: {sorted(df.columns)[:20]}...")
        
        # Use the actual date column name found in the DataFrame
        df = df.withColumn(date_col_actual, F.to_date(F.col(date_col_actual)))
        
        # Update DATE_COL to the actual column name for use in fold creation
        date_col_for_folds = date_col_actual
        
        # Check actual date range in dataset (for validation)
        date_range = df.select(
            F.min(F.col(date_col_for_folds)).alias("min_date"),
            F.max(F.col(date_col_for_folds)).alias("max_date")
        ).first()
        print(f"📅 Dataset date range: {date_range['min_date']} to {date_range['max_date']}")
        
        # Filter datasets to 2015-2019 (2019 is test set year)
        # All versions should be filtered to this range for consistency
        # Note: 60M is already filtered in Custom Join notebook, but we filter anyway to ensure consistency
        initial_count = df.count()
        df = df.filter(
            (F.col(date_col_for_folds) >= F.lit("2015-01-01")) & 
            (F.col(date_col_for_folds) < F.lit("2020-01-01"))
        )
        filtered_count = df.count()
        removed = initial_count - filtered_count
        if removed > 0:
            print(f"🔍 Filtered to 2015-2019: Removed {removed:,} rows outside range ({removed/initial_count*100:.2f}%)")
        print(f"  Remaining: {filtered_count:,} rows")

        # Add flight lineage features
        df = flight_lineage_features.add_flight_lineage_features(df)

        folds = create_sliding_window_folds(
            df=df,
            date_col=date_col_for_folds,  # Use the actual date column name found
            n_folds=N_FOLDS,
            test_fold=CREATE_TEST_FOLD,
            verbose=VERBOSE
        )

        print(f"💾 Writing {len(folds)} folds for {version} to {OUTPUT_FOLDER}")
        save_folds_to_parquet(folds, version)

        print(f"✅ Done writing {version} folds.")
