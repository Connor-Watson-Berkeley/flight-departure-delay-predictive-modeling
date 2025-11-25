#!/usr/bin/env python3
"""
split.py

Create sliding-window CV folds for flight delay data,
filter out null DEP_DELAY rows, create labels, and save
folds in the format:

  OTPW_{SOURCE}_{VERSION}_FOLD_{i}_TRAIN.parquet
  OTPW_{SOURCE}_{VERSION}_FOLD_{i}_VAL.parquet
  OTPW_{SOURCE}_{VERSION}_{VERSION}_FOLD_{i}_TEST.parquet
"""

from __future__ import annotations
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


# ----------------------------------------------------------------------
# CONFIG SECTION — EDIT THESE VALUES AND RUN
# ----------------------------------------------------------------------

INPUT_PATH = "dbfs:/student-groups/Group_4_2/FLIGHTS_3M_FULL.parquet"
OUTPUT_FOLDER = "dbfs:/student-groups/Group_4_2"

VERSION = "3M"      # "3M" or "12M"
SOURCE = "CUSTOM"   # "CUSTOM" or "PROVIDED"

DATE_COL = "FL_DATE"

N_FOLDS = 4         # CV folds (last fold becomes test)
CREATE_TEST_FOLD = True

VERBOSE = True


# ----------------------------------------------------------------------
# FOLD CREATION LOGIC
# ----------------------------------------------------------------------

def create_sliding_window_folds(
    df: DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "FL_DATE",
    n_folds: int = 4,
    test_fold: bool = True,
    verbose: bool = True,
):
    """Create sliding-window CV folds with null-safe labels."""

    # --- Normalize date column
    df = df.withColumn(date_col, F.to_date(F.col(date_col)))

    # --- Remove null label rows
    df_clean = df.filter(F.col("DEP_DELAY").isNotNull())
    if verbose:
        initial = df.count()
        cleaned = df_clean.count()
        print(f"Removed {initial - cleaned:,} null-label rows ({(initial-cleaned)/initial*100:.2f}%).")

    # --- Cast label
    df_clean = df_clean.withColumn("DEP_DELAY", F.col("DEP_DELAY").cast("double"))

    # --- Create binary labels
    df_clean = df_clean.withColumn("SEVERE_DEL60", F.when(F.col("DEP_DELAY") >= 60, 1).otherwise(0))
    df_clean = df_clean.withColumn("DEP_DEL15", F.when(F.col("DEP_DELAY") >= 15, 1).otherwise(0))

    # --- Auto-detect date range
    if start_date is None or end_date is None:
        r = df_clean.select(
            F.min(F.col(date_col)).alias("start"),
            F.max(F.col(date_col)).alias("end")
        ).first()
        start_date = start_date or r["start"].strftime("%Y-%m-%d")
        end_date = end_date or r["end"].strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_dt - start_dt).days
    n_periods = n_folds + 1 if test_fold else n_folds
    period_days = total_days // n_periods

    if verbose:
        print(f"Date range: {start_date} → {end_date}")
        print(f"Period days: {period_days}")
        print(f"Total periods: {n_periods}")

    # Build periods
    periods = []
    for i in range(n_periods):
        p_start = start_dt + timedelta(days=i * period_days)
        p_end = end_dt if i == n_periods - 1 else start_dt + timedelta(days=(i + 1) * period_days)
        periods.append((p_start, p_end))

    folds = []

    # --- Create CV folds
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
            print(f"Fold {f+1}: Train {t_start:%Y-%m-%d} → {t_end:%Y-%m-%d}, "
                  f"Val {v_start:%Y-%m-%d} → {v_end:%Y-%m-%d}")
            print(f"    Train = {train_df.count():,}, Val = {val_df.count():,}")

    # --- Add test fold
    if test_fold:
        test_start, test_end = periods[-1]

        combined_train = df_clean.filter(
            (F.col(date_col) >= F.lit(start_date)) &
            (F.col(date_col) < F.lit(test_start.strftime("%Y-%m-%d")))
        )
        test_df = df_clean.filter(
            (F.col(date_col) >= F.lit(test_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <= F.lit(test_end.strftime("%Y-%m-%d")))
        )

        folds.append((combined_train, test_df))

        if verbose:
            print(f"Test Fold: Train ALL up to {test_start:%Y-%m-%d}, "
                  f"Test {test_start:%Y-%m-%d} → {test_end:%Y-%m-%d}")
            print(f"    Train = {combined_train.count():,}, Test = {test_df.count():,}")

    return folds


# ----------------------------------------------------------------------
# SAVING FOLDS
# ----------------------------------------------------------------------

def save_folds_to_parquet(
    folds,
    folder_path: str,
    version: str,
    source: str,
    mode: str = "overwrite"
):
    """Save the folds in loader-compatible naming format."""

    n_total = len(folds)
    n_cv = n_total - 1  # all except last

    for i, (train_df, val_df) in enumerate(folds, start=1):

        # Train split name
        train_name = f"OTPW_{source}_{version}_FOLD_{i}_TRAIN"
        train_df.write.mode(mode).parquet(f"{folder_path}/{train_name}.parquet")

        # CV or test
        if i <= n_cv:
            val_name = f"OTPW_{source}_{version}_FOLD_{i}_VAL"
            val_df.write.mode(mode).parquet(f"{folder_path}/{val_name}.parquet")
        else:
            test_name = f"OTPW_{source}_{version}_FOLD_{i}_TEST"
            val_df.write.mode(mode).parquet(f"{folder_path}/{test_name}.parquet")


# ----------------------------------------------------------------------
# MAIN EXECUTION (NO ARGPARSE)
# ----------------------------------------------------------------------

if __name__ == "__main__":

    spark = SparkSession.builder.getOrCreate()

    print(f"\n📥 Loading input data from: {INPUT_PATH}")
    df = spark.read.parquet(INPUT_PATH)

    folds = create_sliding_window_folds(
        df=df,
        date_col=DATE_COL,
        n_folds=N_FOLDS,
        test_fold=CREATE_TEST_FOLD,
        verbose=VERBOSE,
    )

    print(f"\n💾 Saving {len(folds)} folds to: {OUTPUT_FOLDER}")
    save_folds_to_parquet(
        folds,
        folder_path=OUTPUT_FOLDER,
        version=VERSION,
        source=SOURCE,
    )

    print("\n✅ Done! Folds successfully written.\n")