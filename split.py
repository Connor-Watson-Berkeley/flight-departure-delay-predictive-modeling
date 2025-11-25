from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

def create_sliding_window_folds(
    df: DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "FL_DATE",
    n_folds: int = 4,
    test_fold: bool = True,
    verbose: bool = True,
):
    """
    Create sliding window time-series CV folds without overlap,
    filtering out null DEP_DELAY before splitting.

    Returns:
        folds: list[(train_df, val_df)] with last fold as test if test_fold=True
    """

    # --- 0) Normalize/ensure date column is comparable as date
    # If FL_DATE is a string, cast to date; if already date, no-op
    df = df.withColumn(date_col, F.to_date(F.col(date_col)))

    # --- 1) Filter null labels ONCE up front
    df_clean = df.filter(F.col("DEP_DELAY").isNotNull())

    if verbose:
        initial_count = df.count()
        filtered_count = df_clean.count()
        null_count = initial_count - filtered_count
        if null_count > 0:
            print(
                f"⚠ Filtered out {null_count:,} rows with null DEP_DELAY "
                f"({null_count/initial_count*100:.2f}%). Remaining: {filtered_count:,}\n"
            )

    # --- 2) Cast numeric label to double (Spark ML expects double)
    df_clean = df_clean.withColumn("DEP_DELAY", F.col("DEP_DELAY").cast("double"))

    # --- 3) Create binary labels after cleaning
    df_clean = df_clean.withColumn(
        "SEVERE_DEL60",
        F.when(F.col("DEP_DELAY") >= 60, 1).otherwise(0)
    )

    if "DEP_DEL15" not in df_clean.columns:
        df_clean = df_clean.withColumn(
            "DEP_DEL15",
            F.when(F.col("DEP_DELAY") >= 15, 1).otherwise(0)
        )
    else:
        df_clean = df_clean.withColumn("DEP_DEL15", F.col("DEP_DEL15").cast("int"))

    # --- 4) Auto-detect date range if not provided
    if start_date is None or end_date is None:
        r = df_clean.select(
            F.min(F.col(date_col)).alias("start_date"),
            F.max(F.col(date_col)).alias("end_date")
        ).first()
        start_date = start_date or r["start_date"].strftime("%Y-%m-%d")
        end_date = end_date or r["end_date"].strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_dt - start_dt).days
    n_periods = n_folds + 1 if test_fold else n_folds
    period_days = total_days // n_periods

    if period_days <= 0:
        raise ValueError(
            f"Dataset too small for {n_folds} folds. "
            f"Date range {start_date}..{end_date} ({total_days} days)"
        )

    if verbose:
        print(f"Detected date range: {start_date} to {end_date}")
        print(f"Total days: {total_days}, Period size: {period_days} days")
        print(f"Number of periods: {n_periods}\n")

    # --- 5) Create period boundaries
    # Ensure last period ends exactly at end_dt (avoid dropping tail)
    periods = []
    for i in range(n_periods):
        p_start = start_dt + timedelta(days=i * period_days)
        if i < n_periods - 1:
            p_end = start_dt + timedelta(days=(i + 1) * period_days)
        else:
            p_end = end_dt  # last one takes remainder
        periods.append((p_start, p_end))

    folds = []

    # --- 6) CV folds: train on period i, validate on period i+1
    for fold_num in range(n_folds):
        train_start, train_end = periods[fold_num]
        val_start, val_end = periods[fold_num + 1]

        train_df = df_clean.filter(
            (F.col(date_col) >= F.lit(train_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <  F.lit(train_end.strftime("%Y-%m-%d")))
        )

        val_df = df_clean.filter(
            (F.col(date_col) >= F.lit(val_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <  F.lit(val_end.strftime("%Y-%m-%d")))
        )

        folds.append((train_df, val_df))

        if verbose:
            print(
                f"Fold {fold_num+1}: "
                f"Train [{train_start:%Y-%m-%d} to {train_end:%Y-%m-%d}), "
                f"Val [{val_start:%Y-%m-%d} to {val_end:%Y-%m-%d})"
            )
            print(f"  Train size: {train_df.count()}, Val size: {val_df.count()}\n")

    # --- 7) Test fold: train on all before last period, test on last period
    if test_fold:
        test_start, test_end = periods[-1]
        combined_train_df = df_clean.filter(
            (F.col(date_col) >= F.lit(start_date)) &
            (F.col(date_col) <  F.lit(test_start.strftime("%Y-%m-%d")))
        )

        test_df = df_clean.filter(
            (F.col(date_col) >= F.lit(test_start.strftime("%Y-%m-%d"))) &
            (F.col(date_col) <  F.lit(test_end.strftime("%Y-%m-%d")))
        )

        folds.append((combined_train_df, test_df))

        if verbose:
            print(
                f"Test Fold: Train [{start_date} to {test_start:%Y-%m-%d}), "
                f"Test [{test_start:%Y-%m-%d} to {test_end:%Y-%m-%d})"
            )
            print(
                f"  Train size: {combined_train_df.count()}, "
                f"Test size: {test_df.count()}\n"
            )

    return folds