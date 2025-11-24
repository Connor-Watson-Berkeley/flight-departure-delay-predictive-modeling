"""
Time-series cross-validation splitting utilities for flight delay prediction.

This module provides functions to create sliding window cross-validation folds
for time-series data, where each fold trains on one time period and validates
on the next sequential period.
"""

from datetime import datetime, timedelta
from pyspark.sql import functions as F


def create_sliding_window_folds(
    df,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "FL_DATE",
    n_folds: int = 4,
    test_fold: bool = True
):
    """
    Create sliding window time-series cross-validation folds without overlap.

    Divides data into equal time periods. Each fold trains on one period and
    validates on the next sequential period (non-overlapping). The last period 
    is reserved for the final test fold.

    Example with n_folds=4 (creates 5 periods total):
    - Fold 1: train on period 1, validate on period 2
    - Fold 2: train on period 2, validate on period 3
    - Fold 3: train on period 3, validate on period 4
    - Fold 4: train on period 4, validate on period 5
    - Test: train on periods 1-4 combined, test on period 5

    Args:
        df: PySpark DataFrame
        start_date: Starting date as string 'YYYY-MM-DD' or None for auto-detection
        end_date: Ending date as string 'YYYY-MM-DD' or None for auto-detection
        date_col: Name of the date column
        n_folds: Number of CV folds to create (not including test fold)
        test_fold: If True, reserve last period for test and create combined train set

    Returns:
        List of tuples (train_df, val_df) for each fold.
        If test_fold=True, last tuple is (combined_train_df, test_df)
    """
    # Cast DEP_DELAY as float and create binary labels
    df = df.withColumn("DEP_DELAY", F.col("DEP_DELAY").cast("float"))

    # Create SEVERE_DEL60 binary variable (1 if delay >= 60 min, 0 otherwise)
    df = df.withColumn(
        "SEVERE_DEL60",
        F.when(F.col("DEP_DELAY") >= 60, 1).otherwise(0)
    )

    # Ensure DEP_DEL15 exists (1 if delay >= 15 min, 0 otherwise)
    if "DEP_DEL15" not in df.columns:
        df = df.withColumn(
            "DEP_DEL15",
            F.when(F.col("DEP_DELAY") >= 15, 1).otherwise(0)
        )

    # Auto-detect dates if not provided
    if start_date is None or end_date is None:
        date_range = df.select(
            F.min(F.col(date_col)).alias("start_date"),
            F.max(F.col(date_col)).alias("end_date")
        ).first()

        start_date = start_date or str(date_range["start_date"])
        end_date = end_date or str(date_range["end_date"])

    # Convert to datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Calculate total days and period size
    total_days = (end_dt - start_dt).days
    # For n_folds CV folds + 1 test fold = n_folds + 1 periods total
    n_periods = n_folds + 1 if test_fold else n_folds
    period_days = total_days // n_periods

    if period_days == 0:
        raise ValueError(f"Dataset too small for {n_folds} folds. Total days: {total_days}")

    # Print header
    print(f"Detected date range: {start_date} to {end_date}")
    print(f"Total days: {total_days}, Period size: {period_days} days (~{period_days/7:.1f} weeks)")
    print(f"Number of periods: {n_periods}")
    print()

    # Create visual timeline
    _print_sliding_timeline(start_dt, end_dt, period_days, n_folds, test_fold)
    print()

    # Create period boundaries
    periods = []
    for i in range(n_periods):
        period_start = start_dt + timedelta(days=i * period_days)
        period_end = start_dt + timedelta(days=(i + 1) * period_days)
        if period_end > end_dt:
            period_end = end_dt
        periods.append((period_start, period_end))

    # Create folds
    folds = []

    # Create CV folds (sliding window)
    for fold_num in range(n_folds):
        train_start, train_end = periods[fold_num]
        val_start, val_end = periods[fold_num + 1]

        # Training: single period
        train_df = df.filter(
            (F.col(date_col) >= F.lit(train_start.strftime('%Y-%m-%d'))) &
            (F.col(date_col) < F.lit(train_end.strftime('%Y-%m-%d')))
        )

        # Validation: next period
        val_df = df.filter(
            (F.col(date_col) >= F.lit(val_start.strftime('%Y-%m-%d'))) &
            (F.col(date_col) < F.lit(val_end.strftime('%Y-%m-%d')))
        )

        folds.append((train_df, val_df))

        train_days = (train_end - train_start).days
        val_days = (val_end - val_start).days

        print(f"Fold {fold_num + 1}: Train [{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}), "
              f"Val [{val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')})")
        print(f"  Train: {train_days} days, Val: {val_days} days")
        print(f"  Train size: {train_df.count()}, Val size: {val_df.count()}")
        print()

    # Create test fold if requested
    if test_fold:
        # Train on all periods except the last
        test_start, test_end = periods[-1]

        combined_train_df = df.filter(
            (F.col(date_col) >= F.lit(start_date)) &
            (F.col(date_col) < F.lit(test_start.strftime('%Y-%m-%d')))
        )

        test_df = df.filter(
            (F.col(date_col) >= F.lit(test_start.strftime('%Y-%m-%d'))) &
            (F.col(date_col) < F.lit(test_end.strftime('%Y-%m-%d')))
        )

        folds.append((combined_train_df, test_df))

        combined_train_days = (test_start - start_dt).days
        test_days = (test_end - test_start).days

        print(f"Test Fold: Train [{start_date} to {test_start.strftime('%Y-%m-%d')}), "
              f"Test [{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')})")
        print(f"  Train: {combined_train_days} days (all previous periods), Test: {test_days} days")
        print(f"  Train size: {combined_train_df.count()}, Test size: {test_df.count()}")
        print()

    return folds


def _print_sliding_timeline(start_dt, end_dt, period_days, n_folds, test_fold):
    """Print visual timeline of the sliding window cross-validation strategy"""

    n_periods = n_folds + 1 if test_fold else n_folds + 1

    # Create period labels
    periods = []
    current = start_dt
    for i in range(n_periods):
        period_end = current + timedelta(days=period_days)
        if period_end > end_dt:
            period_end = end_dt
        periods.append({
            'start': current,
            'end': period_end,
            'label': current.strftime('%Y-%m-%d')
        })
        current = period_end

    # Fixed width for each period segment
    segment_width = 17

    # Print timeline header
    print("Dataset Timeline: ", end="")
    for i, period in enumerate(periods):
        if i == 0:
            print(f"{period['label']}", end="")
        else:
            # Calculate separator to maintain fixed width
            separator_len = segment_width - len(period['label'])
            print(f" {'─' * separator_len} {period['label']}", end="")
    print()

    # Print timeline bars
    print(" " * 18, end="")
    for i in range(len(periods)):
        print(f"|{'═' * (segment_width - 1)}", end="")
    print("|")
    print()

    # Print each CV fold
    for fold_num in range(n_folds):
        train_period = fold_num
        val_period = fold_num + 1

        # Calculate positions with fixed width
        fold_label = f"Fold {fold_num + 1}:"
        fold_indent = 18  # Fixed indent for all fold labels

        # Build train label (single period)
        train_label = f"[Train: {periods[train_period]['label']}]"

        # Build val label
        val_label = f"[Val: {periods[val_period]['label']}]"

        # Calculate spacing to align val label with its period
        val_position = fold_indent + (segment_width * val_period)
        current_pos = len(fold_label) + len(train_label) + fold_indent
        spacing = val_position - current_pos

        # Print the fold description line
        print(f"{fold_label:<18}{train_label}", end="")
        print(" " * max(0, spacing), end="")
        print(val_label)

        # Print visual representation
        print(" " * fold_indent, end="")

        # Skip to train period
        if train_period > 0:
            print(" " * (segment_width * train_period), end="")

        # Train bar
        print("|", end="")
        train_width = segment_width - 2
        print("█" * train_width, end="")
        print("|", end="")

        # Val bar
        print("|", end="")
        val_width = segment_width - 2
        print("─" * val_width, end="")
        print("|")
        print()

    # Print test fold if applicable
    if test_fold:
        fold_label = "Test Fold:"
        fold_indent = 18

        # Train on all previous periods
        train_label = f"[Train: {periods[0]['label']}─────{periods[-2]['label']}]"
        test_label = f"[Test: {periods[-1]['label']}]"

        # Calculate spacing
        test_position = fold_indent + (segment_width * (n_periods - 1))
        current_pos = len(fold_label) + len(train_label) + fold_indent
        spacing = test_position - current_pos

        print(f"{fold_label:<18}{train_label}", end="")
        print(" " * max(0, spacing), end="")
        print(test_label)

        # Print visual representation
        print(" " * fold_indent, end="")

        # Train on all periods except last
        for i in range(n_periods - 1):
            print("|", end="")
            print("█" * (segment_width - 2), end="")
            print("|", end="")

        # Test period
        print("|", end="")
        print("─" * (segment_width - 2), end="")
        print("|")
        print()

    print("Legend:  [███] = Training Data    [───] = Validation/Test Data")


def save_folds(folds, version="3M", group_folder_path="dbfs:/student-groups/Group_4_2"):
    """
    Save cross-validation folds to parquet files.

    Args:
        folds: List of (train_df, val_df) tuples from create_sliding_window_folds
        version: Data version identifier (e.g., "3M", "12M", "CUSTOM")
        group_folder_path: Base path for saving parquet files
    """
    # Save the folds
    for fold_num, (train_df, val_df) in enumerate(folds, 1):
        # Determine if this is the last fold (test set)
        is_last_fold = (fold_num == len(folds))
        split_type = "TEST" if is_last_fold else "VAL"

        # Save train
        train_name = f"OTPW_{version}_FOLD_{fold_num}_TRAIN"
        print(f"Saving Fold {fold_num} train set: {train_name}")
        train_df.write.mode("overwrite").parquet(f"{group_folder_path}/{train_name}.parquet")
        train_count = train_df.count()
        print(f"  ✓ Saved {train_count:,} rows")

        # Save validation or test
        val_name = f"OTPW_{version}_FOLD_{fold_num}_{split_type}"
        print(f"Saving Fold {fold_num} {split_type.lower()} set: {val_name}")
        val_df.write.mode("overwrite").parquet(f"{group_folder_path}/{val_name}.parquet")
        val_count = val_df.count()
        print(f"  ✓ Saved {val_count:,} rows")
        print()


def main():
    """
    Main execution function to create and save temporal cross-validation folds
    for flight delay prediction datasets.

    Processes 3M and 12M datasets, creates sliding window cross-validation splits,
    and saves the folds to parquet files.
    """
    from pyspark.sql import SparkSession

    # Initialize Spark
    spark = SparkSession.builder.appName("FlightDelayCV").getOrCreate()

    # Dataset paths
    dataset_dict = {
        "3M": "dbfs:/mnt/mids-w261/OTPW_3M/OTPW_3M/OTPW_3M_2015.csv.gz",
        "12M": "dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M/OTPW_12M_2015.csv.gz"
    }

    # Output configuration
    group_folder_path = "dbfs:/student-groups/Group_4_2"
    n_folds = 4

    # Process each dataset version
    for version, path in dataset_dict.items():
        print(f"\n{'=' * 80}")
        print(f"Processing {version} Dataset")
        print(f"{'=' * 80}\n")

        try:
            # Load dataset
            print(f"Loading {version} dataset from {path}...")
            df = spark.read.csv(path, header=True, inferSchema=True)
            print(f"✓ Loaded {df.count():,} rows\n")

            # Create sliding window folds
            print(f"Creating {n_folds} cross-validation folds with temporal split...\n")
            folds = create_sliding_window_folds(
                df,
                n_folds=n_folds,
                test_fold=True,
                date_col="FL_DATE"
            )

            # Save folds
            print(f"Saving {version} folds to {group_folder_path}...\n")
            save_folds(folds, version=version, group_folder_path=group_folder_path)

            print(f"✓ Successfully processed {version} dataset\n")

        except Exception as e:
            print(f"✗ Error processing {version} dataset: {str(e)}\n")
            raise

    print(f"{'=' * 80}")
    print("All datasets processed successfully!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
