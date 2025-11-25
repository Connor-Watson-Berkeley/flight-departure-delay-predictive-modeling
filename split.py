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

    Example with n_folds=3 (creates 4 periods for CV + 1 for test = 5 total):
    - Fold 1: train on period 1, validate on period 2
    - Fold 2: train on period 2, validate on period 3
    - Fold 3: train on period 3, validate on period 4
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
    # Filter out rows with null DEP_DELAY first
    initial_count = df.count()
    df = df.filter(F.col("DEP_DELAY").isNotNull())
    filtered_count = df.count()
    null_count = initial_count - filtered_count
    
    if null_count > 0:
        print(f"⚠ Filtered out {null_count:,} rows with null DEP_DELAY ({null_count/initial_count*100:.2f}%)")
        print(f"  Remaining rows: {filtered_count:,}\n")
    
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
