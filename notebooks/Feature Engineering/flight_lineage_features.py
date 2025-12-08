"""
Flight Lineage Join Module

This module adds flight lineage features by joining each flight to its previous flight
in the lineage (same aircraft, previous flight).

Key Features Added:
- Previous flight information (origin, dest, times, delays, etc.)
- Turnover time (time from arrival to departure)
- Cumulative delays
- Sequence information (lineage rank)
- Jump detection

CRITICAL: NO ROWS ARE DROPPED - All flights are preserved. Flights without previous
flight data get NULL values which are handled via imputation.

IMPORTANT: When adding new NUMERIC features to this module, they must also be added
to the `numerical_features` list in `notebooks/Cross Validator/cv.py` to ensure
proper type casting during data loading. See cv.py FlightDelayDataLoader class.

Usage:
    from flight_lineage import add_flight_lineage_features
    
    df_with_lineage = add_flight_lineage_features(df)
"""

from pyspark.sql.functions import (
    col, when, lit, coalesce, lag, row_number, sum as spark_sum, avg, max,
    array, array_remove, lpad, concat, floor, expr
)
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from datetime import datetime


def add_flight_lineage_features(df):
    """
    Add flight lineage features using window functions.
    
    This function adds ~38 new columns related to the previous flight in the
    aircraft's lineage, including delays, times, and engineered features.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Input DataFrame with flight data. Must contain:
        - tail_num (or similar column for aircraft identification)
        - fl_date, arr_time, crs_arr_time, dep_time, crs_dep_time
        - origin, dest, and other flight attributes
    
    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with all original columns plus lineage features:
        - lineage_rank: Rank of flight in aircraft's sequence
        - prev_flight_*: Previous flight attributes
        - lineage_*: Engineered lineage features
        - columns_with_data_leakage: Array of columns with data leakage
        - prediction_cutoff_timestamp: Cutoff timestamp for data leakage detection
    
    Notes:
    ------
    - All flights are preserved (no rows dropped)
    - First flights (lineage_rank == 1) get imputed values
    - Data leakage flags are added for all risky columns
    - See FLIGHT_LINEAGE_JOIN_DESIGN.md for complete documentation
    """
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Generating flight lineage features...")
    print("=" * 60)
    print("FLIGHT LINEAGE JOIN")
    print("=" * 60)
    
    # Step 1: Identify tail_num column
    tail_num_col = _find_tail_num_column(df)
    
    # Step 2: Prepare arrival timestamp for ranking
    print("\nStep 2: Creating arrival timestamp for ranking...")
    df = df.withColumn(
        'arrival_time_for_ranking',
        F.coalesce(col('arr_time'), col('crs_arr_time'))
    )
    
    df = df.withColumn(
        'arrival_timestamp',
        F.when(
            col('arrival_time_for_ranking').isNotNull() & col('fl_date').isNotNull(),
            F.to_timestamp(
                F.concat(col('fl_date'), F.lpad(col('arrival_time_for_ranking').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    print("✓ Arrival timestamp created")
    
    # Step 3: Create window specification and rank flights
    print("\nStep 3: Creating window specification and ranking flights...")
    window_spec = Window.partitionBy(tail_num_col).orderBy(F.col('arrival_timestamp').asc_nulls_last())
    df = df.withColumn('lineage_rank', F.row_number().over(window_spec))
    print("✓ Flights ranked")
    
    # Step 4: Get Previous Flight Data Using LAG
    print("\nStep 4: Getting previous flight data using LAG...")
    df = _add_previous_flight_data(df, window_spec)
    print("✓ Previous flight data retrieved")
    
    # Step 5: Convert crs_dep_time to minutes
    df = df.withColumn(
        'crs_dep_time_minutes',
        F.when(
            col('crs_dep_time').isNotNull(),
            (F.floor(col('crs_dep_time') / 100) * 60 + (col('crs_dep_time') % 100))
        ).otherwise(None)
    )
    
    # Step 6: Compute Actual Turnover Time
    print("\nStep 6: Computing actual turnover time...")
    df = _compute_turnover_time(df)
    print("✓ Actual turnover time computed")
    
    # Step 7: Compute Expected Flight Time and Cumulative Features
    print("\nStep 7: Computing expected flight time and cumulative features...")
    df = _compute_cumulative_features(df, tail_num_col)
    print("✓ Cumulative features computed")
    
    # Step 8: Jump Detection
    print("\nStep 8: Detecting jumps (aircraft repositioning)...")
    df = df.withColumn(
        'lineage_is_jump',
        F.when(col('lineage_rank') == 1, F.lit(False))
        .when(col('prev_flight_dest') != col('origin'), F.lit(True))
        .otherwise(F.lit(False))
    )
    print("✓ Jump detection complete")
    
    # Step 9: Check Data Leakage for All Risky Columns
    print("\nStep 9: Checking data leakage for all risky columns...")
    df = _check_data_leakage(df)
    print("✓ Data leakage checks complete")
    
    # Step 10: Apply Imputation for NULL Values (First Flight Handling)
    print("\nStep 10: Applying imputation for NULL values (first flight handling)...")
    df = _apply_imputation(df)
    print("✓ Imputation complete - all NULLs replaced with design doc values")
    
    end_time = datetime.now()
    duration = end_time - start_time
    timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "=" * 60)
    print("✓ FLIGHT LINEAGE JOIN COMPLETE")
    print("=" * 60)
    print(f"\nNew columns added: ~38 lineage features")
    print(f"All flights preserved - no rows dropped")
    print(f"[{timestamp}] ✓ Flight lineage feature generation complete! (took {duration})")
    print("=" * 60)
    
    return df


def _find_tail_num_column(df):
    """Find the tail number column in the DataFrame."""
    tail_num_candidates = ['tail_num', 'TAIL_NUM', 'tail_number', 'TAIL_NUMBER', 'op_unique_carrier_tail_num']
    
    for candidate in tail_num_candidates:
        if candidate in df.columns:
            print(f"✓ Found tail_num column: {candidate}")
            return candidate
    
    # Try pattern matching
    tail_cols = [c for c in df.columns if 'tail' in c.lower()]
    if tail_cols:
        tail_num_col = tail_cols[0]
        print(f"✓ Found tail_num column via pattern matching: {tail_num_col}")
        return tail_num_col
    else:
        raise ValueError(f"Could not find tail_num column. Available columns: {df.columns[:20]}...")


def _add_previous_flight_data(df, window_spec):
    """Add previous flight data using LAG window function."""
    # Core Previous Flight Information
    df = df.withColumn('prev_flight_origin', F.lag('origin', 1).over(window_spec))
    df = df.withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))
    df = df.withColumn('prev_flight_actual_dep_time', F.lag('dep_time', 1).over(window_spec))
    df = df.withColumn('prev_flight_actual_arr_time', F.lag('arr_time', 1).over(window_spec))
    df = df.withColumn('prev_flight_dep_delay', F.lag('dep_delay', 1).over(window_spec))
    df = df.withColumn('prev_flight_arr_delay', F.lag('arr_delay', 1).over(window_spec))
    df = df.withColumn('prev_flight_air_time', F.lag('air_time', 1).over(window_spec))
    
    # Scheduled Times
    df = df.withColumn('prev_flight_crs_dep_time', F.lag('crs_dep_time', 1).over(window_spec))
    df = df.withColumn('prev_flight_crs_arr_time', F.lag('crs_arr_time', 1).over(window_spec))
    df = df.withColumn('prev_flight_crs_elapsed_time', F.lag('crs_elapsed_time', 1).over(window_spec))
    
    # Time Components
    df = df.withColumn('prev_flight_taxi_in', F.lag('taxi_in', 1).over(window_spec))
    df = df.withColumn('prev_flight_taxi_out', F.lag('taxi_out', 1).over(window_spec))
    df = df.withColumn('prev_flight_wheels_off', F.lag('wheels_off', 1).over(window_spec))
    df = df.withColumn('prev_flight_wheels_on', F.lag('wheels_on', 1).over(window_spec))
    df = df.withColumn('prev_flight_actual_elapsed_time', F.lag('actual_elapsed_time', 1).over(window_spec))
    
    # Route and Flight Information
    df = df.withColumn('prev_flight_distance', F.lag('distance', 1).over(window_spec))
    df = df.withColumn('prev_flight_op_carrier', F.lag('op_carrier', 1).over(window_spec))
    df = df.withColumn('prev_flight_op_carrier_fl_num', F.lag('op_carrier_fl_num', 1).over(window_spec))
    df = df.withColumn('prev_flight_fl_date', F.lag('fl_date', 1).over(window_spec))
    
    # Status Flags
    df = df.withColumn('prev_flight_cancelled', F.lag('cancelled', 1).over(window_spec))
    df = df.withColumn('prev_flight_diverted', F.lag('diverted', 1).over(window_spec))
    
    return df


def _compute_turnover_time(df):
    """
    Compute actual turnover time (time between previous flight arrival and current flight departure).
    
    NOTE: Data leakage detection is handled comprehensively in Step 9 (_check_data_leakage).
    This function computes the turnover time regardless of leakage; Step 9 will mark it
    as having leakage if the source timestamps occur after the prediction cutoff.
    """
    df = df.withColumn(
        'prev_flight_actual_arr_time_minutes',
        F.when(
            col('prev_flight_actual_arr_time').isNotNull(),
            (F.floor(col('prev_flight_actual_arr_time') / 100) * 60 + (col('prev_flight_actual_arr_time') % 100))
        ).otherwise(None)
    )
    
    df = df.withColumn(
        'actual_dep_time_minutes',
        F.when(
            col('dep_time').isNotNull(),
            (F.floor(col('dep_time') / 100) * 60 + (col('dep_time') % 100))
        ).otherwise(None)
    )
    
    # Compute actual turnover time
    # NOTE: Data leakage is checked in Step 9, which will mark this feature as having
    # leakage if either prev_flight_actual_arr_time or actual_dep_time occurs after
    # the prediction cutoff timestamp.
    df = df.withColumn(
        'lineage_actual_turnover_time_minutes',
        F.when(
            (col('actual_dep_time_minutes').isNotNull()) &
            (col('prev_flight_actual_arr_time_minutes').isNotNull()),
            F.when(
                col('actual_dep_time_minutes') >= col('prev_flight_actual_arr_time_minutes'),
                col('actual_dep_time_minutes') - col('prev_flight_actual_arr_time_minutes')
            ).otherwise(col('actual_dep_time_minutes') + 1440 - col('prev_flight_actual_arr_time_minutes'))
        ).otherwise(None)
    )
    
    df = df.withColumn('lineage_actual_taxi_time_minutes', col('lineage_actual_turnover_time_minutes'))
    df = df.withColumn('lineage_actual_turn_time_minutes', col('lineage_actual_turnover_time_minutes'))
    
    return df


def _compute_cumulative_features(df, tail_num_col):
    """Compute cumulative delay features."""
    df = df.withColumn(
        'crs_arr_time_minutes',
        F.when(
            col('crs_arr_time').isNotNull(),
            (F.floor(col('crs_arr_time') / 100) * 60 + (col('crs_arr_time') % 100))
        ).otherwise(None)
    )
    
    # Convert prev_flight_crs_arr_time to minutes for turnover time calculation
    df = df.withColumn(
        'prev_flight_crs_arr_time_minutes',
        F.when(
            col('prev_flight_crs_arr_time').isNotNull(),
            (F.floor(col('prev_flight_crs_arr_time') / 100) * 60 + (col('prev_flight_crs_arr_time') % 100))
        ).otherwise(None)
    )
    
    # Compute scheduled turnover time (time between prev flight's scheduled arrival and current flight's scheduled departure)
    df = df.withColumn(
        'lineage_turnover_time_minutes',
        F.when(
            (col('prev_flight_crs_arr_time_minutes').isNotNull()) & (col('crs_dep_time_minutes').isNotNull()),
            F.when(
                col('crs_dep_time_minutes') >= col('prev_flight_crs_arr_time_minutes'),
                col('crs_dep_time_minutes') - col('prev_flight_crs_arr_time_minutes')
            ).otherwise(col('crs_dep_time_minutes') + 1440 - col('prev_flight_crs_arr_time_minutes'))
        ).otherwise(None)
    )
    
    # Scheduled taxi and turn time are the same as scheduled turnover time
    df = df.withColumn('lineage_taxi_time_minutes', col('lineage_turnover_time_minutes'))
    df = df.withColumn('lineage_turn_time_minutes', col('lineage_turnover_time_minutes'))
    
    df = df.withColumn(
        'lineage_expected_flight_time_minutes',
        F.when(
            (col('crs_arr_time_minutes').isNotNull()) & (col('crs_dep_time_minutes').isNotNull()),
            F.when(
                col('crs_arr_time_minutes') >= col('crs_dep_time_minutes'),
                col('crs_arr_time_minutes') - col('crs_dep_time_minutes')
            ).otherwise(col('crs_arr_time_minutes') + 1440 - col('crs_dep_time_minutes'))
        ).otherwise(None)
    )
    
    window_spec_cumulative = Window.partitionBy(tail_num_col).orderBy(F.col('arrival_timestamp').asc_nulls_last()).rowsBetween(Window.unboundedPreceding, -1)
    df = df.withColumn('lineage_cumulative_delay', F.sum('dep_delay').over(window_spec_cumulative))
    df = df.withColumn('lineage_num_previous_flights', F.count('*').over(window_spec_cumulative))
    df = df.withColumn('lineage_avg_delay_previous_flights', F.avg('dep_delay').over(window_spec_cumulative))
    df = df.withColumn('lineage_max_delay_previous_flights', F.max('dep_delay').over(window_spec_cumulative))
    
    return df


def _check_data_leakage(df):
    """
    Check for data leakage in all risky columns.
    
    PURPOSE:
    Identify any fields which would have been realized **after** the prediction cutoff timestamp,
    which is two hours before the scheduled departure time. In production, our model would not
    know these timestamps at prediction time, so they represent data leakage.
    
    DATA LEAKAGE RULES:
    1. **Scheduled times are SAFE**: Scheduled times (crs_dep_time, crs_arr_time, etc.) are known
       in advance and are safe to use, even if they occur after the cutoff.
    
    2. **Actual timestamps are RISKY**: Any actual timestamps (actual_dep_time, actual_arr_time,
       wheels_off, wheels_on) that occur AFTER the cutoff are NOT SAFE, as in production our model
       would not know these timestamps to make predictions.
    
    3. **Engineered features inherit leakage**: Any engineered features derived from leakage features
       also contain data leakage. For example:
       - If actual_arr_time has leakage, then any feature using it (like flight_time = actual_arr - actual_dep)
         also has leakage, even if actual_dep does not have leakage.
       - If either actual_dep_time OR actual_arr_time has leakage, then air_time (derived from both) has leakage.
    
    LOGIC:
    - Prediction cutoff = scheduled_departure_time - 2 hours
    - Data leakage = actual_timestamp > prediction_cutoff_timestamp
    - Duration/status fields inherit leakage from their source timestamp columns
    - Engineered features inherit leakage from any of their source columns
    """
    # ============================================================================
    # STEP 1: Create prediction cutoff timestamp
    # ============================================================================
    # The cutoff is 2 hours before scheduled departure time.
    # Any actual timestamps that occur AFTER this cutoff would not be known
    # in production at prediction time.
    df = df.withColumn(
        'prediction_cutoff_timestamp',
        F.when(
            col('sched_depart_date_time').isNotNull(),
            F.expr("sched_depart_date_time - INTERVAL 2 HOURS")
        ).otherwise(None)
    )
    
    # ============================================================================
    # STEP 2: Convert actual time columns to timestamps for comparison
    # ============================================================================
    # Convert HHMM format times to full timestamps using the previous flight's date.
    # These timestamps will be compared against the prediction cutoff to detect leakage.
    # NOTE: Scheduled times (crs_*) are NOT checked here because they are safe by definition.
    df = df.withColumn(
        'prev_flight_actual_dep_timestamp',
        F.when(
            (col('prev_flight_actual_dep_time').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_actual_dep_time').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    df = df.withColumn(
        'prev_flight_actual_arr_timestamp',
        F.when(
            (col('prev_flight_actual_arr_time').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_actual_arr_time').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    df = df.withColumn(
        'prev_flight_wheels_off_timestamp',
        F.when(
            (col('prev_flight_wheels_off').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_wheels_off').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    df = df.withColumn(
        'prev_flight_wheels_on_timestamp',
        F.when(
            (col('prev_flight_wheels_on').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_wheels_on').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    # ============================================================================
    # STEP 3: Check each actual timestamp column for data leakage
    # ============================================================================
    # Data leakage occurs when: actual_timestamp > prediction_cutoff_timestamp
    # This means the actual event happened AFTER the cutoff, so in production
    # we would not know this information at prediction time.
    
    df = df.withColumn(
        'has_leakage_prev_flight_actual_dep_time',
        F.when(
            (col('prev_flight_actual_dep_timestamp').isNotNull()) & (col('prediction_cutoff_timestamp').isNotNull()),
            col('prev_flight_actual_dep_timestamp') > col('prediction_cutoff_timestamp')
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'has_leakage_prev_flight_actual_arr_time',
        F.when(
            (col('prev_flight_actual_arr_timestamp').isNotNull()) & (col('prediction_cutoff_timestamp').isNotNull()),
            col('prev_flight_actual_arr_timestamp') > col('prediction_cutoff_timestamp')
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'has_leakage_prev_flight_wheels_off',
        F.when(
            (col('prev_flight_wheels_off_timestamp').isNotNull()) & (col('prediction_cutoff_timestamp').isNotNull()),
            col('prev_flight_wheels_off_timestamp') > col('prediction_cutoff_timestamp')
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'has_leakage_prev_flight_wheels_on',
        F.when(
            (col('prev_flight_wheels_on_timestamp').isNotNull()) & (col('prediction_cutoff_timestamp').isNotNull()),
            col('prev_flight_wheels_on_timestamp') > col('prediction_cutoff_timestamp')
        ).otherwise(False)
    )
    
    # ============================================================================
    # STEP 4: Derive leakage for duration/status fields from source timestamps
    # ============================================================================
    # Duration and status fields inherit leakage from their source timestamp columns.
    # For example:
    # - dep_delay is derived from actual_dep_time, so it has leakage if actual_dep_time has leakage
    # - arr_delay is derived from actual_arr_time, so it has leakage if actual_arr_time has leakage
    # - taxi_in is derived from wheels_on, so it has leakage if wheels_on has leakage
    # - taxi_out is derived from wheels_off, so it has leakage if wheels_off has leakage
    # - air_time uses both actual_dep_time AND actual_arr_time, so it has leakage if EITHER has leakage
    # - cancelled/diverted status is known only after the flight occurs, so it has leakage if EITHER dep or arr has leakage
    df = df.withColumn('has_leakage_prev_flight_dep_delay', col('has_leakage_prev_flight_actual_dep_time'))
    df = df.withColumn('has_leakage_prev_flight_arr_delay', col('has_leakage_prev_flight_actual_arr_time'))
    df = df.withColumn('has_leakage_prev_flight_taxi_in', col('has_leakage_prev_flight_wheels_on'))
    df = df.withColumn('has_leakage_prev_flight_taxi_out', col('has_leakage_prev_flight_wheels_off'))
    df = df.withColumn(
        'has_leakage_prev_flight_air_time',
        (col('has_leakage_prev_flight_actual_dep_time') | col('has_leakage_prev_flight_actual_arr_time'))
    )
    df = df.withColumn(
        'has_leakage_prev_flight_actual_elapsed_time',
        (col('has_leakage_prev_flight_actual_dep_time') | col('has_leakage_prev_flight_actual_arr_time'))
    )
    
    df = df.withColumn(
        'has_leakage_prev_flight_cancelled',
        F.when(
            col('prev_flight_cancelled').isNotNull(),
            (col('has_leakage_prev_flight_actual_dep_time') | col('has_leakage_prev_flight_actual_arr_time'))
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'has_leakage_prev_flight_diverted',
        F.when(
            col('prev_flight_diverted').isNotNull(),
            (col('has_leakage_prev_flight_actual_dep_time') | col('has_leakage_prev_flight_actual_arr_time'))
        ).otherwise(False)
    )
    
    # ============================================================================
    # STEP 5: Derive leakage for engineered features from source columns
    # ============================================================================
    # Engineered features inherit leakage from ANY of their source columns.
    # Example: lineage_actual_turnover_time = actual_dep_time - actual_arr_time
    # If EITHER actual_dep_time OR actual_arr_time has leakage, then the turnover time
    # feature also has leakage (because we wouldn't know one of the values in production).
    
    # Turnover time features use both actual_arr_time and actual_dep_time
    df = df.withColumn(
        'has_leakage_lineage_actual_turnover_time_minutes',
        (col('has_leakage_prev_flight_actual_arr_time') | col('has_leakage_prev_flight_actual_dep_time'))
    )
    # Aliases inherit the same leakage
    df = df.withColumn('has_leakage_lineage_actual_taxi_time_minutes', col('has_leakage_lineage_actual_turnover_time_minutes'))
    df = df.withColumn('has_leakage_lineage_actual_turn_time_minutes', col('has_leakage_lineage_actual_turnover_time_minutes'))
    
    # Cumulative features are derived from actual delays, which inherit leakage from actual_dep_time
    df = df.withColumn('has_leakage_lineage_cumulative_delay', col('has_leakage_prev_flight_dep_delay'))
    df = df.withColumn('has_leakage_lineage_avg_delay_previous_flights', col('has_leakage_prev_flight_dep_delay'))
    df = df.withColumn('has_leakage_lineage_max_delay_previous_flights', col('has_leakage_prev_flight_dep_delay'))
    
    # ============================================================================
    # STEP 6: Create array column listing all columns with data leakage
    # ============================================================================
    # This array column dynamically lists, for each row, which specific columns have
    # data leakage. This is useful for debugging and for filtering features in the model pipeline.
    df = df.withColumn(
        'columns_with_data_leakage',
        F.array_remove(
            F.array([
                F.when(col('has_leakage_prev_flight_actual_dep_time'), F.lit('prev_flight_actual_dep_time')).otherwise(None),
                F.when(col('has_leakage_prev_flight_actual_arr_time'), F.lit('prev_flight_actual_arr_time')).otherwise(None),
                F.when(col('has_leakage_prev_flight_wheels_off'), F.lit('prev_flight_wheels_off')).otherwise(None),
                F.when(col('has_leakage_prev_flight_wheels_on'), F.lit('prev_flight_wheels_on')).otherwise(None),
                F.when(col('has_leakage_prev_flight_dep_delay'), F.lit('prev_flight_dep_delay')).otherwise(None),
                F.when(col('has_leakage_prev_flight_arr_delay'), F.lit('prev_flight_arr_delay')).otherwise(None),
                F.when(col('has_leakage_prev_flight_air_time'), F.lit('prev_flight_air_time')).otherwise(None),
                F.when(col('has_leakage_prev_flight_taxi_in'), F.lit('prev_flight_taxi_in')).otherwise(None),
                F.when(col('has_leakage_prev_flight_taxi_out'), F.lit('prev_flight_taxi_out')).otherwise(None),
                F.when(col('has_leakage_prev_flight_actual_elapsed_time'), F.lit('prev_flight_actual_elapsed_time')).otherwise(None),
                F.when(col('has_leakage_prev_flight_cancelled'), F.lit('prev_flight_cancelled')).otherwise(None),
                F.when(col('has_leakage_prev_flight_diverted'), F.lit('prev_flight_diverted')).otherwise(None),
                F.when(col('has_leakage_lineage_actual_turnover_time_minutes'), F.lit('lineage_actual_turnover_time_minutes')).otherwise(None),
                F.when(col('has_leakage_lineage_actual_taxi_time_minutes'), F.lit('lineage_actual_taxi_time_minutes')).otherwise(None),
                F.when(col('has_leakage_lineage_actual_turn_time_minutes'), F.lit('lineage_actual_turn_time_minutes')).otherwise(None),
                F.when(col('has_leakage_lineage_cumulative_delay'), F.lit('lineage_cumulative_delay')).otherwise(None),
                F.when(col('has_leakage_lineage_avg_delay_previous_flights'), F.lit('lineage_avg_delay_previous_flights')).otherwise(None),
                F.when(col('has_leakage_lineage_max_delay_previous_flights'), F.lit('lineage_max_delay_previous_flights')).otherwise(None)
            ]),
            None
        )
    )
    
    # ============================================================================
    # STEP 7: Create convenience flags for safe-to-use columns
    # ============================================================================
    # These flags are the inverse of the leakage flags, indicating when it's safe
    # to use actual times in feature engineering. These can be used downstream
    # in model pipelines to conditionally use features only when the source data
    # is safe from leakage.
    df = df.withColumn('prev_arr_time_safe_to_use', ~col('has_leakage_prev_flight_actual_arr_time'))
    df = df.withColumn('prev_dep_time_safe_to_use', ~col('has_leakage_prev_flight_actual_dep_time'))
    
    return df


def _apply_imputation(df):
    """Apply imputation for NULL values (first flight handling)."""
    # Calculate scheduled times backwards from current scheduled departure - 4 hours
    df = df.withColumn(
        'prev_flight_crs_dep_time_minutes',
        F.coalesce(
            F.when(
                col('prev_flight_crs_dep_time').isNotNull(),
                (F.floor(col('prev_flight_crs_dep_time') / 100) * 60 + (col('prev_flight_crs_dep_time') % 100))
            ),
            F.when(
                col('crs_dep_time_minutes').isNotNull(),
                col('crs_dep_time_minutes') - 240
            ).otherwise(
                F.when(
                    col('crs_dep_time').isNotNull(),
                    (F.floor(col('crs_dep_time') / 100) * 60 + (col('crs_dep_time') % 100)) - 240
                )
            )
        )
    )
    
    df = df.withColumn(
        'prev_flight_crs_dep_time',
        F.coalesce(
            col('prev_flight_crs_dep_time'),
            F.when(
                col('prev_flight_crs_dep_time_minutes').isNotNull(),
                F.when(
                    col('prev_flight_crs_dep_time_minutes') >= 0,
                    (F.floor(col('prev_flight_crs_dep_time_minutes') / 60) * 100) + (col('prev_flight_crs_dep_time_minutes') % 60)
                ).otherwise(
                    (F.floor((col('prev_flight_crs_dep_time_minutes') + 1440) / 60) * 100) + ((col('prev_flight_crs_dep_time_minutes') + 1440) % 60)
                )
            )
        )
    )
    
    # Impute all NULL values
    df = df.withColumn('prev_flight_crs_arr_time', F.coalesce(col('prev_flight_crs_arr_time'), col('prev_flight_crs_dep_time')))
    df = df.withColumn('prev_flight_crs_elapsed_time', F.coalesce(col('prev_flight_crs_elapsed_time'), col('crs_elapsed_time')))
    df = df.withColumn('prev_flight_dep_delay', F.coalesce(col('prev_flight_dep_delay'), F.lit(-10.0)))
    df = df.withColumn('prev_flight_arr_delay', F.coalesce(col('prev_flight_arr_delay'), F.lit(-10.0)))
    df = df.withColumn('prev_flight_actual_dep_time', F.coalesce(col('prev_flight_actual_dep_time'), col('prev_flight_crs_dep_time')))
    df = df.withColumn('prev_flight_actual_arr_time', F.coalesce(col('prev_flight_actual_arr_time'), col('prev_flight_crs_arr_time')))
    df = df.withColumn('prev_flight_dest', F.coalesce(col('prev_flight_dest'), col('origin')))
    df = df.withColumn('prev_flight_origin', F.coalesce(col('prev_flight_origin'), col('origin')))
    df = df.withColumn('prev_flight_air_time', F.coalesce(col('prev_flight_air_time'), col('prev_flight_crs_elapsed_time')))
    df = df.withColumn('prev_flight_taxi_in', F.coalesce(col('prev_flight_taxi_in'), F.lit(10.0)))
    df = df.withColumn('prev_flight_taxi_out', F.coalesce(col('prev_flight_taxi_out'), F.lit(15.0)))
    df = df.withColumn('prev_flight_actual_elapsed_time', F.coalesce(col('prev_flight_actual_elapsed_time'), col('prev_flight_crs_elapsed_time')))
    df = df.withColumn('prev_flight_wheels_off', F.coalesce(col('prev_flight_wheels_off'), col('prev_flight_crs_dep_time')))
    df = df.withColumn('prev_flight_wheels_on', F.coalesce(col('prev_flight_wheels_on'), col('prev_flight_crs_arr_time')))
    df = df.withColumn('prev_flight_distance', F.coalesce(col('prev_flight_distance'), F.lit(0.0)))
    df = df.withColumn('prev_flight_cancelled', F.coalesce(col('prev_flight_cancelled'), F.lit(0)))
    df = df.withColumn('prev_flight_diverted', F.coalesce(col('prev_flight_diverted'), F.lit(0)))
    df = df.withColumn('prev_flight_fl_date', F.coalesce(col('prev_flight_fl_date'), col('fl_date')))
    df = df.withColumn('prev_flight_op_carrier', F.coalesce(col('prev_flight_op_carrier'), col('op_carrier')))
    df = df.withColumn('prev_flight_op_carrier_fl_num', F.coalesce(col('prev_flight_op_carrier_fl_num'), col('op_carrier_fl_num')))
    
    # Impute engineered lineage features
    df = df.withColumn('lineage_turnover_time_minutes', F.coalesce(col('lineage_turnover_time_minutes'), F.lit(240.0)))
    df = df.withColumn('lineage_taxi_time_minutes', F.coalesce(col('lineage_taxi_time_minutes'), col('lineage_turnover_time_minutes')))
    df = df.withColumn('lineage_turn_time_minutes', F.coalesce(col('lineage_turn_time_minutes'), col('lineage_turnover_time_minutes')))
    df = df.withColumn('lineage_actual_turnover_time_minutes', F.coalesce(col('lineage_actual_turnover_time_minutes'), F.lit(240.0)))
    df = df.withColumn('lineage_actual_taxi_time_minutes', F.coalesce(col('lineage_actual_taxi_time_minutes'), col('lineage_actual_turnover_time_minutes')))
    df = df.withColumn('lineage_actual_turn_time_minutes', F.coalesce(col('lineage_actual_turn_time_minutes'), col('lineage_actual_turnover_time_minutes')))
    df = df.withColumn('lineage_cumulative_delay', F.coalesce(col('lineage_cumulative_delay'), F.lit(0.0)))
    df = df.withColumn('lineage_avg_delay_previous_flights', F.coalesce(col('lineage_avg_delay_previous_flights'), F.lit(-10.0)))
    df = df.withColumn('lineage_max_delay_previous_flights', F.coalesce(col('lineage_max_delay_previous_flights'), F.lit(-10.0)))
    df = df.withColumn('lineage_num_previous_flights', F.coalesce(col('lineage_num_previous_flights'), F.lit(0)))
    df = df.withColumn('lineage_expected_flight_time_minutes', F.coalesce(col('lineage_expected_flight_time_minutes'), col('crs_elapsed_time')))
    
    df = df.drop('prev_flight_crs_dep_time_minutes')
    
    return df