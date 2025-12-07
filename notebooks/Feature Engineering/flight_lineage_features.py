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
        - prediction_cutoff_minutes: Cutoff time for data leakage detection
    
    Notes:
    ------
    - All flights are preserved (no rows dropped)
    - First flights (lineage_rank == 1) get imputed values
    - Data leakage flags are added for all risky columns
    - See FLIGHT_LINEAGE_JOIN_DESIGN.md for complete documentation
    """
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
    
    # Step 6: Compute Actual Turnover Time (with data leakage check)
    print("\nStep 6: Computing actual turnover time (with data leakage check)...")
    df = _compute_turnover_time(df)
    print("✓ Actual turnover time computed with data leakage check")
    
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
    
    print("\n" + "=" * 60)
    print("✓ FLIGHT LINEAGE JOIN COMPLETE")
    print("=" * 60)
    print(f"\nNew columns added: ~38 lineage features")
    print(f"All flights preserved - no rows dropped")
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
    """Compute actual turnover time with data leakage check."""
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
    
    # Check data leakage: prev_arr_time must be <= crs_dep_time - 2 hours (120 minutes)
    df = df.withColumn(
        'prev_arr_time_safe_to_use',
        F.when(
            (col('prev_flight_actual_arr_time_minutes').isNotNull()) & (col('crs_dep_time_minutes').isNotNull()),
            col('prev_flight_actual_arr_time_minutes') <= (col('crs_dep_time_minutes') - 120)
        ).otherwise(False)
    )
    
    # Actual turnover time (only compute if safe)
    df = df.withColumn(
        'lineage_actual_turnover_time_minutes',
        F.when(
            (col('prev_arr_time_safe_to_use') == True) &
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
    """Check for data leakage in all risky columns."""
    # Create prediction cutoff
    df = df.withColumn(
        'prediction_cutoff_minutes',
        F.when(col('crs_dep_time_minutes').isNotNull(), col('crs_dep_time_minutes') - 120).otherwise(None)
    )
    
    df = df.withColumn(
        'prediction_cutoff_timestamp',
        F.when(
            col('sched_depart_date_time').isNotNull(),
            F.expr("sched_depart_date_time - INTERVAL 2 HOURS")
        ).otherwise(None)
    )
    
    # Convert all actual time columns to minutes for checking
    df = df.withColumn(
        'prev_flight_actual_dep_time_minutes',
        F.when(
            col('prev_flight_actual_dep_time').isNotNull(),
            (F.floor(col('prev_flight_actual_dep_time') / 100) * 60 + (col('prev_flight_actual_dep_time') % 100))
        ).otherwise(None)
    )
    
    df = df.withColumn(
        'prev_flight_wheels_off_minutes',
        F.when(
            col('prev_flight_wheels_off').isNotNull(),
            (F.floor(col('prev_flight_wheels_off') / 100) * 60 + (col('prev_flight_wheels_off') % 100))
        ).otherwise(None)
    )
    
    df = df.withColumn(
        'prev_flight_wheels_on_minutes',
        F.when(
            col('prev_flight_wheels_on').isNotNull(),
            (F.floor(col('prev_flight_wheels_on') / 100) * 60 + (col('prev_flight_wheels_on') % 100))
        ).otherwise(None)
    )
    
    # Check each risky timestamp column for data leakage
    df = df.withColumn(
        'prev_flight_actual_dep_time_has_leakage',
        F.when(
            (col('prev_flight_actual_dep_time_minutes').isNotNull()) & (col('prediction_cutoff_minutes').isNotNull()),
            col('prev_flight_actual_dep_time_minutes') > col('prediction_cutoff_minutes')
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'prev_flight_actual_arr_time_has_leakage',
        F.when(
            (col('prev_flight_actual_arr_time_minutes').isNotNull()) & (col('prediction_cutoff_minutes').isNotNull()),
            col('prev_flight_actual_arr_time_minutes') > col('prediction_cutoff_minutes')
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'prev_flight_wheels_off_has_leakage',
        F.when(
            (col('prev_flight_wheels_off_minutes').isNotNull()) & (col('prediction_cutoff_minutes').isNotNull()),
            col('prev_flight_wheels_off_minutes') > col('prediction_cutoff_minutes')
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'prev_flight_wheels_on_has_leakage',
        F.when(
            (col('prev_flight_wheels_on_minutes').isNotNull()) & (col('prediction_cutoff_minutes').isNotNull()),
            col('prev_flight_wheels_on_minutes') > col('prediction_cutoff_minutes')
        ).otherwise(False)
    )
    
    # For duration fields, check if their source columns have data leakage
    df = df.withColumn('prev_flight_dep_delay_has_leakage', col('prev_flight_actual_dep_time_has_leakage'))
    df = df.withColumn('prev_flight_arr_delay_has_leakage', col('prev_flight_actual_arr_time_has_leakage'))
    df = df.withColumn('prev_flight_taxi_in_has_leakage', col('prev_flight_wheels_on_has_leakage'))
    df = df.withColumn('prev_flight_taxi_out_has_leakage', col('prev_flight_wheels_off_has_leakage'))
    df = df.withColumn(
        'prev_flight_air_time_has_leakage',
        (col('prev_flight_actual_dep_time_has_leakage') | col('prev_flight_actual_arr_time_has_leakage'))
    )
    df = df.withColumn(
        'prev_flight_actual_elapsed_time_has_leakage',
        (col('prev_flight_actual_dep_time_has_leakage') | col('prev_flight_actual_arr_time_has_leakage'))
    )
    
    df = df.withColumn(
        'prev_flight_cancelled_has_leakage',
        F.when(
            col('prev_flight_cancelled').isNotNull(),
            (col('prev_flight_actual_dep_time_has_leakage') | col('prev_flight_actual_arr_time_has_leakage'))
        ).otherwise(False)
    )
    
    df = df.withColumn(
        'prev_flight_diverted_has_leakage',
        F.when(
            col('prev_flight_diverted').isNotNull(),
            (col('prev_flight_actual_dep_time_has_leakage') | col('prev_flight_actual_arr_time_has_leakage'))
        ).otherwise(False)
    )
    
    # Engineered features using actual times
    df = df.withColumn(
        'lineage_actual_turnover_time_minutes_has_leakage',
        (col('prev_flight_actual_arr_time_has_leakage') | col('prev_flight_actual_dep_time_has_leakage'))
    )
    df = df.withColumn('lineage_actual_taxi_time_minutes_has_leakage', col('lineage_actual_turnover_time_minutes_has_leakage'))
    df = df.withColumn('lineage_actual_turn_time_minutes_has_leakage', col('lineage_actual_turnover_time_minutes_has_leakage'))
    
    # Cumulative features derived from actual delays
    df = df.withColumn('lineage_cumulative_delay_has_leakage', col('prev_flight_dep_delay_has_leakage'))
    df = df.withColumn('lineage_avg_delay_previous_flights_has_leakage', col('prev_flight_dep_delay_has_leakage'))
    df = df.withColumn('lineage_max_delay_previous_flights_has_leakage', col('prev_flight_dep_delay_has_leakage'))
    
    # Create an array column listing all columns that have data leakage
    df = df.withColumn(
        'columns_with_data_leakage',
        F.array_remove(
            F.array([
                F.when(col('prev_flight_actual_dep_time_has_leakage'), F.lit('prev_flight_actual_dep_time')).otherwise(None),
                F.when(col('prev_flight_actual_arr_time_has_leakage'), F.lit('prev_flight_actual_arr_time')).otherwise(None),
                F.when(col('prev_flight_wheels_off_has_leakage'), F.lit('prev_flight_wheels_off')).otherwise(None),
                F.when(col('prev_flight_wheels_on_has_leakage'), F.lit('prev_flight_wheels_on')).otherwise(None),
                F.when(col('prev_flight_dep_delay_has_leakage'), F.lit('prev_flight_dep_delay')).otherwise(None),
                F.when(col('prev_flight_arr_delay_has_leakage'), F.lit('prev_flight_arr_delay')).otherwise(None),
                F.when(col('prev_flight_air_time_has_leakage'), F.lit('prev_flight_air_time')).otherwise(None),
                F.when(col('prev_flight_taxi_in_has_leakage'), F.lit('prev_flight_taxi_in')).otherwise(None),
                F.when(col('prev_flight_taxi_out_has_leakage'), F.lit('prev_flight_taxi_out')).otherwise(None),
                F.when(col('prev_flight_actual_elapsed_time_has_leakage'), F.lit('prev_flight_actual_elapsed_time')).otherwise(None),
                F.when(col('prev_flight_cancelled_has_leakage'), F.lit('prev_flight_cancelled')).otherwise(None),
                F.when(col('prev_flight_diverted_has_leakage'), F.lit('prev_flight_diverted')).otherwise(None),
                F.when(col('lineage_actual_turnover_time_minutes_has_leakage'), F.lit('lineage_actual_turnover_time_minutes')).otherwise(None),
                F.when(col('lineage_actual_taxi_time_minutes_has_leakage'), F.lit('lineage_actual_taxi_time_minutes')).otherwise(None),
                F.when(col('lineage_actual_turn_time_minutes_has_leakage'), F.lit('lineage_actual_turn_time_minutes')).otherwise(None),
                F.when(col('lineage_cumulative_delay_has_leakage'), F.lit('lineage_cumulative_delay')).otherwise(None),
                F.when(col('lineage_avg_delay_previous_flights_has_leakage'), F.lit('lineage_avg_delay_previous_flights')).otherwise(None),
                F.when(col('lineage_max_delay_previous_flights_has_leakage'), F.lit('lineage_max_delay_previous_flights')).otherwise(None)
            ]),
            None
        )
    )
    
    df = df.withColumn('prev_arr_time_safe_to_use', ~col('prev_flight_actual_arr_time_has_leakage'))
    df = df.withColumn('prev_dep_time_safe_to_use', ~col('prev_flight_actual_dep_time_has_leakage'))
    
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

