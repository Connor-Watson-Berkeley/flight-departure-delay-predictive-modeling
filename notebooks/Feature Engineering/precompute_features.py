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
        - lineage_rotation_time_minutes: Rotation time (prev_dep → curr_sched_dep, entire sequence)
          Rotation Time = Air Time + Turnover Time (aviation terminology)
        - Note: Turnover time features already exist (scheduled_lineage_turnover_time_minutes, lineage_actual_turnover_time_minutes)
        - required_time_prev_flight_minutes: Expected air_time + expected_turnover_time (convenience feature)
        - impossible_on_time_flag: Boolean flag when required_time > available_time
        - safe_*: Safe versions of features with intelligent data leakage imputation
          - safe_lineage_rotation_time_minutes: Safe rotation time
          - safe_prev_departure_delay: Safe previous flight departure delay
          - safe_prev_arrival_delay: Safe previous flight arrival delay
          - safe_time_since_prev_arrival: Time between previous flight's arrival and prediction cutoff
          - safe_required_time_prev_flight_minutes: Safe required time
          - safe_impossible_on_time_flag: Safe impossible on-time flag
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
    
    # Step 1.5: Determine dep_delay column name (handle both DEP_DELAY and dep_delay)
    # Standardize on DEP_DELAY (uppercase) to match cv.py, but be flexible for backwards compatibility
    dep_delay_col = None
    if "DEP_DELAY" in df.columns:
        dep_delay_col = "DEP_DELAY"
    elif "dep_delay" in df.columns:
        dep_delay_col = "dep_delay"
    else:
        raise ValueError("Neither 'DEP_DELAY' nor 'dep_delay' column found in DataFrame. Required for flight lineage features.")
    print(f"✓ Using departure delay column: '{dep_delay_col}'")
    
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
    df = _add_previous_flight_data(df, window_spec, dep_delay_col)
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
    df = _compute_cumulative_features(df, tail_num_col, dep_delay_col)
    print("✓ Cumulative features computed")
    
    # Step 7.5: Compute Rotation Time (Previous Departure to Current Scheduled Departure)
    print("\nStep 7.5: Computing rotation time (prev_dep → curr_sched_dep)...")
    df = _compute_time_between_prev_dep_and_sched_dep(df)
    print("✓ Rotation time computed")
    
    # Step 8: Jump Detection
    print("\nStep 8: Detecting jumps (aircraft repositioning)...")
    df = df.withColumn(
        'lineage_is_jump',
        F.when(col('lineage_rank') == 1, F.lit(False))
        .when(col('prev_flight_dest') != col('origin'), F.lit(True))
        .otherwise(F.lit(False))
    )
    print("✓ Jump detection complete")
    
    # Step 8.5: Ensure sched_depart_date_time exists (needed for data leakage check)
    # Handle both CUSTOM (sched_depart_date_time) and OTPW (sched_depart_date_time_utc)
    if 'sched_depart_date_time' not in df.columns:
        if 'sched_depart_date_time_utc' in df.columns:
            # Use UTC version from OTPW
            df = df.withColumn('sched_depart_date_time', col('sched_depart_date_time_utc'))
        else:
            # Create from fl_date and crs_dep_time
            df = df.withColumn(
                'sched_depart_date_time',
                F.when(
                    col('fl_date').isNotNull() & col('crs_dep_time').isNotNull(),
                    F.to_timestamp(
                        F.concat(col('fl_date'), F.lpad(col('crs_dep_time').cast('string'), 4, '0')),
                        'yyyy-MM-ddHHmm'
                    )
                ).otherwise(None)
            )
    
    # Step 9: Check Data Leakage for All Risky Columns
    print("\nStep 9: Checking data leakage for all risky columns...")
    df = _check_data_leakage(df)
    print("✓ Data leakage checks complete")
    
    # Step 9.5: Compute Safe Features (Intelligent Data Leakage Imputation)
    print("\nStep 9.5: Computing safe features with intelligent data leakage imputation...")
    df = _compute_safe_features(df)
    print("✓ Safe features computed")
    
    # Step 9.6: Compute Required Time Features (for Model Learning)
    print("\nStep 9.6: Computing required time features (expected_air_time + expected_turnover_time)...")
    df = _compute_required_time_features(df)
    print("✓ Required time features computed")
    
    # Step 9.7: Compute Rolling Average Delay Features
    print("\nStep 9.7: Computing rolling average delay features (24hr, 7-day, 30-day)...")
    df = _compute_rolling_average_delays(df, tail_num_col, dep_delay_col)
    print("✓ Rolling average delay features computed")
    
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
    print(f"\nNew columns added: ~54+ lineage features (including rolling averages)")
    print(f"\n📊 DATA LEAKAGE-FREE FEATURES (Safe for Training):")
    print(f"  Rolling Average Delay Features:")
    print(f"    - tail_num_rolling_avg_delay_24h: 24-hour rolling average delay per tail number")
    print(f"    - tail_num_rolling_avg_delay_7d: 7-day rolling average delay per tail number")
    print(f"    - tail_num_rolling_avg_delay_30d: 30-day rolling average delay per tail number")
    print(f"    - origin_rolling_avg_delay_24h: 24-hour rolling average delay per origin airport")
    print(f"    - origin_rolling_avg_delay_7d: 7-day rolling average delay per origin airport")
    print(f"    - origin_rolling_avg_delay_30d: 30-day rolling average delay per origin airport")
    print(f"      (All rolling averages only include flights that departed >= 2 hours before current flight's scheduled departure)")
    print(f"  Rotation Time Features:")
    print(f"    - safe_lineage_rotation_time_minutes: Safe rotation time (handles data leakage)")
    print(f"    - scheduled_lineage_rotation_time_minutes: Scheduled rotation time (prev_crs_dep → curr_crs_dep)")
    print(f"  Turnover Time Features:")
    print(f"    - scheduled_lineage_turnover_time_minutes: Scheduled turnover time (prev_crs_arr → curr_crs_dep)")
    print(f"  Air Time Features:")
    print(f"    - prev_flight_scheduled_flight_time_minutes: Scheduled flight time for previous flight (arr - dep)")
    print(f"    - prev_flight_crs_elapsed_time: Scheduled air time for previous flight (from raw data, for backward compatibility)")
    print(f"    - crs_elapsed_time: Scheduled air time for current flight")
    print(f"  Required Time Features:")
    print(f"    - required_time_prev_flight_minutes: Expected air_time + expected_turnover_time")
    print(f"      (Uses conditional expected values if available, otherwise scheduled times)")
    print(f"    - safe_required_time_prev_flight_minutes: Safe version (always data leakage-free)")
    print(f"  Delay Features:")
    print(f"    - safe_prev_departure_delay: Safe previous flight departure delay (handles data leakage)")
    print(f"      Uses actual delay if available, or computes delay from cutoff timestamp if not yet departed")
    print(f"    - safe_prev_arrival_delay: Safe previous flight arrival delay (handles data leakage)")
    print(f"      Uses actual delay if available, or computes delay from cutoff timestamp if not yet arrived")
    print(f"    - safe_time_since_prev_arrival: Time between previous flight's arrival and prediction cutoff")
    print(f"      Uses actual arrival if available, or estimates based on scheduled arrival and cutoff")
    print(f"  Other Safe Features:")
    print(f"    - safe_impossible_on_time_flag: Safe binary flag (required_time > rotation_time)")
    print(f"\n⚠️  FEATURES WITH DATA LEAKAGE (Use with caution):")
    print(f"    - lineage_rotation_time_minutes: Uses actual prev_dep (may have leakage)")
    print(f"    - lineage_actual_turnover_time_minutes: Uses actual times (may have leakage)")
    print(f"    - impossible_on_time_flag: Uses rotation_time (may have leakage)")
    print(f"\n📈 Model Learning Objective:")
    print(f"    departure_delay ≈ max(0, required_time - rotation_time)")
    print(f"    Rotation Time = Air Time + Turnover Time (aviation terminology)")
    print(f"\nAll flights preserved - no rows dropped")
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


def _add_previous_flight_data(df, window_spec, dep_delay_col):
    """Add previous flight data using LAG window function."""
    # Core Previous Flight Information
    df = df.withColumn('prev_flight_origin', F.lag('origin', 1).over(window_spec))
    df = df.withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))
    df = df.withColumn('prev_flight_actual_dep_time', F.lag('dep_time', 1).over(window_spec))
    df = df.withColumn('prev_flight_actual_arr_time', F.lag('arr_time', 1).over(window_spec))
    df = df.withColumn('prev_flight_dep_delay', F.lag(dep_delay_col, 1).over(window_spec))
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
    
    # Previous Flight Weather Features (CRITICAL for predictive modeling)
    # These capture weather conditions during the previous flight, which are highly predictive
    # for current flight delays, air time, and duration predictions
    # NOTE: These will only be available if weather data is joined (e.g., OTPW dataset)
    # If columns don't exist, they will be NULL (handled gracefully by imputation)
    
    # Hourly Weather Variables (most granular, most predictive)
    weather_cols_to_lag = [
        # Core hourly weather (most important for flight operations)
        'hourlyprecipitation',
        'hourlywindspeed',
        'hourlywinddirection',
        'hourlyvisibility',
        'hourlydrybulbtemperature',
        'hourlydewpointtemperature',
        'hourlyrelativehumidity',
        'hourlysealevelpressure',
        'hourlystationpressure',
        'hourlyaltimetersetting',
        'hourlywetbulbtemperature',
        'hourlywindgustspeed',
        'hourlyskyconditions',
        'hourlypresentweathertype',
        'hourlypressurechange',
        'hourlypressuretendency',
        
        # Daily Weather Variables (aggregate conditions)
        'dailyprecipitation',
        'dailyaveragewindspeed',
        'dailypeakwindspeed',
        'dailymaximumdrybulbtemperature',
        'dailyminimumdrybulbtemperature',
        'dailyaveragedrybulbtemperature',
        'dailysnowfall',
        'dailysnowdepth',
        'dailyweather',
        
        # Monthly Weather Variables (long-term patterns)
        'monthlyaveragerh',
        'monthlydeparturefromnormalaveragetemperature',
    ]
    
    # Only lag weather columns that actually exist in the DataFrame
    # This allows the function to work with or without weather data
    existing_weather_cols = [col for col in weather_cols_to_lag if col in df.columns]
    missing_weather_cols = [col for col in weather_cols_to_lag if col not in df.columns]
    
    if missing_weather_cols:
        print(f"  ⚠ Note: {len(missing_weather_cols)} weather columns not found (will be skipped):")
        print(f"    Missing: {missing_weather_cols[:10]}{'...' if len(missing_weather_cols) > 10 else ''}")
        print(f"    This is expected if weather data is not joined (e.g., CUSTOM dataset without weather)")
        print(f"    Previous flight weather features will only be available for flights with weather data")
    
    if existing_weather_cols:
        print(f"  ✓ Lagging {len(existing_weather_cols)} weather columns as previous flight features...")
        for weather_col in existing_weather_cols:
            prev_col_name = f'prev_flight_{weather_col}'
            df = df.withColumn(prev_col_name, F.lag(weather_col, 1).over(window_spec))
        print(f"    Created {len(existing_weather_cols)} prev_flight_* weather features")
    else:
        print(f"  ⚠ No weather columns found - previous flight weather features will not be available")
        print(f"    This is expected if weather data is not joined")
    
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


def _compute_cumulative_features(df, tail_num_col, dep_delay_col):
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
    
    # Convert prev_flight_crs_dep_time to minutes for rotation time calculation
    df = df.withColumn(
        'prev_flight_crs_dep_time_minutes',
        F.when(
            col('prev_flight_crs_dep_time').isNotNull(),
            (F.floor(col('prev_flight_crs_dep_time') / 100) * 60 + (col('prev_flight_crs_dep_time') % 100))
        ).otherwise(None)
    )
    
    # Compute scheduled flight time for previous flight (prev flight's scheduled arrival - scheduled departure)
    # This represents how long the previous flight was scheduled to be in the air
    # Used for hypothesis: when rotation_time >> scheduled_flight_time, we have more buffer = likely to depart on time
    df = df.withColumn(
        'prev_flight_scheduled_flight_time_minutes',
        F.when(
            (col('prev_flight_crs_arr_time_minutes').isNotNull()) & (col('prev_flight_crs_dep_time_minutes').isNotNull()),
            F.when(
                col('prev_flight_crs_arr_time_minutes') >= col('prev_flight_crs_dep_time_minutes'),
                col('prev_flight_crs_arr_time_minutes') - col('prev_flight_crs_dep_time_minutes')
            ).otherwise(col('prev_flight_crs_arr_time_minutes') + 1440 - col('prev_flight_crs_dep_time_minutes'))
        ).otherwise(None)
    )
    
    # Compute scheduled rotation time (time between prev flight's scheduled departure and current flight's scheduled departure)
    # This is the scheduled version of rotation_time: prev_crs_dep → curr_crs_dep (entire scheduled sequence)
    df = df.withColumn(
        'scheduled_lineage_rotation_time_minutes',
        F.when(
            (col('prev_flight_crs_dep_time_minutes').isNotNull()) & (col('crs_dep_time_minutes').isNotNull()),
            F.when(
                col('crs_dep_time_minutes') >= col('prev_flight_crs_dep_time_minutes'),
                col('crs_dep_time_minutes') - col('prev_flight_crs_dep_time_minutes')
            ).otherwise(col('crs_dep_time_minutes') + 1440 - col('prev_flight_crs_dep_time_minutes'))
        ).otherwise(None)
    )
    
    # Compute scheduled turnover time (time between prev flight's scheduled arrival and current flight's scheduled departure)
    df = df.withColumn(
        'scheduled_lineage_turnover_time_minutes',
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
    
    # Cumulative features: Look at flights BEFORE the immediate previous flight (exclude flight n-1)
    # This reduces data leakage risk since we're not using the immediate previous flight's delay
    # Use -2 instead of -1 to exclude the immediate previous flight
    # Handle dep_delay column - use same column as determined above (DEP_DELAY or dep_delay)
    window_spec_cumulative = Window.partitionBy(tail_num_col).orderBy(F.col('arrival_timestamp').asc_nulls_last()).rowsBetween(Window.unboundedPreceding, -2)
    df = df.withColumn('lineage_cumulative_delay', F.sum(dep_delay_col).over(window_spec_cumulative))
    df = df.withColumn('lineage_num_previous_flights', F.count('*').over(window_spec_cumulative))
    df = df.withColumn('lineage_avg_delay_previous_flights', F.avg(dep_delay_col).over(window_spec_cumulative))
    df = df.withColumn('lineage_max_delay_previous_flights', F.max(dep_delay_col).over(window_spec_cumulative))
    
    return df


def _compute_time_between_prev_dep_and_sched_dep(df):
    """
    Compute rotation time: time between previous actual departure and current scheduled departure.
    
    Rotation Time = time from previous departure to current departure
    This captures the entire sequence: prev_dep → flight → arrival → turnover → sched_dep
    
    Aviation Terminology:
    - Air Time: Time in the air (flight time)
    - Turnover Time: Time from arrival to next departure (ground time)
    - Rotation Time: Time from one departure to the next departure (entire sequence)
    
    Rotation Time = Air Time + Turnover Time
    
    This is a key component in the departure time calculation model:
    departure_time = previous_departure_time + rotation_time
    where rotation_time = air_time + turnover_time
    
    Note: We already have turnover time features (scheduled_lineage_turnover_time_minutes, 
    lineage_actual_turnover_time_minutes) which measure prev_arr → curr_dep, so we don't
    need a separate prev_arr → sched_dep feature (it would be redundant with turnover time).
    
    Note: Safe version (with data leakage handling) is computed in _compute_safe_features()
    after data leakage flags are set.
    """
    # Convert previous actual departure time to minutes (if not already done)
    if 'prev_flight_actual_dep_time_minutes' not in df.columns:
        df = df.withColumn(
            'prev_flight_actual_dep_time_minutes',
            F.when(
                col('prev_flight_actual_dep_time').isNotNull(),
                (F.floor(col('prev_flight_actual_dep_time') / 100) * 60 + (col('prev_flight_actual_dep_time') % 100))
            ).otherwise(None)
        )
    
    # Compute rotation time: scheduled_dep_time - prev_actual_dep_time
    # Rotation time = time from previous departure to current scheduled departure
    # Handle day rollover (if scheduled is earlier in day than previous departure, assume next day)
    df = df.withColumn(
        'lineage_rotation_time_minutes',
        F.when(
            (col('prev_flight_actual_dep_time_minutes').isNotNull()) & 
            (col('crs_dep_time_minutes').isNotNull()),
            F.when(
                col('crs_dep_time_minutes') >= col('prev_flight_actual_dep_time_minutes'),
                col('crs_dep_time_minutes') - col('prev_flight_actual_dep_time_minutes')
            ).otherwise(col('crs_dep_time_minutes') + 1440 - col('prev_flight_actual_dep_time_minutes'))
        ).otherwise(None)
    )
    
    return df


def _compute_safe_features(df):
    """
    Compute safe_ prefixed features that intelligently handle data leakage.
    
    For rotation time, we compute "time from last departure to next departure":
    (1) If previous flight already departed (actual_dep_time <= prediction_cutoff): use actual_dep_time
    (2) If previous flight hasn't departed yet (actual_dep_time > prediction_cutoff):
       - If scheduled_dep_time >= prediction_cutoff (scheduled is in future): use scheduled_dep_time
       - If scheduled_dep_time < prediction_cutoff (scheduled is in past): use "right now" (prediction_cutoff)
    
    Logic: At prediction time (2 hours before scheduled departure = prediction_cutoff), we know:
    - If previous flight already departed (actual_dep_time <= prediction_cutoff): use that actual time
    - If previous flight hasn't departed yet (actual_dep_time > prediction_cutoff):
       - This means actual_dep_time > (next_departure - 2 hours)
       - Use scheduled time if it's in the future (>= cutoff)
       - Use "right now" (cutoff) if scheduled is in the past (< cutoff)
    """
    # Convert scheduled departure to timestamp for comparison (CURRENT flight's scheduled departure)
    df = df.withColumn(
        'sched_dep_timestamp',
        F.when(
            col('sched_depart_date_time').isNotNull(),
            col('sched_depart_date_time')
        ).otherwise(
            F.when(
                col('sched_depart_date_time_utc').isNotNull(),
                col('sched_depart_date_time_utc')
            ).otherwise(
                F.when(
                    col('fl_date').isNotNull() & col('crs_dep_time').isNotNull(),
                    F.to_timestamp(
                        F.concat(col('fl_date'), F.lpad(col('crs_dep_time').cast('string'), 4, '0')),
                        'yyyy-MM-ddHHmm'
                    )
                )
            )
        )
    )
    
    # Convert previous flight's scheduled departure to timestamp for comparison
    df = df.withColumn(
        'prev_flight_sched_dep_timestamp',
        F.when(
            (col('prev_flight_crs_dep_time').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_crs_dep_time').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    # Create safe previous departure timestamp
    df = df.withColumn(
        'safe_prev_flight_dep_timestamp',
        F.when(
            # Case 1: Previous flight already departed (actual_dep_time <= prediction_cutoff) - use actual departure time
            (col('prev_flight_actual_dep_time').isNotNull()) & 
            (col('prev_flight_fl_date').isNotNull()) & 
            (~col('has_leakage_prev_flight_actual_dep_time')),  # No leakage = already departed (actual_dep_time <= cutoff)
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_actual_dep_time').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(
            # Case 2: Previous flight hasn't departed yet (actual_dep_time > prediction_cutoff)
            # This means: at prediction time (2 hours before next departure), we know the previous flight
            # hasn't departed yet, so we use scheduled if in future, or "right now" (cutoff) if in past
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & (col('prev_flight_sched_dep_timestamp').isNotNull()),
                F.when(
                    col('prev_flight_sched_dep_timestamp') >= col('prediction_cutoff_timestamp'),
                    col('prev_flight_sched_dep_timestamp')  # Scheduled is in future, use scheduled
                ).otherwise(col('prediction_cutoff_timestamp'))  # Scheduled is in past, use "right now" (cutoff)
            ).otherwise(col('prev_flight_sched_dep_timestamp'))
        )
    )
    
    # Convert safe timestamp back to minutes for comparison with scheduled departure
    df = df.withColumn(
        'safe_prev_flight_dep_time_minutes',
        F.when(
            col('safe_prev_flight_dep_timestamp').isNotNull(),
            F.hour(col('safe_prev_flight_dep_timestamp')) * 60 + F.minute(col('safe_prev_flight_dep_timestamp'))
        ).otherwise(None)
    )
    
    # Compute safe rotation time using the safe previous departure time
    df = df.withColumn(
        'safe_lineage_rotation_time_minutes',
        F.when(
            (col('safe_prev_flight_dep_time_minutes').isNotNull()) & 
            (col('crs_dep_time_minutes').isNotNull()),
            F.when(
                col('crs_dep_time_minutes') >= col('safe_prev_flight_dep_time_minutes'),
                col('crs_dep_time_minutes') - col('safe_prev_flight_dep_time_minutes')
            ).otherwise(col('crs_dep_time_minutes') + 1440 - col('safe_prev_flight_dep_time_minutes'))
        ).otherwise(
            # Fallback to regular version if safe version unavailable
            col('lineage_rotation_time_minutes')
        )
    )
    
    # Compute safe previous flight departure delay
    # Logic:
    # (1) If previous flight already departed (actual_dep <= cutoff): use actual departure delay
    # (2) If previous flight hasn't departed yet (actual_dep > cutoff):
    #     - If scheduled_dep < cutoff: use (cutoff - scheduled_dep) in minutes (at least delayed until now)
    #     - If scheduled_dep >= cutoff: use 0 (scheduled is in future, no delay yet)
    # (3) Otherwise: fallback to 0 or imputed value
    df = df.withColumn(
        'safe_prev_departure_delay',
        F.when(
            # Case 1: Previous flight already departed (actual_dep <= cutoff) - use actual delay
            (col('prev_flight_dep_delay').isNotNull()) & 
            (~col('has_leakage_prev_flight_actual_dep_time')),  # No leakage = already departed
            col('prev_flight_dep_delay')
        ).otherwise(
            # Case 2: Previous flight hasn't departed yet (actual_dep > cutoff or NULL)
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & 
                (col('prev_flight_sched_dep_timestamp').isNotNull()),
                F.when(
                    # Scheduled departure is before cutoff: compute delay as (cutoff - scheduled_dep) in minutes
                    col('prev_flight_sched_dep_timestamp') < col('prediction_cutoff_timestamp'),
                    # Calculate difference in minutes: (cutoff - scheduled_dep)
                    (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
                     F.unix_timestamp(col('prev_flight_sched_dep_timestamp'))) / 60.0
                ).otherwise(
                    # Scheduled departure is after cutoff: no delay yet (scheduled is in future)
                    F.lit(0.0)
                )
            ).otherwise(
                # Fallback: if we can't compute, use 0 (no delay information available)
                F.lit(0.0)
            )
        )
    )
    
    # Convert previous flight's scheduled arrival to timestamp for comparison
    df = df.withColumn(
        'prev_flight_sched_arr_timestamp',
        F.when(
            (col('prev_flight_crs_arr_time').isNotNull()) & (col('prev_flight_fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('prev_flight_fl_date'), F.lpad(col('prev_flight_crs_arr_time').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    # Compute safe previous flight arrival delay
    # Logic:
    # (1) If previous flight already arrived (actual_arr <= cutoff): use actual arrival delay
    # (2) If previous flight hasn't arrived yet (actual_arr > cutoff):
    #     - If scheduled_arr < cutoff: use (cutoff - scheduled_arr) in minutes (at least delayed until now)
    #     - If scheduled_arr >= cutoff: use 0 (scheduled is in future, no delay yet)
    # (3) Otherwise: fallback to 0 or imputed value
    df = df.withColumn(
        'safe_prev_arrival_delay',
        F.when(
            # Case 1: Previous flight already arrived (actual_arr <= cutoff) - use actual delay
            (col('prev_flight_arr_delay').isNotNull()) & 
            (~col('has_leakage_prev_flight_actual_arr_time')),  # No leakage = already arrived
            col('prev_flight_arr_delay')
        ).otherwise(
            # Case 2: Previous flight hasn't arrived yet (actual_arr > cutoff or NULL)
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & 
                (col('prev_flight_sched_arr_timestamp').isNotNull()),
                F.when(
                    # Scheduled arrival is before cutoff: compute delay as (cutoff - scheduled_arr) in minutes
                    col('prev_flight_sched_arr_timestamp') < col('prediction_cutoff_timestamp'),
                    # Calculate difference in minutes: (cutoff - scheduled_arr)
                    (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
                     F.unix_timestamp(col('prev_flight_sched_arr_timestamp'))) / 60.0
                ).otherwise(
                    # Scheduled arrival is after cutoff: no delay yet (scheduled is in future)
                    F.lit(0.0)
                )
            ).otherwise(
                # Fallback: if we can't compute, use 0 (no delay information available)
                F.lit(0.0)
            )
        )
    )
    
    return df


def _compute_required_time_features(df):
    """
    Compute required time for previous flight completion (expected_air_time + expected_turnover_time).
    
    This is a convenience feature that combines expected air time and expected turnover time.
    The model will learn the relationship:
    
    departure_delay ≈ max(0, required_time_prev_flight_minutes - lineage_rotation_time_minutes)
    
    Or equivalently, the model will learn:
    departure_delay ≈ max(0, -time_buffer) where:
    time_buffer = lineage_rotation_time_minutes - required_time_prev_flight_minutes
    
    Note: We don't compute time_buffer_minutes explicitly as it's a linear combination the model can learn.
    However, we compute required_time as a convenience feature to make the relationship clearer.
    
    Uses conditional expected values from ConditionalExpectedValuesEstimator if available,
    otherwise falls back to scheduled times. The ConditionalExpectedValuesEstimator should be
    run before this function to add these columns:
    - expected_air_time_route_temporal_minutes (temporal, Prophet-based, preferred)
    - expected_air_time_route_minutes (route-based baseline)
    - expected_air_time_aircraft_minutes (aircraft-based)
    - expected_turnover_time_temporal_minutes (temporal, Prophet-based, preferred)
    - expected_turnover_time_carrier_airport_minutes (carrier-airport baseline)
    - expected_turnover_time_aircraft_minutes (aircraft-based)
    
    REQUIRED COLUMNS (must exist - function will raise error if missing):
    - lineage_rotation_time_minutes
    - safe_lineage_rotation_time_minutes
    - At least one of: prev_flight_crs_elapsed_time, crs_elapsed_time (for air time fallback)
    - scheduled_lineage_turnover_time_minutes (for turnover time fallback)
    """
    # Validate required columns exist - fail fast if missing
    available_cols = set(df.columns)
    required_cols = [
        'lineage_rotation_time_minutes',
        'safe_lineage_rotation_time_minutes'
    ]
    missing_cols = [c for c in required_cols if c not in available_cols]
    # Check for scheduled turnover time (required)
    if 'scheduled_lineage_turnover_time_minutes' not in available_cols:
        missing_cols.append('scheduled_lineage_turnover_time_minutes')
    if missing_cols:
        raise ValueError(
            f"Missing required columns in _compute_required_time_features: {missing_cols}. "
            f"Available columns: {sorted(list(available_cols))[:50]}..."
        )
    
    # Validate at least one air time fallback column exists
    if 'prev_flight_crs_elapsed_time' not in available_cols and 'crs_elapsed_time' not in available_cols:
        raise ValueError(
            "Missing required columns for air time fallback: neither 'prev_flight_crs_elapsed_time' "
            "nor 'crs_elapsed_time' found in DataFrame."
        )
    
    # Build coalesce expression for expected air time
    # Order of preference: temporal (Prophet) > route > aircraft > scheduled
    # Only include conditional expected value columns if they exist (from ConditionalExpectedValuesEstimator)
    expected_air_time_parts = []
    if 'expected_air_time_route_temporal_minutes' in available_cols:
        expected_air_time_parts.append(col('expected_air_time_route_temporal_minutes'))
    if 'expected_air_time_route_minutes' in available_cols:
        expected_air_time_parts.append(col('expected_air_time_route_minutes'))
    if 'expected_air_time_aircraft_minutes' in available_cols:
        expected_air_time_parts.append(col('expected_air_time_aircraft_minutes'))
    # Always include fallback to scheduled (validated above)
    if 'prev_flight_crs_elapsed_time' in available_cols:
        expected_air_time_parts.append(col('prev_flight_crs_elapsed_time'))
    else:
        expected_air_time_parts.append(col('crs_elapsed_time'))
    
    # Build coalesce expression
    expected_air_time_col = F.coalesce(*expected_air_time_parts)
    
    # Build coalesce expression for expected turnover time
    # Order of preference: temporal (Prophet) > carrier-airport > aircraft > scheduled turnover
    # Only include conditional expected value columns if they exist (from ConditionalExpectedValuesEstimator)
    expected_turnover_time_parts = []
    if 'expected_turnover_time_temporal_minutes' in available_cols:
        expected_turnover_time_parts.append(col('expected_turnover_time_temporal_minutes'))
    if 'expected_turnover_time_carrier_airport_minutes' in available_cols:
        expected_turnover_time_parts.append(col('expected_turnover_time_carrier_airport_minutes'))
    if 'expected_turnover_time_aircraft_minutes' in available_cols:
        expected_turnover_time_parts.append(col('expected_turnover_time_aircraft_minutes'))
    # Always include fallback to scheduled turnover time (validated above)
    expected_turnover_time_parts.append(col('scheduled_lineage_turnover_time_minutes'))
    
    # Build coalesce expression
    expected_turnover_time_col = F.coalesce(*expected_turnover_time_parts)
    
    # Compute required time (flight time + turnover time)
    # This is a convenience feature - the model can also compute this from the components
    df = df.withColumn(
        'required_time_prev_flight_minutes',
        expected_air_time_col + expected_turnover_time_col
    )
    
    # Safe version uses the same expected values (expected values are safe, computed from historical data)
    df = df.withColumn(
        'safe_required_time_prev_flight_minutes',
        expected_air_time_col + expected_turnover_time_col
    )
    
    # Flag impossible on-time scenarios: when required_time > available rotation_time
    # This is a non-linear binary indicator that might be useful for the model
    df = df.withColumn(
        'impossible_on_time_flag',
        col('required_time_prev_flight_minutes') > col('lineage_rotation_time_minutes')
    )
    
    df = df.withColumn(
        'safe_impossible_on_time_flag',
        col('safe_required_time_prev_flight_minutes') > col('safe_lineage_rotation_time_minutes')
    )
    
    return df


def _compute_rolling_average_delays(df, tail_num_col, dep_delay_col):
    """
    Compute rolling average delay features per tail number and per airport.
    
    Features computed:
    - 24-hour rolling average departure delay per tail number
    - 7-day rolling average departure delay per tail number
    - 30-day rolling average departure delay per tail number
    - 24-hour rolling average departure delay per origin airport
    - 7-day rolling average departure delay per origin airport
    - 30-day rolling average departure delay per origin airport
    
    Data leakage prevention:
    - Only includes flights where actual_dep_time <= prediction_cutoff_timestamp
    - prediction_cutoff_timestamp = scheduled_departure_time - 2 hours
    - This ensures we only use information available 2 hours before the current flight's scheduled departure
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame with flight data, must have:
        - prediction_cutoff_timestamp (from _check_data_leakage)
        - dep_time, fl_date, crs_dep_time, origin, dest
        - dep_delay_col (DEP_DELAY or dep_delay)
    
    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with rolling average delay features added
    """
    # Step 1: Create actual departure timestamp for all flights
    # This is needed to filter flights that departed before the prediction cutoff
    df = df.withColumn(
        'actual_dep_timestamp',
        F.when(
            (col('dep_time').isNotNull()) & (col('fl_date').isNotNull()),
            F.to_timestamp(
                F.concat(col('fl_date'), F.lpad(col('dep_time').cast('string'), 4, '0')),
                'yyyy-MM-ddHHmm'
            )
        ).otherwise(None)
    )
    
    # Step 2: Create scheduled departure timestamp if not already present
    # Handle both CUSTOM (sched_depart_date_time) and OTPW (sched_depart_date_time_utc)
    if 'sched_depart_date_time' not in df.columns:
        # Check if OTPW version exists
        if 'sched_depart_date_time_utc' in df.columns:
            # Use UTC version from OTPW
            df = df.withColumn(
                'sched_depart_date_time',
                col('sched_depart_date_time_utc')
            )
        else:
            # Create from fl_date and crs_dep_time
            df = df.withColumn(
                'sched_depart_date_time',
                F.when(
                    col('fl_date').isNotNull() & col('crs_dep_time').isNotNull(),
                    F.to_timestamp(
                        F.concat(col('fl_date'), F.lpad(col('crs_dep_time').cast('string'), 4, '0')),
                        'yyyy-MM-ddHHmm'
                    )
                ).otherwise(None)
            )
    
    # Ensure prediction_cutoff_timestamp exists (should be created in _check_data_leakage)
    if 'prediction_cutoff_timestamp' not in df.columns:
        df = df.withColumn(
            'prediction_cutoff_timestamp',
            F.when(
                col('sched_depart_date_time').isNotNull(),
                F.expr("sched_depart_date_time - INTERVAL 2 HOURS")
            ).otherwise(None)
        )
    
    # Step 3: Create a timestamp for window ordering (use actual if available, otherwise scheduled)
    df = df.withColumn(
        'dep_timestamp_for_window',
        F.coalesce(col('actual_dep_timestamp'), col('sched_depart_date_time'))
    )
    
    # Step 4: Compute rolling averages using self-join for accurate data leakage prevention
    # 
    # CRITICAL DATA LEAKAGE PREVENTION:
    # We need to compare: previous_row.actual_dep_timestamp <= current_row.prediction_cutoff_timestamp
    # This ensures we only use flights that actually departed at least 2 hours before
    # the current flight's scheduled departure.
    #
    # Window functions can't directly reference the current row's value when filtering previous rows,
    # so we use a self-join approach for accuracy.
    #
    # Strategy: For each current flight, we join to all previous flights (same tail_num or origin)
    # where the previous flight's actual_dep_timestamp <= current flight's prediction_cutoff_timestamp
    # and within the time window (24h, 7d, 30d).
    
    # Add a deterministic row ID for joining
    # Use row_number() with a stable ordering to ensure determinism across shuffles
    # Order by prediction_cutoff_timestamp (or scheduled departure) and other stable columns
    row_id_window = Window.orderBy(
        F.col('prediction_cutoff_timestamp').asc_nulls_last(),
        F.col('fl_date').asc_nulls_last(),
        F.col('crs_dep_time').asc_nulls_last(),
        F.col(tail_num_col).asc_nulls_last()
    )
    df = df.withColumn('_row_id', F.row_number().over(row_id_window))
    
    # Create aliases for self-join
    current = df.alias('current')
    previous = df.alias('previous')
    
    # Helper function to compute rolling average via self-join
    def compute_rolling_avg_via_join(partition_col, window_seconds, feature_name):
        """Compute rolling average using self-join for accurate data leakage prevention."""
        # Join condition: same partition (tail_num or origin) and time constraints
        join_condition = (
            (F.col('current.' + partition_col) == F.col('previous.' + partition_col)) &
            # Data leakage prevention: previous flight must have actually departed
            # before current flight's prediction cutoff (2 hours before scheduled departure)
            (F.col('previous.actual_dep_timestamp').isNotNull()) &
            (F.col('current.prediction_cutoff_timestamp').isNotNull()) &
            (F.col('previous.actual_dep_timestamp') <= F.col('current.prediction_cutoff_timestamp')) &
            # Time window: within the specified window size
            (F.unix_timestamp(F.col('current.prediction_cutoff_timestamp')) - 
             F.unix_timestamp(F.col('previous.actual_dep_timestamp')) <= window_seconds) &
            (F.unix_timestamp(F.col('current.prediction_cutoff_timestamp')) - 
             F.unix_timestamp(F.col('previous.actual_dep_timestamp')) >= 0) &
            # Only include previous flights (not the current flight itself)
            (F.col('current._row_id') != F.col('previous._row_id')) &
            # Only include flights with valid delay data
            (F.col('previous.' + dep_delay_col).isNotNull())
        )
        
        # Aggregate: compute average delay for matching previous flights
        rolling_avg = previous.join(
            current,
            join_condition,
            'right'
        ).groupBy(
            F.col('current._row_id').alias('_row_id')
        ).agg(
            F.avg(F.col('previous.' + dep_delay_col)).alias(feature_name)
        )
        
        return rolling_avg
    
    # Compute all rolling averages in separate aggregations, then join them all at once
    print("  Computing rolling averages per tail number...")
    tail_24h = compute_rolling_avg_via_join(tail_num_col, 86400, 'tail_num_rolling_avg_delay_24h')
    tail_7d = compute_rolling_avg_via_join(tail_num_col, 604800, 'tail_num_rolling_avg_delay_7d')
    tail_30d = compute_rolling_avg_via_join(tail_num_col, 2592000, 'tail_num_rolling_avg_delay_30d')
    
    print("  Computing rolling averages per origin airport...")
    origin_24h = compute_rolling_avg_via_join('origin', 86400, 'origin_rolling_avg_delay_24h')
    origin_7d = compute_rolling_avg_via_join('origin', 604800, 'origin_rolling_avg_delay_7d')
    origin_30d = compute_rolling_avg_via_join('origin', 2592000, 'origin_rolling_avg_delay_30d')
    
    # The aggregated results already have _row_id aliased correctly from groupBy
    # Join all rolling averages back to the main dataframe using qualified column names
    # Use explicit aliases and drop the duplicate _row_id from the right side after each join
    
    # Join each rolling average feature one at a time
    df = df.alias('df_main').join(
        tail_24h.alias('tail_24h'),
        F.col('df_main._row_id') == F.col('tail_24h._row_id'),
        'left'
    ).drop(F.col('tail_24h._row_id'))
    
    df = df.alias('df_main').join(
        tail_7d.alias('tail_7d'),
        F.col('df_main._row_id') == F.col('tail_7d._row_id'),
        'left'
    ).drop(F.col('tail_7d._row_id'))
    
    df = df.alias('df_main').join(
        tail_30d.alias('tail_30d'),
        F.col('df_main._row_id') == F.col('tail_30d._row_id'),
        'left'
    ).drop(F.col('tail_30d._row_id'))
    
    df = df.alias('df_main').join(
        origin_24h.alias('origin_24h'),
        F.col('df_main._row_id') == F.col('origin_24h._row_id'),
        'left'
    ).drop(F.col('origin_24h._row_id'))
    
    df = df.alias('df_main').join(
        origin_7d.alias('origin_7d'),
        F.col('df_main._row_id') == F.col('origin_7d._row_id'),
        'left'
    ).drop(F.col('origin_7d._row_id'))
    
    df = df.alias('df_main').join(
        origin_30d.alias('origin_30d'),
        F.col('df_main._row_id') == F.col('origin_30d._row_id'),
        'left'
    ).drop(F.col('origin_30d._row_id'))
    
    # Clean up temporary columns
    df = df.drop('actual_dep_timestamp', 'dep_timestamp_for_window', '_row_id')
    
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
    
    # Cumulative features: Look at flights BEFORE the immediate previous flight (exclude flight n-1)
    # These are much safer since they don't use the immediate previous flight's delay
    # However, they may still have leakage if any of the earlier flights haven't completed yet
    # For now, mark as having leakage only if the immediate previous flight has leakage
    # (This is conservative - in practice, flights before n-1 have usually completed)
    df = df.withColumn('has_leakage_lineage_cumulative_delay', col('has_leakage_prev_flight_dep_delay'))
    df = df.withColumn('has_leakage_lineage_avg_delay_previous_flights', col('has_leakage_prev_flight_dep_delay'))
    df = df.withColumn('has_leakage_lineage_max_delay_previous_flights', col('has_leakage_prev_flight_dep_delay'))
    
    # Rotation time inherits leakage from previous departure
    df = df.withColumn('has_leakage_lineage_rotation_time_minutes', col('has_leakage_prev_flight_actual_dep_time'))
    
    # Required time features: Expected values are safe (computed from historical data)
    # But impossible_on_time_flag depends on rotation_time, so it inherits leakage
    df = df.withColumn('has_leakage_required_time_prev_flight_minutes', F.lit(False))  # Expected values are safe
    df = df.withColumn('has_leakage_impossible_on_time_flag', col('has_leakage_lineage_rotation_time_minutes'))
    
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
                F.when(col('has_leakage_lineage_max_delay_previous_flights'), F.lit('lineage_max_delay_previous_flights')).otherwise(None),
                F.when(col('has_leakage_lineage_rotation_time_minutes'), F.lit('lineage_rotation_time_minutes')).otherwise(None),
                F.when(col('has_leakage_impossible_on_time_flag'), F.lit('impossible_on_time_flag')).otherwise(None)
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
    """
    Apply imputation for NULL values (first flight handling).
    
    KEY IMPUTATION LOGIC:
    - NULL rotation_time = no previous flight = plane already at airport
    - Large rotation_time buffer = plane is already at airport, plenty of time = likely to leave on time
    - Impute to 1440 minutes (24 hours) to indicate "essentially unlimited buffer" for first flights
    """
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
    
    # Scheduled flight time for previous flight: impute to 0 for first flights or jumps
    # This is computed as prev_flight_crs_arr_time - prev_flight_crs_dep_time (in _compute_cumulative_features)
    # If rotation_time >> scheduled_flight_time, then we have more buffer and are more likely to depart on time
    df = df.withColumn(
        'prev_flight_scheduled_flight_time_minutes',
        F.when(
            (col('lineage_rank') == 1) | col('lineage_is_jump'),
            F.lit(0.0)  # First flight or jump: no previous scheduled flight time
        ).otherwise(
            F.coalesce(col('prev_flight_scheduled_flight_time_minutes'), F.lit(0.0))
        )
    )
    
    # Keep prev_flight_crs_elapsed_time for backward compatibility (it's used in fallback logic)
    # But prefer the explicitly computed prev_flight_scheduled_flight_time_minutes
    df = df.withColumn(
        'prev_flight_crs_elapsed_time',
        F.when(
            (col('lineage_rank') == 1) | col('lineage_is_jump'),
            F.lit(0.0)  # First flight or jump: no previous scheduled flight time
        ).otherwise(
            F.coalesce(
                col('prev_flight_scheduled_flight_time_minutes'),  # Use explicitly computed value if available
                col('prev_flight_crs_elapsed_time'),  # Otherwise fall back to raw crs_elapsed_time
                F.lit(0.0)
            )
        )
    )
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
    
    # Impute turnover time features (if previous flight unavailable, plane is already at airport)
    # NULL turnover time = no previous flight = plane already at airport = large buffer = likely to leave on time
    # Impute to a very large value (1440 minutes = 24 hours) to indicate "essentially unlimited buffer"
    LARGE_TURNOVER_TIME_MINUTES = 1440.0  # 24 hours - indicates plane already at airport, plenty of buffer
    df = df.withColumn('scheduled_lineage_turnover_time_minutes', F.coalesce(col('scheduled_lineage_turnover_time_minutes'), F.lit(LARGE_TURNOVER_TIME_MINUTES)))
    df = df.withColumn('lineage_actual_turnover_time_minutes', F.coalesce(col('lineage_actual_turnover_time_minutes'), F.lit(LARGE_TURNOVER_TIME_MINUTES)))
    df = df.withColumn('lineage_actual_taxi_time_minutes', F.coalesce(col('lineage_actual_taxi_time_minutes'), col('lineage_actual_turnover_time_minutes')))
    df = df.withColumn('lineage_actual_turn_time_minutes', F.coalesce(col('lineage_actual_turn_time_minutes'), col('lineage_actual_turnover_time_minutes')))
    # Impute NULLs for cumulative features (when no flights exist before n-1)
    # Mean delay (9.37 minutes) was identified from EDA as the average departure delay across all flights
    df = df.withColumn('lineage_cumulative_delay', F.coalesce(col('lineage_cumulative_delay'), F.lit(0.0)))
    df = df.withColumn('lineage_avg_delay_previous_flights', F.coalesce(col('lineage_avg_delay_previous_flights'), F.lit(9.37)))
    df = df.withColumn('lineage_max_delay_previous_flights', F.coalesce(col('lineage_max_delay_previous_flights'), F.lit(0.0)))
    df = df.withColumn('lineage_num_previous_flights', F.coalesce(col('lineage_num_previous_flights'), F.lit(0)))
    df = df.withColumn('lineage_expected_flight_time_minutes', F.coalesce(col('lineage_expected_flight_time_minutes'), col('crs_elapsed_time')))
    
    # Impute rotation time features (if previous flight unavailable, plane is already at airport)
    # NULL rotation time = no previous flight = plane already at airport = large buffer = likely to leave on time
    # Impute to a very large value (1440 minutes = 24 hours) to indicate "essentially unlimited buffer"
    LARGE_ROTATION_TIME_MINUTES = 1440.0  # 24 hours - indicates plane already at airport, plenty of buffer
    df = df.withColumn('lineage_rotation_time_minutes', 
                       F.coalesce(col('lineage_rotation_time_minutes'), F.lit(LARGE_ROTATION_TIME_MINUTES)))
    df = df.withColumn('scheduled_lineage_rotation_time_minutes',
                       F.coalesce(col('scheduled_lineage_rotation_time_minutes'), F.lit(LARGE_ROTATION_TIME_MINUTES)))
    df = df.withColumn('safe_lineage_rotation_time_minutes',
                       F.coalesce(col('safe_lineage_rotation_time_minutes'), 
                                  col('lineage_rotation_time_minutes'), F.lit(LARGE_ROTATION_TIME_MINUTES)))
    
    # Impute required time features
    df = df.withColumn('required_time_prev_flight_minutes',
                       F.coalesce(col('required_time_prev_flight_minutes'), col('crs_elapsed_time'), F.lit(120.0)))
    df = df.withColumn('safe_required_time_prev_flight_minutes',
                       F.coalesce(col('safe_required_time_prev_flight_minutes'), col('required_time_prev_flight_minutes')))
    df = df.withColumn('impossible_on_time_flag',
                       F.coalesce(col('impossible_on_time_flag'), F.lit(False)))
    df = df.withColumn('safe_impossible_on_time_flag',
                       F.coalesce(col('safe_impossible_on_time_flag'), col('impossible_on_time_flag')))
    
    # Compute safe_time_since_prev_arrival
    # This measures the time between previous flight's arrival and the prediction cutoff
    # Logic:
    # (1) If actual prev arrival is before cutoff: use (cutoff - actual_arrival) in minutes
    #     This is the actual time that has passed since arrival
    # (2) If actual arrival is after cutoff (hasn't arrived yet):
    #     - If scheduled arrival is before cutoff: use 0
    #       Reason: Flight is delayed, earliest it can arrive is right now (at cutoff)
    #       So time since arrival = 0 (it arrives right now, no time has passed yet)
    #     - If scheduled arrival is after cutoff: use (cutoff - scheduled_arrival) in minutes (negative)
    #       This is time until scheduled arrival (negative value)
    df = df.withColumn(
        'safe_time_since_prev_arrival',
        F.when(
            # Case 1: Previous flight already arrived (actual_arr <= cutoff)
            (col('prev_flight_actual_arr_time').isNotNull()) & 
            (col('prev_flight_fl_date').isNotNull()) &
            (~col('has_leakage_prev_flight_actual_arr_time')),  # No leakage = already arrived
            # Use (cutoff - actual_arrival) in minutes
            (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
             F.unix_timestamp(
                 F.to_timestamp(
                     F.concat(col('prev_flight_fl_date'), 
                             F.lpad(col('prev_flight_actual_arr_time').cast('string'), 4, '0')),
                     'yyyy-MM-ddHHmm'
                 )
             )) / 60.0
        ).otherwise(
            # Case 2: Previous flight hasn't arrived yet (actual_arr > cutoff or NULL)
            F.when(
                (col('prediction_cutoff_timestamp').isNotNull()) & 
                (col('prev_flight_sched_arr_timestamp').isNotNull()),
                F.when(
                    # Scheduled arrival is before cutoff: use 0 (earliest it can arrive is right now)
                    col('prev_flight_sched_arr_timestamp') < col('prediction_cutoff_timestamp'),
                    F.lit(0.0)
                ).otherwise(
                    # Scheduled arrival is after cutoff: use (cutoff - scheduled_arrival) in minutes (negative)
                    (F.unix_timestamp(col('prediction_cutoff_timestamp')) - 
                     F.unix_timestamp(col('prev_flight_sched_arr_timestamp'))) / 60.0
                )
            ).otherwise(
                # Fallback: if we can't compute, use 0
                F.lit(0.0)
            )
        )
    )
    
    # Impute safe delay features: for first flights (no previous flight), use 0 (no delay)
    df = df.withColumn('safe_prev_departure_delay',
                       F.coalesce(col('safe_prev_departure_delay'), F.lit(0.0)))
    df = df.withColumn('safe_prev_arrival_delay',
                       F.coalesce(col('safe_prev_arrival_delay'), F.lit(0.0)))
    df = df.withColumn('safe_time_since_prev_arrival',
                       F.coalesce(col('safe_time_since_prev_arrival'), F.lit(0.0)))
    
    df = df.drop('prev_flight_crs_dep_time_minutes')
    
    return df