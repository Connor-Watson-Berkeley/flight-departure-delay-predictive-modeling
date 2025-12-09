"""
conditional_expected_values_experiment.py

Experimentation script to generate conditional expected values for air time and turnover time.
This script generates lookup tables with conditional means that can be used for feature engineering.

Key Points:
- Conditional means are computed from TRAINING DATA ONLY (cross-validation handles this)
- No data leakage concerns for conditional means themselves
- Data leakage only matters for raw features used in cross-validation evaluation
- Outputs parquet files with conditional expected values for use in ConditionalExpectedValuesEstimator

Usage:
    from conditional_expected_values_experiment import generate_conditional_expected_values
    
    # Load training data (already filtered to training fold)
    train_df = spark.read.parquet("path/to/train_data.parquet")
    
    # Generate conditional expected values
    conditional_means = generate_conditional_expected_values(
        train_df,
        output_path="dbfs:/path/to/conditional_expected_values/",
        use_prophet=True
    )
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, to_timestamp, when, avg, count as spark_count, min as spark_min, max as spark_max
import pandas as pd
from datetime import datetime
import os
import time
from contextlib import contextmanager

# Suppress verbose cmdstanpy output BEFORE importing Prophet
import logging
# Suppress cmdstanpy INFO messages
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# Also suppress prophet internal logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('pystan').setLevel(logging.WARNING)

from prophet import Prophet


@contextmanager
def timer(description, verbose=True):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    if verbose and description:
        print(f"⏱️  {description}: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


def _fit_prophet_model(data, y_col='y', min_days_required=14, changepoint_prior_scale=0.05):
    """
    Fit a Prophet model on time series data.
    
    Parameters:
    - data: pandas DataFrame with 'ds' (date) and y_col columns
    - y_col: name of the target column
    - min_days_required: minimum days required to fit Prophet model
    
    Returns:
    - Forecast DataFrame (or None if insufficient data)
    """
    if len(data) < min_days_required:
        return None
    
    # Prepare data for Prophet
    prophet_data = data[['ds', y_col]].copy()
    prophet_data = prophet_data.rename(columns={y_col: 'y'})
    prophet_data = prophet_data.dropna()
    
    if len(prophet_data) < min_days_required:
        return None
    
    # Determine seasonality based on data availability
    has_enough_for_yearly = len(prophet_data) >= 365
    has_enough_for_weekly = len(prophet_data) >= 14
    
    # Fit Prophet model
    prophet_model = Prophet(
        yearly_seasonality=has_enough_for_yearly,
        weekly_seasonality=has_enough_for_weekly,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.95,
        changepoint_prior_scale=changepoint_prior_scale
    )
    prophet_model.fit(prophet_data)
    
    # Generate forecast
    forecast = prophet_model.predict(prophet_data[['ds']])
    
    return forecast


def generate_conditional_expected_values(
    df,
    output_path=None,
    date_col="FL_DATE",
    carrier_col="op_carrier",
    origin_col="origin",
    dest_col="dest",
    air_time_col="air_time",
    min_observations=10,
    min_days_required=14,
    changepoint_prior_scale=0.05,
    use_prophet=True,
    verbose=True,
    save_results=True
):
    """
    Generate conditional expected values for air time and turnover time.
    
    This function is timed - execution time is reported in the summary.
    
    Computes multiple conditional means:
    - Air Time: route-based, temporal (Prophet), aircraft-based
    - Turnover Time: carrier-airport, temporal (Prophet), aircraft-based
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Training data (must be filtered to training fold only)
    output_path : str, optional
        Base path where conditional expected value tables will be saved.
        If None and save_results=True, results will not be saved.
    save_results : bool
        Whether to save results to parquet files (default: True).
        If False, results are only returned as DataFrames.
    date_col : str
        Name of date column (default: "FL_DATE")
    carrier_col : str
        Name of carrier column (default: "op_carrier")
    origin_col : str
        Name of origin airport column (default: "origin")
    dest_col : str
        Name of destination airport column (default: "dest")
    air_time_col : str
        Name of air time column (default: "air_time")
    min_observations : int
        Minimum observations required for non-temporal conditional means (default: 10)
    min_days_required : int
        Minimum days required for Prophet models (default: 14)
    changepoint_prior_scale : float
        Prophet changepoint prior scale (default: 0.05)
    use_prophet : bool
        Whether to generate temporal conditional means using Prophet (default: True)
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    dict : Dictionary with conditional expected value DataFrames (and optionally paths if saved)
        Keys: 'expected_air_time_route', 'expected_air_time_route_temporal', etc.
        Values: Spark DataFrames (and file paths if save_results=True)
    """
    start_time = datetime.now()
    start_time_seconds = time.time()
    timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if verbose:
        print("=" * 80)
        print("CONDITIONAL EXPECTED VALUES GENERATION")
        print("=" * 80)
        print(f"[{timestamp}] Starting conditional expected value generation...")
        if output_path:
            print(f"Output path: {output_path}")
        else:
            print(f"Output path: None (results will not be saved to disk)")
        print(f"Training data size: {df.count():,} rows")
        print(f"Save results: {save_results}")
        print()
    
    spark = SparkSession.builder.getOrCreate()
    
    # Find tail_num column (flexible like flight_lineage_features.py)
    tail_num_col = None
    tail_num_candidates = ['tail_num', 'TAIL_NUM', 'tail_number', 'TAIL_NUMBER', 'op_unique_carrier_tail_num']
    for candidate in tail_num_candidates:
        if candidate in df.columns:
            tail_num_col = candidate
            break
    if tail_num_col is None:
        # Try pattern matching
        tail_cols = [c for c in df.columns if 'tail' in c.lower()]
        if tail_cols:
            tail_num_col = tail_cols[0]
    
    if verbose:
        if tail_num_col:
            print(f"✓ Found tail_num column: {tail_num_col}")
        else:
            print("⚠ Warning: No tail_num column found. Aircraft-based conditional means will be skipped.")
    
    # Prepare date column
    with timer("Data preparation", verbose=verbose):
        initial_count = df.count()
        df_prep = df.withColumn(
            "date",
            to_timestamp(col(date_col), "yyyy-MM-dd").cast("date")
        ).withColumn(
            "date_str",
            F.date_format(col("date"), "yyyy-MM-dd")
        ).filter(
            col("date").isNotNull()
        )
        final_count = df_prep.count()
    
    if verbose:
        print(f"✓ Data preparation complete: {final_count:,} rows after filtering invalid dates (dropped {initial_count - final_count:,} rows)")
        print(f"✓ Column detection:")
        print(f"    - air_time_col: {air_time_col} {'✓' if air_time_col in df_prep.columns else '✗ NOT FOUND'}")
        print(f"    - carrier_col: {carrier_col} {'✓' if carrier_col in df_prep.columns else '✗ NOT FOUND'}")
        print(f"    - origin_col: {origin_col} {'✓' if origin_col in df_prep.columns else '✗ NOT FOUND'}")
        print(f"    - dest_col: {dest_col} {'✓' if dest_col in df_prep.columns else '✗ NOT FOUND'}")
        print()
    
    saved_paths = {}
    results = {}  # Store DataFrames for return
    
    # ============================================================================
    # AIR TIME CONDITIONAL EXPECTED VALUES
    # ============================================================================
    
    if verbose:
        print("\n" + "-" * 80)
        print("AIR TIME CONDITIONAL EXPECTED VALUES")
        print("-" * 80)
    
    # 1. Non-temporal: Route-based expected air time
    if air_time_col in df_prep.columns:
        if verbose:
            print("\n1. Computing route-based expected air time...")
        
        with timer("  Route-based air time computation", verbose=verbose):
            route_air = (
            df_prep
            .filter(col(air_time_col).isNotNull())
            .filter(col(origin_col).isNotNull())
            .filter(col(dest_col).isNotNull())
            .groupBy(origin_col, dest_col)
            .agg(
                avg(air_time_col).alias("expected_air_time_route_minutes"),
                spark_count("*").alias("count"),
                spark_min(air_time_col).alias("min_air_time"),
                spark_max(air_time_col).alias("max_air_time"),
                F.stddev(air_time_col).alias("stddev_air_time")
            )
            .filter(col("count") >= min_observations)
            .orderBy(origin_col, dest_col)
            )
            results['expected_air_time_route'] = route_air
        
        route_count = route_air.count()
        if verbose:
            print(f"    ✓ Generated expected air time for {route_count:,} routes")
            if route_count > 0:
                # Show sample statistics
                sample_stats = route_air.select(
                    F.avg("expected_air_time_route_minutes").alias("avg_expected"),
                    spark_min("expected_air_time_route_minutes").alias("min_expected"),
                    spark_max("expected_air_time_route_minutes").alias("max_expected"),
                    F.avg("count").alias("avg_observations_per_route")
                ).first()
                if sample_stats:
                    print(f"      Statistics: avg={sample_stats['avg_expected']:.1f} min, "
                          f"min={sample_stats['min_expected']:.1f} min, "
                          f"max={sample_stats['max_expected']:.1f} min, "
                          f"avg observations/route={sample_stats['avg_observations_per_route']:.1f}")
        
        if save_results and output_path:
            route_air_path = os.path.join(output_path, "expected_air_time_route.parquet")
            route_air.write.mode("overwrite").parquet(route_air_path)
            saved_paths['expected_air_time_route'] = route_air_path
            if verbose:
                print(f"    ✓ Saved to: {route_air_path}")
        
        # 2. Temporal: Route-based expected air time with Prophet
        if use_prophet:
            if verbose:
                print("\n2. Computing temporal route-based expected air time (Prophet)...")
            
            with timer("  Temporal route-based air time (Prophet)", verbose=verbose):
                route_temporal_spark = (
                df_prep
                .filter(col(air_time_col).isNotNull())
                .filter(col(origin_col).isNotNull())
                .filter(col(dest_col).isNotNull())
                .groupBy(origin_col, dest_col, "date")
                .agg(avg(air_time_col).alias("avg_air_time"))
                .orderBy(origin_col, dest_col, "date")
                )
                
                route_temporal = route_temporal_spark.toPandas()
            
            if len(route_temporal) > 0:
                route_temporal['ds'] = pd.to_datetime(route_temporal['date'])
                route_temporal_list = []
                
                # Get unique routes
                unique_routes = route_temporal[[origin_col, dest_col]].drop_duplicates()
                
                if verbose:
                    print(f"    Processing {len(unique_routes)} routes with temporal data...")
                
                for idx, (_, route_row) in enumerate(unique_routes.iterrows(), 1):
                    origin = route_row[origin_col]
                    dest = route_row[dest_col]
                    
                    route_data = route_temporal[
                        (route_temporal[origin_col] == origin) & 
                        (route_temporal[dest_col] == dest)
                    ].sort_values('ds')
                    
                    if len(route_data) >= min_days_required:
                        forecast = _fit_prophet_model(
                            route_data, 
                            'avg_air_time',
                            min_days_required=min_days_required,
                            changepoint_prior_scale=changepoint_prior_scale
                        )
                        
                        if forecast is not None:
                            forecast[origin_col] = origin
                            forecast[dest_col] = dest
                            forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')
                            route_temporal_list.append(forecast[['origin', 'dest', 'date_str', 'yhat']])
                    
                    if verbose and idx % 100 == 0:
                        print(f"    Processed {idx}/{len(unique_routes)} routes...")
                
                if route_temporal_list:
                    expected_air_time_route_temporal = pd.concat(route_temporal_list, ignore_index=True)
                    expected_air_time_route_temporal = expected_air_time_route_temporal.rename(
                        columns={'yhat': 'expected_air_time_route_temporal_minutes'}
                    )
                    
                    # Convert back to Spark DataFrame
                    route_temporal_spark_out = spark.createDataFrame(expected_air_time_route_temporal)
                    results['expected_air_time_route_temporal'] = route_temporal_spark_out
                    
                    unique_routes_count = expected_air_time_route_temporal[[origin_col, dest_col]].drop_duplicates().shape[0]
                    if verbose:
                        print(f"    ✓ Generated temporal expected air time for {unique_routes_count} routes")
                    
                    if save_results and output_path:
                        route_temporal_path = os.path.join(output_path, "expected_air_time_route_temporal.parquet")
                        route_temporal_spark_out.write.mode("overwrite").parquet(route_temporal_path)
                        saved_paths['expected_air_time_route_temporal'] = route_temporal_path
                        if verbose:
                            print(f"    ✓ Saved to: {route_temporal_path}")
        
        # 3. Aircraft-based expected air time
        if tail_num_col and tail_num_col in df_prep.columns:
            if verbose:
                print("\n3. Computing aircraft-based expected air time...")
            
            with timer("  Aircraft-based air time computation", verbose=verbose):
                aircraft_air = (
                df_prep
                .filter(col(air_time_col).isNotNull())
                .filter(col(tail_num_col).isNotNull())
                .groupBy(tail_num_col)
                .agg(
                    avg(air_time_col).alias("expected_air_time_aircraft_minutes"),
                    spark_count("*").alias("count"),
                    spark_min(air_time_col).alias("min_air_time"),
                    spark_max(air_time_col).alias("max_air_time")
                )
                .filter(col("count") >= min_observations)
                .orderBy(tail_num_col)
                )
                results['expected_air_time_aircraft'] = aircraft_air
            
            aircraft_count = aircraft_air.count()
            if verbose:
                print(f"    ✓ Generated expected air time for {aircraft_count:,} aircraft")
            
            if save_results and output_path:
                aircraft_air_path = os.path.join(output_path, "expected_air_time_aircraft.parquet")
                aircraft_air.write.mode("overwrite").parquet(aircraft_air_path)
                saved_paths['expected_air_time_aircraft'] = aircraft_air_path
                if verbose:
                    print(f"    ✓ Saved to: {aircraft_air_path}")
    
    # ============================================================================
    # TURNOVER TIME CONDITIONAL EXPECTED VALUES
    # ============================================================================
    
    if verbose:
        print("\n" + "-" * 80)
        print("TURNOVER TIME CONDITIONAL EXPECTED VALUES")
        print("-" * 80)
    
    # Check if lineage features are available
    # For conditional means, prefer actual turnover time (what actually happened), fallback to scheduled
    turnover_time_col = None
    if 'lineage_actual_turnover_time_minutes' in df_prep.columns:
        turnover_time_col = 'lineage_actual_turnover_time_minutes'
        if verbose:
            print(f"✓ Using actual turnover time column: {turnover_time_col}")
    elif 'scheduled_lineage_turnover_time_minutes' in df_prep.columns:
        turnover_time_col = 'scheduled_lineage_turnover_time_minutes'
        if verbose:
            print(f"✓ Using scheduled turnover time column: {turnover_time_col} (actual not available)")
    
    if turnover_time_col is None:
        if verbose:
            print("\n⚠ Warning: Neither lineage_actual_turnover_time_minutes nor scheduled_lineage_turnover_time_minutes found in data.")
            print("  Turnover time conditional means cannot be computed without lineage features.")
            print("  Make sure flight_lineage_features has been applied to the data.")
    else:
        # 1. Non-temporal: Carrier-Airport expected turnover time
        if verbose:
            print("\n1. Computing carrier-airport expected turnover time...")
        
        with timer("  Carrier-airport turnover time computation", verbose=verbose):
            turnover_spark = (
            df_prep
            .filter(col(turnover_time_col).isNotNull())
            .filter(col(carrier_col).isNotNull())
            .filter(col(origin_col).isNotNull())
            .groupBy(carrier_col, origin_col)
            .agg(
                avg(turnover_time_col).alias("expected_turnover_time_carrier_airport_minutes"),
                spark_count("*").alias("count"),
                spark_min(turnover_time_col).alias("min_turnover_time"),
                spark_max(turnover_time_col).alias("max_turnover_time"),
                F.stddev(turnover_time_col).alias("stddev_turnover_time")
            )
            .filter(col("count") >= min_observations)
            .orderBy(carrier_col, origin_col)
            )
            results['expected_turnover_time_carrier_airport'] = turnover_spark
        
        turnover_count = turnover_spark.count()
        if verbose:
            # Show sample statistics
            sample_stats = turnover_spark.select(
                F.avg("expected_turnover_time_carrier_airport_minutes").alias("avg_expected"),
                spark_min("expected_turnover_time_carrier_airport_minutes").alias("min_expected"),
                spark_max("expected_turnover_time_carrier_airport_minutes").alias("max_expected"),
                F.avg("count").alias("avg_observations_per_pair")
            ).first()
            print(f"    ✓ Generated expected turnover time for {turnover_count:,} carrier-airport pairs")
            if sample_stats:
                print(f"      Statistics: avg={sample_stats['avg_expected']:.1f} min, "
                      f"min={sample_stats['min_expected']:.1f} min, "
                      f"max={sample_stats['max_expected']:.1f} min, "
                      f"avg observations/pair={sample_stats['avg_observations_per_pair']:.1f}")
        
        if save_results and output_path:
            turnover_path = os.path.join(output_path, "expected_turnover_time_carrier_airport.parquet")
            turnover_spark.write.mode("overwrite").parquet(turnover_path)
            saved_paths['expected_turnover_time_carrier_airport'] = turnover_path
            if verbose:
                print(f"    ✓ Saved to: {turnover_path}")
        
        # 2. Temporal: Carrier-Airport expected turnover time with Prophet
        if use_prophet:
            if verbose:
                print("\n2. Computing temporal carrier-airport expected turnover time (Prophet)...")
            
            with timer("  Temporal carrier-airport turnover time (Prophet)", verbose=verbose):
                turnover_temporal_spark = (
                df_prep
                .filter(col(turnover_time_col).isNotNull())
                .filter(col(carrier_col).isNotNull())
                .filter(col(origin_col).isNotNull())
                .groupBy(carrier_col, origin_col, "date")
                .agg(avg(turnover_time_col).alias("avg_turnover_time"))
                .orderBy(carrier_col, origin_col, "date")
                )
                
                turnover_temporal = turnover_temporal_spark.toPandas()
            
            if len(turnover_temporal) > 0:
                turnover_temporal['ds'] = pd.to_datetime(turnover_temporal['date'])
                turnover_temporal_list = []
                
                # Get unique carrier-airport pairs
                unique_pairs = turnover_temporal[[carrier_col, origin_col]].drop_duplicates()
                
                if verbose:
                    print(f"    Processing {len(unique_pairs)} carrier-airport pairs with temporal data...")
                
                for idx, (_, pair_row) in enumerate(unique_pairs.iterrows(), 1):
                    carrier = pair_row[carrier_col]
                    airport = pair_row[origin_col]
                    
                    pair_data = turnover_temporal[
                        (turnover_temporal[carrier_col] == carrier) & 
                        (turnover_temporal[origin_col] == airport)
                    ].sort_values('ds')
                    
                    if len(pair_data) >= min_days_required:
                        forecast = _fit_prophet_model(
                            pair_data,
                            'avg_turnover_time',
                            min_days_required=min_days_required,
                            changepoint_prior_scale=changepoint_prior_scale
                        )
                        
                        if forecast is not None:
                            forecast[carrier_col] = carrier
                            forecast[origin_col] = airport
                            forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')
                            turnover_temporal_list.append(forecast[[carrier_col, origin_col, 'date_str', 'yhat']])
                    
                    if verbose and idx % 50 == 0:
                        print(f"    Processed {idx}/{len(unique_pairs)} carrier-airport pairs...")
                
                if turnover_temporal_list:
                    expected_turnover_time_temporal = pd.concat(turnover_temporal_list, ignore_index=True)
                    expected_turnover_time_temporal = expected_turnover_time_temporal.rename(
                        columns={'yhat': 'expected_turnover_time_temporal_minutes'}
                    )
                    
                    # Convert back to Spark DataFrame
                    turnover_temporal_spark_out = spark.createDataFrame(expected_turnover_time_temporal)
                    results['expected_turnover_time_temporal'] = turnover_temporal_spark_out
                    
                    unique_pairs_count = expected_turnover_time_temporal[[carrier_col, origin_col]].drop_duplicates().shape[0]
                    if verbose:
                        print(f"    ✓ Generated temporal expected turnover time for {unique_pairs_count} carrier-airport pairs")
                    
                    if save_results and output_path:
                        turnover_temporal_path = os.path.join(output_path, "expected_turnover_time_temporal.parquet")
                        turnover_temporal_spark_out.write.mode("overwrite").parquet(turnover_temporal_path)
                        saved_paths['expected_turnover_time_temporal'] = turnover_temporal_path
                        if verbose:
                            print(f"    ✓ Saved to: {turnover_temporal_path}")
        
        # 3. Aircraft-based expected turnover time
        if tail_num_col and tail_num_col in df_prep.columns:
            if verbose:
                print("\n3. Computing aircraft-based expected turnover time...")
            
            with timer("  Aircraft-based turnover time computation", verbose=verbose):
                aircraft_turnover = (
                df_prep
                .filter(col(turnover_time_col).isNotNull())
                .filter(col(tail_num_col).isNotNull())
                .groupBy(tail_num_col)
                .agg(
                    avg(turnover_time_col).alias("expected_turnover_time_aircraft_minutes"),
                    spark_count("*").alias("count"),
                    spark_min(turnover_time_col).alias("min_turnover_time"),
                    spark_max(turnover_time_col).alias("max_turnover_time")
                )
                .filter(col("count") >= min_observations)
                .orderBy(tail_num_col)
                )
                results['expected_turnover_time_aircraft'] = aircraft_turnover
            
            aircraft_turnover_count = aircraft_turnover.count()
            if verbose:
                print(f"    ✓ Generated expected turnover time for {aircraft_turnover_count:,} aircraft")
            
            if save_results and output_path:
                aircraft_turnover_path = os.path.join(output_path, "expected_turnover_time_aircraft.parquet")
                aircraft_turnover.write.mode("overwrite").parquet(aircraft_turnover_path)
                saved_paths['expected_turnover_time_aircraft'] = aircraft_turnover_path
                if verbose:
                    print(f"    ✓ Saved to: {aircraft_turnover_path}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    
    end_time = datetime.now()
    end_time_seconds = time.time()
    elapsed_seconds = end_time_seconds - start_time_seconds
    duration = end_time - start_time
    timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    if verbose:
        print("\n" + "=" * 80)
        print("CONDITIONAL EXPECTED VALUES GENERATION COMPLETE")
        print("=" * 80)
        print(f"[{timestamp}] ✓ Generation complete!")
        print(f"⏱️  Total execution time: {elapsed_seconds:.2f} seconds ({elapsed_seconds/60:.2f} minutes)")
        print(f"\nGenerated conditional expected value tables:")
        for name in sorted(results.keys()):
            try:
                row_count = results[name].count() if hasattr(results[name], 'count') else 'N/A'
                print(f"  ✓ {name}: {row_count:,} rows")
            except Exception as e:
                print(f"  ⚠ {name}: Error counting rows ({str(e)})")
        if saved_paths:
            print(f"\nSaved files:")
            for name, path in saved_paths.items():
                print(f"  - {name}: {path}")
        print("=" * 80)
    
    # Return DataFrames in results dict, with paths if saved
    results.update(saved_paths)
    return results


def load_conditional_expected_values(spark, base_path):
    """
    Load conditional expected value tables from parquet files.
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session
    base_path : str
        Base path where conditional expected value tables are stored
    
    Returns:
    --------
    dict : Dictionary with loaded DataFrames (empty dict if files don't exist)
    """
    loaded = {}
    
    # Helper function to safely check and load parquet files
    def try_load_parquet(path, key):
        try:
            df = spark.read.parquet(path)
            loaded[key] = df
            return True
        except Exception as e:
            # File doesn't exist or can't be read
            return False
    
    # Air time tables
    route_path = os.path.join(base_path, "expected_air_time_route.parquet")
    try_load_parquet(route_path, 'expected_air_time_route')
    
    route_temporal_path = os.path.join(base_path, "expected_air_time_route_temporal.parquet")
    try_load_parquet(route_temporal_path, 'expected_air_time_route_temporal')
    
    aircraft_air_path = os.path.join(base_path, "expected_air_time_aircraft.parquet")
    try_load_parquet(aircraft_air_path, 'expected_air_time_aircraft')
    
    # Turnover time tables
    turnover_path = os.path.join(base_path, "expected_turnover_time_carrier_airport.parquet")
    try_load_parquet(turnover_path, 'expected_turnover_time_carrier_airport')
    
    turnover_temporal_path = os.path.join(base_path, "expected_turnover_time_temporal.parquet")
    try_load_parquet(turnover_temporal_path, 'expected_turnover_time_temporal')
    
    aircraft_turnover_path = os.path.join(base_path, "expected_turnover_time_aircraft.parquet")
    try_load_parquet(aircraft_turnover_path, 'expected_turnover_time_aircraft')
    
    return loaded


def run_experiments(train_df, sample_size=None, use_prophet=True, verbose=True):
    """
    Run a series of experiments to validate conditional expected value generation.
    
    Parameters:
    -----------
    train_df : pyspark.sql.DataFrame
        Training data with flight lineage features
    sample_size : int, optional
        If provided, sample this many rows for faster experimentation
    use_prophet : bool
        Whether to run Prophet-based temporal experiments
    verbose : bool
        Whether to print detailed output
    
    Returns:
    --------
    dict : Results dictionary with all generated conditional expected values
    """
    print("\n" + "=" * 80)
    print("RUNNING CONDITIONAL EXPECTED VALUES EXPERIMENTS")
    print("=" * 80)
    
    # Optionally sample data for faster experiments
    if sample_size:
        print(f"\n📊 Sampling {sample_size:,} rows for faster experimentation...")
        train_df = train_df.sample(fraction=1.0).limit(sample_size)
        actual_count = train_df.count()
        print(f"✓ Using {actual_count:,} rows")
    
    # Run main experiment
    print("\n" + "=" * 80)
    print("EXPERIMENT: Generate All Conditional Expected Values")
    print("=" * 80)
    
    results = generate_conditional_expected_values(
        train_df,
        save_results=False,  # No file I/O for experiments
        use_prophet=use_prophet,
        verbose=verbose
    )
    
    # Validate results
    print("\n" + "=" * 80)
    print("VALIDATION: Checking Generated Results")
    print("=" * 80)
    
    validation_passed = True
    
    # Check that expected tables exist
    expected_tables = [
        'expected_air_time_route',
        'expected_turnover_time_carrier_airport'
    ]
    
    if use_prophet:
        expected_tables.extend([
            'expected_air_time_route_temporal',
            'expected_turnover_time_temporal'
        ])
    
    for table_name in expected_tables:
        if table_name in results:
            try:
                row_count = results[table_name].count()
                if row_count > 0:
                    print(f"✓ {table_name}: {row_count:,} rows")
                    
                    # Show sample
                    if verbose:
                        print(f"  Sample:")
                        results[table_name].show(5, truncate=False)
                else:
                    print(f"⚠ {table_name}: 0 rows (no data)")
                    validation_passed = False
            except Exception as e:
                print(f"✗ {table_name}: Error - {str(e)}")
                validation_passed = False
        else:
            print(f"✗ {table_name}: Missing from results")
            validation_passed = False
    
    print("\n" + "=" * 80)
    if validation_passed:
        print("✓ VALIDATION PASSED - All expected tables generated successfully!")
    else:
        print("⚠ VALIDATION WARNINGS - Some issues detected (see above)")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    """
    Automatically run experiments when this script is executed.
    
    This will:
    1. Load training data from the first fold using FlightDelayDataLoader
    2. Run conditional expected value experiments
    3. Display results and validation
    """
    print("=" * 80)
    print("CONDITIONAL EXPECTED VALUES EXPERIMENT")
    print("=" * 80)
    print("\n🚀 Starting automatic experiment execution...")
    
    try:
        # Get Spark session
        spark = SparkSession.builder.getOrCreate()
        print(f"✓ Spark session available (version: {spark.version})")
        
        # Load data using the same pattern as other notebooks
        print("\n" + "-" * 80)
        print("LOADING TRAINING DATA")
        print("-" * 80)
        
        # Import cv module (matching pattern from Flight Lineage Features Experiment notebook)
        import importlib.util
        
        cv_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/cv.py"
        print(f"Loading cv module from: {cv_path}")
        
        try:
            spec = importlib.util.spec_from_file_location("cv", cv_path)
            cv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cv)
            print("✓ Loaded cv module")
        except Exception as e:
            print(f"✗ Could not load cv module from {cv_path}")
            print(f"  Error: {str(e)}")
            print("  Please ensure cv.py is accessible in your Databricks workspace")
            raise
        
        # Load data (matching the pattern from your notebooks)
        print("\nLoading data for conditional expected values experiment...")
        
        data_loader = cv.FlightDelayDataLoader()
        data_loader.load()
        
        VERSION = "3M"  # Start with 3M for faster experiments
        folds = data_loader.get_version(VERSION)
        
        print(f"✓ Loaded data for version: {VERSION}")
        
        if not folds:
            print(f"✗ No folds found for version {VERSION}")
            print("  Available versions may be different. Please check your data.")
            raise ValueError(f"No folds available for version {VERSION}")
        
        print(f"✓ Loaded {len(folds)} folds for version {VERSION}")
        
        # Use the first training fold for experiments
        if len(folds) > 0:
            train_df, val_df = folds[0]
            train_count = train_df.count()
            val_count = val_df.count()
            print(f"\n✓ Using Fold 1:")
            print(f"  Training: {train_count:,} rows")
            print(f"  Validation: {val_count:,} rows")
            
            # Run experiments
            print("\n" + "=" * 80)
            print("RUNNING EXPERIMENTS")
            print("=" * 80)
            
            results = run_experiments(
                train_df,
                sample_size=None,  # Use all training data
                use_prophet=True,
                verbose=True
            )
            
            print("\n" + "=" * 80)
            print("✅ EXPERIMENTS COMPLETE!")
            print("=" * 80)
            print("\nResults are available in the 'results' dictionary.")
            print("You can access them like:")
            print("  results['expected_air_time_route'].show(10)")
            print("  results['expected_turnover_time_carrier_airport'].show(10)")
            print("\n" + "=" * 80)
            
        else:
            print("✗ No folds available")
            raise ValueError("No folds found")
            
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR RUNNING EXPERIMENTS")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Ensure cv.py is accessible")
        print("  2. Check that training folds exist in the expected location")
        print("  3. Verify you're running this in a Databricks notebook")
        print("  4. If needed, manually run:")
        print("     from conditional_expected_values_experiment import run_experiments")
        print("     results = run_experiments(train_df)")
        import traceback
        traceback.print_exc()

