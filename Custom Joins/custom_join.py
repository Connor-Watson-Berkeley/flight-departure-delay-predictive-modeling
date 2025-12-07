"""
Custom Join Pipeline

This script implements the custom join pipeline to combine flights, airports, stations, and weather data.

Pipeline Steps:
1. Load and deduplicate data
2. Join with airport codes (latitude/longitude)
3. Match airports to weather stations
4. Join with weather data
5. Add flight lineage features
6. Save final result

Usage:
    In Databricks: %run /path/to/custom_join.py
    Or: exec(open('custom_join.py').read())
"""

import requests
from pyspark.sql.functions import (
    col, regexp_replace, split, trim, to_timestamp, broadcast, expr,
    F, when, lit, coalesce, lag, row_number, sum as spark_sum, avg, max,
    array, array_remove, lpad, concat, floor
)
from pyspark.sql.window import Window

# Import flight lineage module
from flight_lineage import add_flight_lineage_features


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_VERSION = "3m"  # "3m", "6m", "1y", "" -> blank is full
SECTION = "4"
NUMBER = "2"
BUCKET_INTERVAL_MINUTES = 30


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanity_check(df, step_name, prev_count=None):
    """Print row count and key statistics for a pipeline step."""
    print(f"\n{'='*80}")
    print(f"SANITY CHECK: {step_name}")
    print('='*80)
    
    count = df.count()
    print(f"Row count: {count:,}")
    
    if prev_count is not None:
        diff = count - prev_count
        pct = (diff / prev_count * 100) if prev_count > 0 else 0
        print(f"Change: {diff:+,} ({pct:+.2f}%)")
    
    # Check key columns for NULLs
    key_cols = ['origin', 'dest', 'origin_station_id', 'origin_latitude', 'origin_longitude']
    for col_name in key_cols:
        if col_name in df.columns:
            nulls = df.filter(F.col(col_name).isNull()).count()
            if nulls > 0:
                pct = (nulls / count * 100) if count > 0 else 0
                print(f"  {col_name}: {nulls:,} NULLs ({pct:.2f}%)")
    
    print('='*80)
    return count


def setup_spark_config(spark):
    """Configure Spark settings for efficient execution."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)
    
    print("Spark version:", spark.version)
    print("AQE:", spark.conf.get("spark.sql.adaptive.enabled"))
    print("Skew join:", spark.conf.get("spark.sql.adaptive.skewJoin.enabled"))


def setup_paths(section, number, data_version):
    """Set up directory paths and create them if needed."""
    section_dir = f"dbfs:/mnt/mids-w261/student-groups/Group_{section}_{number}"
    
    raw_dir = f"{section_dir}/raw"
    processed_dir = f"{section_dir}/processed"
    checkpoints_dir = f"{section_dir}/checkpoints"
    intermediate_dir = f"{section_dir}/intermediate"
    
    flights_weather_joined_path = f"{processed_dir}/flights_weather_joined_{data_version}"
    
    # Create directories
    for path in [raw_dir, processed_dir, checkpoints_dir, intermediate_dir]:
        dbutils.fs.mkdirs(path)
    
    spark.sparkContext.setCheckpointDir(checkpoints_dir)
    print(f"Output path: {flights_weather_joined_path}")
    
    return flights_weather_joined_path, raw_dir


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_base_datasets(spark, data_version):
    """Load flights, stations, and weather datasets."""
    # Airline Data
    if data_version == "":
        df_flights = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
    else:
        df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_{data_version}/")
    
    # Stations data
    df_stations = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
    
    # Weather data
    if data_version == "":
        df_weather = spark.read.parquet("dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")
    else:
        df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_{data_version}/")
    
    return df_flights, df_stations, df_weather


def load_airport_codes(spark, raw_dir):
    """Download and load airport codes CSV."""
    url = "https://datahub.io/core/airport-codes/r/airport-codes.csv"
    local_path = "/tmp/airport-codes.csv"
    target_path = f"{raw_dir}/airport-codes.csv"
    
    # Download if not exists
    try:
        dbutils.fs.ls(target_path)
        print("Airport codes file exists, skipping download")
    except:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        dbutils.fs.cp("file://" + local_path, target_path)
    
    df_airport_codes = spark.read.option("header", "true").csv(f'/dbfs{raw_dir}/airport-codes.csv').cache()
    df_airport_codes.count()  # Materialize cache
    return df_airport_codes


def load_airport_timezones(spark):
    """Download and load airport timezones CSV."""
    url = "https://raw.githubusercontent.com/opentraveldata/opentraveldata/master/opentraveldata/optd_por_public.csv"
    local_path = "/dbfs/tmp/airport-timezones.csv"
    dbfs_path = "dbfs:/tmp/airport-timezones.csv"
    
    try:
        dbutils.fs.ls(dbfs_path)
        print("Airport timezones file exists, skipping download")
    except:
        with open(local_path, "wb") as f:
            f.write(requests.get(url).content)
        dbutils.fs.cp("file:" + local_path, dbfs_path)
    
    df_airport_timezones = (
        spark.read
        .option("header", True)
        .option("delimiter", "^")
        .option("inferSchema", True)
        .csv(dbfs_path)
        .select("iata_code", "icao_code", "faa_code", "timezone", "latitude", "longitude")
    )
    return df_airport_timezones


def prepare_airport_data(df_airport_codes, df_airport_timezones):
    """Join airport codes with timezones and prepare for use."""
    # Join airport codes with timezones
    df_airport_joined = (
        df_airport_codes.alias("a")
        .join(
            df_airport_timezones.alias("b"),
            (F.col("a.iata_code") == F.col("b.iata_code")) | (F.col("a.icao_code") == F.col("b.icao_code")),
            "left"
        )
    )
    
    df_airport_joined_clean = df_airport_joined.select(
        "ident", "type", "name", "elevation_ft", "continent", "iso_country", "iso_region", "municipality",
        "a.icao_code", "a.iata_code", "gps_code", "local_code", "coordinates",
        F.col("b.latitude").alias("latitude"),
        F.col("b.longitude").alias("longitude"),
        F.col("b.timezone").alias("timezone"),
        F.col("b.faa_code").alias("faa_code_otd")
    )
    
    # Deduplicate airports (prefer records with latitude and timezone)
    w = Window.partitionBy("iata_code", "icao_code").orderBy(
        F.when(F.col("latitude").isNotNull(), 1).otherwise(2),
        F.when(F.col("timezone").isNotNull(), 1).otherwise(2)
    )
    
    df_airport_dedup = (
        df_airport_joined_clean
        .withColumn("row_num", F.row_number().over(w))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )
    
    # Extract latitude/longitude from coordinates column
    df_airports = (
        df_airport_dedup
        .withColumn("coordinates", regexp_replace("coordinates", "[()]", ""))
        .withColumn("lat_lon", split("coordinates", ","))
        .withColumn("latitude", trim(col("lat_lon").getItem(0)))
        .withColumn("longitude", trim(col("lat_lon").getItem(1)))
        .drop("lat_lon")
        .cache()
    )
    
    return df_airports


def prepare_datasets(df_flights, df_weather, df_stations, df_airports):
    """Prepare all datasets: deduplicate and normalize column names."""
    # Deduplicate all datasets
    df_flights = df_flights.dropDuplicates()
    df_weather = df_weather.dropDuplicates()
    df_stations = df_stations.dropDuplicates()
    df_airports = df_airports.dropDuplicates()
    
    # Convert column names to lowercase
    df_flights = df_flights.toDF(*[c.lower() for c in df_flights.columns])
    df_weather = df_weather.toDF(*[c.lower() for c in df_weather.columns])
    df_stations = df_stations.toDF(*[c.lower() for c in df_stations.columns])
    df_airports = df_airports.toDF(*[c.lower() for c in df_airports.columns])
    
    # Prepare station data: convert ICAO codes to IATA codes
    df_stations = df_stations.withColumn("neighbor_iata", F.expr("substring(neighbor_call, 2, 3)"))
    
    # Create scheduled departure timestamp for flights
    df_flights = df_flights.withColumn(
        "sched_depart_date_time",
        to_timestamp(F.concat(col("fl_date"), F.lpad(col("crs_dep_time"), 4, "0")), "yyyy-MM-ddHHmm")
    )
    
    # Add timestamps
    df_flights = df_flights.withColumn("fl_date_timestamp", to_timestamp(col("fl_date"), "yyyy-MM-dd"))
    df_weather = df_weather.withColumn("date_timestamp", F.to_timestamp(F.col("date"), "yyyy-MM-dd'T'HH:mm:ss"))
    
    return df_flights, df_weather, df_stations, df_airports


# ============================================================================
# JOIN FUNCTIONS
# ============================================================================

def join_airports(df_flights, df_airports):
    """Join flights with airport data to add latitude/longitude and timezone."""
    # Prepare and broadcast airports
    df_airports_clean = (
        df_airports
        .select("iata_code", "name", "latitude", "longitude", "iso_country", "timezone")
        .withColumn("latitude", col("latitude").cast("double"))
        .withColumn("longitude", col("longitude").cast("double"))
        .filter(col("iata_code").isNotNull() & col("latitude").isNotNull() & col("longitude").isNotNull())
    )
    
    airports_broadcast = broadcast(df_airports_clean)
    
    # Join flights with airports (LEFT JOIN preserves all flights)
    df_flights_with_airports = (
        df_flights.alias("f")
        .join(airports_broadcast.alias("ao"), col("f.origin") == col("ao.iata_code"), "left")
        .join(airports_broadcast.alias("ad"), col("f.dest") == col("ad.iata_code"), "left")
        .select(
            col("f.*"),
            col("ao.name").alias("origin_airport_name"),
            col("ao.latitude").alias("origin_latitude"),
            col("ao.longitude").alias("origin_longitude"),
            col("ao.iso_country").alias("origin_country"),
            col("ao.timezone").alias("origin_timezone"),
            col("ad.name").alias("destination_airport_name"),
            col("ad.latitude").alias("destination_latitude"),
            col("ad.longitude").alias("destination_longitude"),
            col("ad.iso_country").alias("destination_country"),
            col("ad.timezone").alias("destination_timezone")
        )
    )
    
    # Convert to UTC timestamps
    df_flights_with_airports = (
        df_flights_with_airports
        .withColumn("sched_depart_date_time_UTC", F.to_utc_timestamp("sched_depart_date_time", F.col("origin_timezone")))
        .withColumn("two_hours_prior_depart_UTC", expr("sched_depart_date_time_UTC - INTERVAL 2 HOURS"))
        .withColumn("four_hours_prior_depart_UTC", expr("sched_depart_date_time_UTC - INTERVAL 4 HOURS"))
    )
    
    return df_flights_with_airports, df_airports_clean


def join_stations(df_flights_with_airports, df_stations):
    """Match airports to weather stations using spatial join."""
    # Get distinct origin airports with coordinates
    df_distinct_origins = (
        df_flights_with_airports
        .select("origin", "origin_latitude", "origin_longitude")
        .distinct()
        .filter(col("origin_latitude").isNotNull() & col("origin_longitude").isNotNull())
        .repartition(200, "origin")
    )
    
    # Prepare stations
    df_stations_clean = (
        df_stations
        .select("station_id", col("lat").cast("double"), col("lon").cast("double"))
        .filter(col("station_id").isNotNull() & col("lat").isNotNull() & col("lon").isNotNull())
    )
    stations_broadcast = broadcast(df_stations_clean)
    
    # Spatial join with bounding box filter (0.5 degree ≈ 55km)
    df_candidates = (
        df_distinct_origins.alias("a")
        .join(
            stations_broadcast.alias("s"),
            (col("s.lat").between(col("a.origin_latitude") - 0.5, col("a.origin_latitude") + 0.5)) &
            (col("s.lon").between(col("a.origin_longitude") - 0.5, col("a.origin_longitude") + 0.5)),
            "inner"
        )
        .withColumn(
            "distance_km",
            F.expr("""
                6371 * acos(
                    least(1.0,
                        cos(radians(a.origin_latitude)) * cos(radians(s.lat)) *
                        cos(radians(s.lon) - radians(a.origin_longitude)) +
                        sin(radians(a.origin_latitude)) * sin(radians(s.lat))
                    )
                )
            """)
        )
        .filter(col("distance_km") < 50)
    )
    
    # Get nearest station per airport
    window_spec = Window.partitionBy("a.origin").orderBy("distance_km")
    df_nearest_stations = (
        df_candidates
        .withColumn("rank", F.row_number().over(window_spec))
        .filter(col("rank") == 1)
        .select(
            col("a.origin").alias("origin"),
            col("s.station_id").alias("origin_station_id"),
            col("distance_km").alias("station_distance_km")
        )
    )
    
    # Join stations back to flights (LEFT JOIN preserves all flights)
    df_flights_with_station = df_flights_with_airports.join(df_nearest_stations, on="origin", how="left")
    
    # Filter out flights without stations (CRITICAL DROP POINT)
    df_flights_with_station_clean = df_flights_with_station.filter(col("origin_station_id").isNotNull())
    
    return df_flights_with_station_clean


def join_weather(df_flights_with_station, df_weather, bucket_interval_minutes=30):
    """Join flights with weather data using station + bucket co-partitioning."""
    print("=" * 60)
    print("JOIN 3: Flights + Weather (Station + Bucket Co-Partitioned)")
    print("=" * 60)
    
    # Disable Photon for this complex join step
    spark.conf.set("spark.databricks.photon.enabled", "false")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    
    bucket_interval_seconds = bucket_interval_minutes * 60
    
    # Ensure station IDs have consistent types
    df_weather = df_weather.withColumn("station", col("station").cast("string"))
    df_flights_with_station = df_flights_with_station.withColumn(
        "origin_station_id", col("origin_station_id").cast("string")
    )
    
    # Weather: compute base bucket
    df_weather_bucketed = (
        df_weather
        .withColumn("weather_ts", col("date_timestamp").cast("timestamp"))
        .withColumn(
            "bucket",
            (col("weather_ts").cast("long") / F.lit(bucket_interval_seconds)).cast("long") * F.lit(bucket_interval_seconds)
        )
    )
    
    # Create shifted version for previous bucket
    df_weather_shifted_for_prev = df_weather_bucketed.withColumn(
        "bucket", col("bucket") + F.lit(bucket_interval_seconds)
    )
    
    # Union original + shifted weather
    df_weather_for_join = df_weather_bucketed.unionByName(df_weather_shifted_for_prev)
    df_weather_for_join = df_weather_for_join.repartition("station", "bucket")
    
    # Flights: compute bucket using "two_hours_prior_depart_utc"
    df_flights_with_buckets = (
        df_flights_with_station
        .withColumn("flight_ts", col("two_hours_prior_depart_UTC").cast("timestamp"))
        .withColumn(
            "bucket",
            (col("flight_ts").cast("long") / F.lit(bucket_interval_seconds)).cast("long") * F.lit(bucket_interval_seconds)
        )
    )
    df_flights_with_buckets = df_flights_with_buckets.repartition("origin_station_id", "bucket")
    
    # Join on station + bucket
    print("\nExecuting station + bucket join...")
    df_weather_for_join = df_weather_for_join.withColumnRenamed("year", "weather_year")
    
    df_joined = (
        df_flights_with_buckets.alias("f")
        .join(
            df_weather_for_join.alias("w"),
            ((col("f.origin_station_id") == col("w.station")) & (col("f.bucket") == col("w.bucket"))),
            "left"
        )
        .select(col("f.*"), col("w.*"))
    )
    
    # Filter weather <= flight time and rank
    print("\nFiltering weather <= flight time and ranking...")
    df_joined_filtered = df_joined.filter(
        (col("w.weather_ts").isNull()) | (col("w.weather_ts") <= col("f.flight_ts"))
    )
    
    df_with_diff = (
        df_joined_filtered
        .withColumn(
            "time_diff_sec",
            F.when(col("w.weather_ts").isNull(), F.lit(None).cast("long"))
            .otherwise(col("f.flight_ts").cast("long") - col("w.weather_ts").cast("long"))
        )
    )
    
    window_closest = Window.partitionBy(
        "f.origin_station_id", "f.fl_date", "f.crs_dep_time", "f.op_carrier_fl_num"
    ).orderBy(
        col("time_diff_sec").isNull().asc(),
        col("time_diff_sec").asc_nulls_last()
    )
    
    df_ranked = (
        df_with_diff
        .withColumn("has_weather", col("w.weather_ts").isNotNull())
        .withColumn("weather_rank", F.row_number().over(window_closest))
    )
    
    # Keep closest weather or all flights with no weather
    df_final = (
        df_ranked
        .filter((~col("has_weather")) | (col("weather_rank") == 1))
        .drop("weather_rank", "time_diff_sec", "bucket", "station", "has_weather", "weather_ts", "flight_ts")
    )
    
    print("✓ Closest-weather selection complete (all flights retained).")
    
    # Re-enable Photon
    spark.conf.set("spark.databricks.photon.enabled", "true")
    
    print("\n" + "=" * 60)
    print("✓ JOIN 3 COMPLETE (Station + Bucket Co-Partitioned)")
    print("=" * 60)
    
    return df_final


# ============================================================================
# MAIN PIPELINE
# ============================================================================
# Note: Flight lineage features are added via the flight_lineage module
# See flight_lineage.py for implementation details

def run_custom_join_pipeline(spark, data_version=DATA_VERSION, section=SECTION, number=NUMBER, bucket_interval_minutes=BUCKET_INTERVAL_MINUTES):
    """Run the complete custom join pipeline."""
    print("="*80)
    print("CUSTOM JOIN PIPELINE")
    print("="*80)
    
    # Setup
    setup_spark_config(spark)
    flights_weather_joined_path, raw_dir = setup_paths(section, number, data_version)
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    df_flights, df_stations, df_weather = load_base_datasets(spark, data_version)
    initial_count = sanity_check(df_flights, "After Initial Load")
    
    df_airport_codes = load_airport_codes(spark, raw_dir)
    df_airport_timezones = load_airport_timezones(spark)
    df_airports = prepare_airport_data(df_airport_codes, df_airport_timezones)
    
    # Prepare datasets
    df_flights, df_weather, df_stations, df_airports = prepare_datasets(
        df_flights, df_weather, df_stations, df_airports
    )
    dedup_count = sanity_check(df_flights, "After dropDuplicates", initial_count)
    
    # Join airports
    print("\n" + "="*80)
    print("STEP 2: Joining with Airport Data")
    print("="*80)
    df_flights_with_airports, df_airports_clean = join_airports(df_flights, df_airports)
    airport_count = sanity_check(df_flights_with_airports, "After Airport Join", dedup_count)
    
    # Join stations
    print("\n" + "="*80)
    print("STEP 3: Matching Airports to Weather Stations")
    print("="*80)
    df_flights_with_station_clean = join_stations(df_flights_with_airports, df_stations)
    station_join_count = sanity_check(df_flights_with_airports.join(
        df_flights_with_airports.select("origin").distinct().join(
            df_stations.select("station_id").distinct().withColumnRenamed("station_id", "origin_station_id"),
            on=expr("1=1"), how="cross"
        ).select("origin", "origin_station_id").distinct(),
        on="origin", how="left"
    ), "After Station Join (Left)", airport_count)
    station_filter_count = sanity_check(df_flights_with_station_clean, "After Station Filter", station_join_count)
    
    # Join weather
    print("\n" + "="*80)
    print("STEP 4: Joining with Weather Data")
    print("="*80)
    df_final = join_weather(df_flights_with_station_clean, df_weather, bucket_interval_minutes)
    weather_count = sanity_check(df_final, "After Weather Join", station_filter_count)
    
    # Add flight lineage features
    print("\n" + "="*80)
    print("STEP 5: Adding Flight Lineage Features")
    print("="*80)
    df_final = add_flight_lineage_features(df_final)
    final_count = sanity_check(df_final, "Final Before Save", weather_count)
    
    # Save
    print("\n" + "="*80)
    print("STEP 6: Saving Final Result")
    print("="*80)
    num_partitions = df_final.rdd.getNumPartitions()
    if num_partitions > 500:
        print(f"⚠ Too many partitions, coalescing to 200")
        df_final = df_final.coalesce(200)
    elif num_partitions < 10:
        print(f"⚠ Too few partitions, repartitioning to 50")
        df_final = df_final.repartition(50)
    else:
        print(f"✓ Partition count looks good: {num_partitions}")
    
    df_final.repartition(200).write.mode("overwrite").parquet(flights_weather_joined_path)
    print(f"✓ Data saved to: {flights_weather_joined_path}")
    
    # Cleanup
    df_airports_clean.unpersist()
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Initial Load: {initial_count:,}")
    print(f"After dropDuplicates: {dedup_count:,} ({((dedup_count-initial_count)/initial_count*100):+.2f}%)")
    print(f"After Airport Join: {airport_count:,} ({((airport_count-dedup_count)/dedup_count*100):+.2f}%)")
    print(f"After Station Filter: {station_filter_count:,} ({((station_filter_count-airport_count)/airport_count*100):+.2f}%)")
    print(f"After Weather Join: {weather_count:,} ({((weather_count-station_filter_count)/station_filter_count*100):+.2f}%)")
    print(f"Final: {final_count:,} ({((final_count-initial_count)/initial_count*100):+.2f}%)")
    print(f"\nTotal Retention: {(final_count/initial_count*100):.2f}%")
    print("="*80)
    
    return df_final


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # In Databricks, spark and dbutils are available globally
    # For local testing, you would need to create a SparkSession
    try:
        # Check if running in Databricks
        if 'spark' not in globals():
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("CustomJoin").getOrCreate()
        if 'dbutils' not in globals():
            # For local testing, create a mock dbutils
            class MockDbutils:
                class Fs:
                    def mkdirs(self, path): pass
                    def ls(self, path): raise Exception("File not found")
                    def cp(self, src, dst): pass
                def __init__(self): self.fs = self.Fs()
            dbutils = MockDbutils()
        
        df_final = run_custom_join_pipeline(spark)
        print("\n✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise

