#!/usr/bin/env python3
"""
validate_timezone.py - Validate if OTPW sched_depart_date_time is UTC or local time

This script compares equivalent rows between OTPW and Custom Join datasets to determine
if OTPW's sched_depart_date_time is in UTC or local time.

If times match exactly: OTPW is in UTC
If times differ by timezone offset: OTPW is in local time (problematic!)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, abs as spark_abs
import importlib.util
import sys

# Load column mapping module
column_mapping_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/OTPW integration/column_mapping.py"
try:
    spec = importlib.util.spec_from_file_location("column_mapping", column_mapping_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load column_mapping from {column_mapping_path}")
except (FileNotFoundError, Exception) as e:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    column_mapping_path = os.path.join(script_dir, "column_mapping.py")
    print(f"⚠ Using fallback path for column_mapping: {column_mapping_path}")
    spec = importlib.util.spec_from_file_location("column_mapping", column_mapping_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load column_mapping from either path")

column_mapping = importlib.util.module_from_spec(spec)
sys.modules["column_mapping"] = column_mapping
spec.loader.exec_module(column_mapping)
map_otpw_columns_to_custom = column_mapping.map_otpw_columns_to_custom

spark = SparkSession.builder.appName("ValidateTimezone").getOrCreate()

print("="*80)
print("TIMEZONE VALIDATION: OTPW vs Custom Join")
print("="*80)

# Configuration: Which version to check
VERSION = "60M"  # Options: "3M", "12M", "60M"
# NOTE: 60M is especially important as it uses a different data source!

print(f"\n⚠️  Checking version: {VERSION}")
if VERSION == "60M":
    print("   ⚠️  60M uses OTPW_60M_Backup (different source) - timezone may differ!")

# Load OTPW data
print(f"\n1. Loading OTPW {VERSION} data...")
if VERSION == "3M":
    otpw_path = "dbfs:/mnt/mids-w261/OTPW_3M_2015.csv"
    df_otpw_raw = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(otpw_path)
elif VERSION == "12M":
    otpw_path = "dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M"
    df_otpw_raw = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(otpw_path)
elif VERSION == "60M":
    otpw_path = "dbfs:/mnt/mids-w261/OTPW_60M_Backup/"
    df_otpw_raw = spark.read.parquet(otpw_path)
else:
    raise ValueError(f"Unknown version: {VERSION}")

print(f"   ✓ Loaded: {df_otpw_raw.count():,} rows, {len(df_otpw_raw.columns)} columns")
print(f"   Source: {otpw_path}")

# Apply column mapping
print(f"\n2. Applying column mapping to OTPW {VERSION}...")
df_otpw = map_otpw_columns_to_custom(df_otpw_raw, verbose=False)
print(f"   ✓ Mapped: {len(df_otpw.columns)} columns")

# Filter 60M to 2015-2019 for fair comparison
if VERSION == "60M":
    print(f"\n2b. Filtering 60M to 2015-2019 for comparison...")
    initial_count = df_otpw.count()
    df_otpw = df_otpw.filter(
        (F.col("fl_date") >= F.lit("2015-01-01")) & 
        (F.col("fl_date") < F.lit("2020-01-01"))
    )
    filtered_count = df_otpw.count()
    print(f"   ✓ Filtered: {filtered_count:,} rows (from {initial_count:,})")

# Load Custom Join data (matching version)
print(f"\n3. Loading Custom Join {VERSION} data...")
if VERSION == "3M":
    custom_path = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m"
elif VERSION == "12M":
    custom_path = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015"
elif VERSION == "60M":
    custom_path = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M"
else:
    raise ValueError(f"Unknown version: {VERSION}")

df_custom = spark.read.parquet(custom_path)
print(f"   ✓ Loaded: {df_custom.count():,} rows, {len(df_custom.columns)} columns")
print(f"   Source: {custom_path}")

# Check what time columns exist
print("\n4. Checking available time columns...")
otpw_time_cols = [c for c in df_otpw.columns if "sched" in c.lower() or "depart" in c.lower() or "time" in c.lower()]
custom_time_cols = [c for c in df_custom.columns if "sched" in c.lower() or "depart" in c.lower() or "time" in c.lower()]

print(f"   OTPW time columns: {sorted(otpw_time_cols)[:10]}")
print(f"   Custom time columns: {sorted(custom_time_cols)[:10]}")

# Check if sched_depart_date_time exists in OTPW
has_otpw_sched = "sched_depart_date_time" in df_otpw.columns
has_otpw_sched_utc = "sched_depart_date_time_utc" in df_otpw.columns
has_custom_sched = "sched_depart_date_time" in df_custom.columns
has_custom_sched_utc = "sched_depart_date_time_utc" in df_custom.columns

print(f"\n   OTPW has sched_depart_date_time: {has_otpw_sched}")
print(f"   OTPW has sched_depart_date_time_utc: {has_otpw_sched_utc}")
print(f"   Custom has sched_depart_date_time: {has_custom_sched}")
print(f"   Custom has sched_depart_date_time_utc: {has_custom_sched_utc}")

# Prepare join keys (use multiple keys for accurate matching)
join_keys = ["tail_num", "fl_date", "origin", "dest", "op_carrier_fl_num", "crs_dep_time"]
# Check which keys exist in both datasets
available_keys = []
for key in join_keys:
    if key in df_otpw.columns and key in df_custom.columns:
        available_keys.append(key)
    else:
        print(f"   ⚠ Key '{key}' missing in one or both datasets")

print(f"\n5. Using join keys: {available_keys}")

# Sample data for comparison (take a sample to speed up)
print("\n6. Sampling data for comparison...")
df_otpw_sample = df_otpw.filter(
    col("tail_num").isNotNull() &
    col("fl_date").isNotNull() &
    col("origin").isNotNull() &
    col("dest").isNotNull()
).sample(fraction=0.01, seed=42)  # 1% sample

df_custom_sample = df_custom.filter(
    col("tail_num").isNotNull() &
    col("fl_date").isNotNull() &
    col("origin").isNotNull() &
    col("dest").isNotNull()
).sample(fraction=0.01, seed=42)  # 1% sample

print(f"   OTPW sample: {df_otpw_sample.count():,} rows")
print(f"   Custom sample: {df_custom_sample.count():,} rows")

# Join on matching keys
print("\n7. Joining datasets on matching keys...")
df_joined = df_otpw_sample.alias("otpw").join(
    df_custom_sample.alias("custom"),
    on=[col(f"otpw.{key}") == col(f"custom.{key}") for key in available_keys],
    how="inner"
)

join_count = df_joined.count()
print(f"   ✓ Joined: {join_count:,} matching rows")

if join_count == 0:
    print("\n   ❌ No matching rows found! Cannot compare timestamps.")
    print("   This might indicate:")
    print("   - Different date ranges")
    print("   - Different data sources")
    print("   - Column name mismatches")
    sys.exit(1)

# Compare timestamps
print("\n8. Comparing timestamps...")

# Select columns for comparison with proper aliases
comparison_cols = []

# Add join key columns with aliases (for reference)
for key in available_keys:
    comparison_cols.append(col(f"otpw.{key}").alias(f"otpw_{key}"))
    comparison_cols.append(col(f"custom.{key}").alias(f"custom_{key}"))

# Add time columns if they exist
if has_otpw_sched:
    comparison_cols.append(col("otpw.sched_depart_date_time").alias("otpw_sched_depart_date_time"))
if has_otpw_sched_utc:
    comparison_cols.append(col("otpw.sched_depart_date_time_utc").alias("otpw_sched_depart_date_time_utc"))
if has_custom_sched:
    comparison_cols.append(col("custom.sched_depart_date_time").alias("custom_sched_depart_date_time"))
if has_custom_sched_utc:
    comparison_cols.append(col("custom.sched_depart_date_time_utc").alias("custom_sched_depart_date_time_utc"))

df_comparison = df_joined.select(*comparison_cols)

# Debug: Check null counts in time columns
print("\n   Debug: Checking null counts in time columns...")
null_check_cols = []
if has_otpw_sched:
    null_check_cols.append(F.sum(F.when(col("otpw_sched_depart_date_time").isNull(), 1).otherwise(0)).alias("otpw_sched_nulls"))
if has_otpw_sched_utc:
    null_check_cols.append(F.sum(F.when(col("otpw_sched_depart_date_time_utc").isNull(), 1).otherwise(0)).alias("otpw_sched_utc_nulls"))
if has_custom_sched:
    null_check_cols.append(F.sum(F.when(col("custom_sched_depart_date_time").isNull(), 1).otherwise(0)).alias("custom_sched_nulls"))
if has_custom_sched_utc:
    null_check_cols.append(F.sum(F.when(col("custom_sched_depart_date_time_utc").isNull(), 1).otherwise(0)).alias("custom_sched_utc_nulls"))

if null_check_cols:
    null_stats = df_comparison.select(
        F.count("*").alias("total_rows"),
        *null_check_cols
    ).first()
    print(f"   Total joined rows: {null_stats['total_rows']:,}")
    for col_name in null_stats.asDict():
        if col_name != "total_rows":
            null_count = null_stats[col_name]
            non_null_count = null_stats['total_rows'] - null_count
            print(f"   {col_name}: {null_count:,} nulls, {non_null_count:,} non-nulls")

# Check data types
print("\n   Debug: Checking data types...")
if has_otpw_sched_utc:
    otpw_type = [f for f in df_comparison.schema.fields if f.name == "otpw_sched_depart_date_time_utc"]
    if otpw_type:
        print(f"   OTPW sched_depart_date_time_utc type: {otpw_type[0].dataType}")
if has_custom_sched_utc:
    custom_type = [f for f in df_comparison.schema.fields if f.name == "custom_sched_depart_date_time_utc"]
    if custom_type:
        print(f"   Custom sched_depart_date_time_utc type: {custom_type[0].dataType}")

# Show sample of actual values
print("\n   Debug: Sample values (first 5 rows)...")
sample_debug_cols = []
if has_otpw_sched_utc:
    sample_debug_cols.append(col("otpw_sched_depart_date_time_utc"))
if has_custom_sched_utc:
    sample_debug_cols.append(col("custom_sched_depart_date_time_utc"))
if sample_debug_cols:
    df_comparison.select(*sample_debug_cols).show(5, truncate=False)

# Calculate time differences
if has_otpw_sched and has_custom_sched_utc:
    print("\n   Comparing: OTPW.sched_depart_date_time vs Custom.sched_depart_date_time_utc")
    df_comparison = df_comparison.withColumn(
        "time_diff_seconds",
        F.when(
            col("otpw_sched_depart_date_time").isNotNull() & col("custom_sched_depart_date_time_utc").isNotNull(),
            F.abs(F.unix_timestamp(col("otpw_sched_depart_date_time")) - 
                  F.unix_timestamp(col("custom_sched_depart_date_time_utc")))
        ).otherwise(None)
    )
elif has_otpw_sched_utc and has_custom_sched_utc:
    print("\n   Comparing: OTPW.sched_depart_date_time_utc vs Custom.sched_depart_date_time_utc")
    # Try to handle both string and timestamp types
    df_comparison = df_comparison.withColumn(
        "otpw_ts",
        F.when(
            col("otpw_sched_depart_date_time_utc").isNotNull(),
            F.coalesce(
                F.unix_timestamp(col("otpw_sched_depart_date_time_utc")),
                F.unix_timestamp(col("otpw_sched_depart_date_time_utc"), "yyyy-MM-dd HH:mm:ss"),
                F.unix_timestamp(col("otpw_sched_depart_date_time_utc"), "yyyy-MM-dd'T'HH:mm:ss"),
                F.unix_timestamp(col("otpw_sched_depart_date_time_utc"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
            )
        ).otherwise(None)
    ).withColumn(
        "custom_ts",
        F.when(
            col("custom_sched_depart_date_time_utc").isNotNull(),
            F.coalesce(
                F.unix_timestamp(col("custom_sched_depart_date_time_utc")),
                F.unix_timestamp(col("custom_sched_depart_date_time_utc"), "yyyy-MM-dd HH:mm:ss"),
                F.unix_timestamp(col("custom_sched_depart_date_time_utc"), "yyyy-MM-dd'T'HH:mm:ss"),
                F.unix_timestamp(col("custom_sched_depart_date_time_utc"), "yyyy-MM-dd'T'HH:mm:ss.SSS")
            )
        ).otherwise(None)
    ).withColumn(
        "time_diff_seconds",
        F.when(
            col("otpw_ts").isNotNull() & col("custom_ts").isNotNull(),
            F.abs(col("otpw_ts") - col("custom_ts"))
        ).otherwise(None)
    )
    
    # Analyze differences
    diff_stats = df_comparison.select(
        F.count("*").alias("total"),
        F.count("time_diff_seconds").alias("with_times"),
        F.min("time_diff_seconds").alias("min_diff_sec"),
        F.max("time_diff_seconds").alias("max_diff_sec"),
        F.avg("time_diff_seconds").alias("avg_diff_sec"),
        F.percentile_approx("time_diff_seconds", 0.5).alias("median_diff_sec")
    ).first()
    
    print(f"\n   Time Difference Statistics:")
    print(f"   - Total rows: {diff_stats['total']:,}")
    print(f"   - Rows with both times: {diff_stats['with_times']:,}")
    if diff_stats['with_times'] > 0:
        print(f"   - Min difference: {diff_stats['min_diff_sec']:.0f} seconds ({diff_stats['min_diff_sec']/3600:.2f} hours)")
        print(f"   - Max difference: {diff_stats['max_diff_sec']:.0f} seconds ({diff_stats['max_diff_sec']/3600:.2f} hours)")
        print(f"   - Avg difference: {diff_stats['avg_diff_sec']:.0f} seconds ({diff_stats['avg_diff_sec']/3600:.2f} hours)")
        print(f"   - Median difference: {diff_stats['median_diff_sec']:.0f} seconds ({diff_stats['median_diff_sec']/3600:.2f} hours)")
        
        # Count exact matches (within 1 second)
        exact_matches = df_comparison.filter(
            (col("time_diff_seconds").isNotNull()) & (col("time_diff_seconds") <= 1)
        ).count()
        print(f"   - Exact matches (≤1 sec): {exact_matches:,} ({exact_matches/diff_stats['with_times']*100:.1f}%)")
        
        # Count timezone offset matches (common US timezones: -5, -6, -7, -8 hours = -18000, -21600, -25200, -28800 seconds)
        timezone_offsets = [-18000, -21600, -25200, -28800, -14400]  # EST, CST, MST, PST, EDT
        for offset_sec in timezone_offsets:
            offset_hours = offset_sec / 3600
            matches = df_comparison.filter(
                (col("time_diff_seconds").isNotNull()) & 
                (spark_abs(col("time_diff_seconds") - abs(offset_sec)) <= 60)  # Within 1 minute
            ).count()
            if matches > 0:
                print(f"   - ~{offset_hours:.0f} hour offset matches: {matches:,} ({matches/diff_stats['with_times']*100:.1f}%)")
        
        # Show sample rows
        print(f"\n9. Sample comparison rows:")
        print("   (Showing first 10 rows with time differences)")
        sample_cols = []
        # Add key columns for reference
        for key in available_keys[:3]:  # Show first 3 keys
            sample_cols.extend([
                col(f"otpw_{key}").alias(f"otpw_{key}"),
                col(f"custom_{key}").alias(f"custom_{key}")
            ])
        # Add time columns
        if has_otpw_sched:
            sample_cols.append(col("otpw_sched_depart_date_time"))
        if has_custom_sched_utc:
            sample_cols.append(col("custom_sched_depart_date_time_utc"))
        sample_cols.extend([
            col("time_diff_seconds"),
            (col("time_diff_seconds") / 3600).alias("time_diff_hours")
        ])
        df_comparison.filter(col("time_diff_seconds").isNotNull()).select(*sample_cols).show(10, truncate=False)
        
        # Conclusion
        print(f"\n{'='*80}")
        print("CONCLUSION:")
        print(f"{'='*80}")
        if exact_matches / diff_stats['with_times'] > 0.9:
            print("✅ OTPW sched_depart_date_time appears to be in UTC (matches Custom UTC)")
            print("   → Safe to use directly")
        elif diff_stats['avg_diff_sec'] > 10000:  # More than ~3 hours average difference
            print("⚠️  OTPW sched_depart_date_time appears to be in LOCAL TIME")
            print("   → This will cause issues! Need to convert to UTC")
            print(f"   → Average offset: {diff_stats['avg_diff_sec']/3600:.2f} hours")
        else:
            print("⚠️  OTPW sched_depart_date_time has some differences from Custom UTC")
            print("   → May be timezone-related or data quality issue")
            print(f"   → Average difference: {diff_stats['avg_diff_sec']/3600:.2f} hours")
        print(f"{'='*80}")

else:
    print("\n   ⚠️  Cannot compare - missing required time columns")
    print(f"   OTPW has sched_depart_date_time: {has_otpw_sched}")
    print(f"   OTPW has sched_depart_date_time_utc: {has_otpw_sched_utc}")
    print(f"   Custom has sched_depart_date_time: {has_custom_sched}")
    print(f"   Custom has sched_depart_date_time_utc: {has_custom_sched_utc}")
    print(f"   Need: (OTPW sched_depart_date_time OR sched_depart_date_time_utc)")
    print(f"   AND: (Custom sched_depart_date_time OR sched_depart_date_time_utc)")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)

