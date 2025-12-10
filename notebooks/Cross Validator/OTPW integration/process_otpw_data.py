#!/usr/bin/env python3
"""
process_otpw_data.py - Process OTPW data and output in Custom Join format

This script processes OTPW data for 3M, 12M, and 60M, applies column mapping,
filters to correct date ranges, and outputs in the same format as Custom Join
so that split.py can seamlessly pick it up.

Output format matches Custom Join:
- dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m
- dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015
- dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M

Usage:
    Set VERSIONS list below, then run all cells.
    Or run as script: python process_otpw_data.py
"""

from __future__ import annotations
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date
import importlib.util
import sys
import os

# Load column mapping module
column_mapping_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/OTPW integration/column_mapping.py"
try:
    spec = importlib.util.spec_from_file_location("column_mapping", column_mapping_path)
except:
    # Fallback to relative path for local development
    script_dir = os.path.dirname(os.path.abspath(__file__))
    column_mapping_path = os.path.join(script_dir, "column_mapping.py")
    spec = importlib.util.spec_from_file_location("column_mapping", column_mapping_path)

column_mapping = importlib.util.module_from_spec(spec)
sys.modules["column_mapping"] = column_mapping
spec.loader.exec_module(column_mapping)

map_otpw_columns_to_custom = column_mapping.map_otpw_columns_to_custom

# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M"])
VERSIONS = ["3M", "12M", "60M"]  # <-- EDIT THIS LIST

# OTPW source paths (from Indri's analysis)
OTPW_SOURCES = {
    "3M": "dbfs:/mnt/mids-w261/OTPW_3M_2015.csv",  # CSV format
    "12M": "dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M",  # CSV format (directory)
    "60M": "dbfs:/mnt/mids-w261/OTPW_60M/OTPW_60M/",  # Parquet format (different path!)
}

# Output paths (matches Custom Join structure)
OUTPUT_PATHS = {
    "3M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m",
    "12M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015",
    "60M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M",
}

# Date filtering (CRITICAL for 60M)
DATE_FILTER_START = "2015-01-01"
DATE_FILTER_END = "2020-01-01"  # End of 2019 (exclusive)

WRITE_MODE = "overwrite"
VERBOSE = True

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def load_otpw_data(version: str, spark: SparkSession, verbose: bool = VERBOSE) -> DataFrame:
    """
    Load OTPW data for the specified version.
    
    Args:
        version: Version string ("3M", "12M", or "60M")
        spark: SparkSession
        verbose: Whether to print verbose output
        
    Returns:
        DataFrame with OTPW data
    """
    if version not in OTPW_SOURCES:
        raise ValueError(f"Unknown version: {version}. Available: {list(OTPW_SOURCES.keys())}")
    
    source_path = OTPW_SOURCES[version]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Loading OTPW {version} data")
        print(f"{'='*80}")
        print(f"Source: {source_path}")
    
    try:
        if version == "60M":
            # 60M is in parquet format
            df = spark.read.parquet(source_path)
            if verbose:
                print(f"✓ Loaded {version} from parquet: {df.count():,} rows, {len(df.columns)} columns")
        else:
            # 3M and 12M are in CSV format
            df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(source_path)
            if verbose:
                print(f"✓ Loaded {version} from CSV: {df.count():,} rows, {len(df.columns)} columns")
        
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load OTPW {version} data from {source_path}: {e}")


def validate_date_range(df: DataFrame, version: str, date_col: str = "fl_date", verbose: bool = VERBOSE) -> tuple:
    """
    Validate and report date range in the DataFrame.
    
    Args:
        df: DataFrame to check
        version: Version string for reporting
        date_col: Date column name
        
    Returns:
        tuple: (min_date, max_date, row_count)
    """
    # Ensure date column is date type
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Convert to date if needed
    if df.schema[date_col].dataType.typeName() != "date":
        df = df.withColumn(date_col, to_date(col(date_col)))
    
    date_range = df.select(
        F.min(F.col(date_col)).alias("min_date"),
        F.max(F.col(date_col)).alias("max_date"),
        F.count("*").alias("count")
    ).first()
    
    min_date = date_range["min_date"]
    max_date = date_range["max_date"]
    count = date_range["count"]
    
    if verbose:
        print(f"\n  Date range validation:")
        print(f"    Min date: {min_date}")
        print(f"    Max date: {max_date}")
        print(f"    Total rows: {count:,}")
    
    return min_date, max_date, count


def filter_to_date_range(df: DataFrame, date_col: str = "fl_date", 
                        start_date: str = DATE_FILTER_START, 
                        end_date: str = DATE_FILTER_END,
                        version: str = "",
                        verbose: bool = VERBOSE) -> DataFrame:
    """
    Filter DataFrame to specified date range.
    
    Args:
        df: DataFrame to filter
        date_col: Date column name
        start_date: Start date (inclusive)
        end_date: End date (exclusive)
        version: Version string for reporting
        
    Returns:
        Filtered DataFrame
    """
    # Ensure date column is date type
    if df.schema[date_col].dataType.typeName() != "date":
        df = df.withColumn(date_col, to_date(col(date_col)))
    
    initial_count = df.count()
    
    df_filtered = df.filter(
        (F.col(date_col) >= F.lit(start_date)) & 
        (F.col(date_col) < F.lit(end_date))
    )
    
    filtered_count = df_filtered.count()
    removed = initial_count - filtered_count
    
    if verbose:
        print(f"\n  Date filtering ({start_date} to {end_date}):")
        print(f"    Initial rows: {initial_count:,}")
        print(f"    Filtered rows: {filtered_count:,}")
        print(f"    Removed: {removed:,} ({removed/initial_count*100:.2f}%)")
    
    if removed > 0:
        print(f"  ⚠ Removed {removed:,} rows outside date range")
    
    # Validate filtered date range
    min_date, max_date, _ = validate_date_range(df_filtered, version, date_col)
    
    # Check if filtering was correct
    if min_date and min_date.strftime("%Y-%m-%d") < start_date:
        print(f"  ⚠ WARNING: Min date {min_date} is before filter start {start_date}")
    if max_date and max_date.strftime("%Y-%m-%d") >= end_date:
        print(f"  ⚠ WARNING: Max date {max_date} is after filter end {end_date}")
    
    return df_filtered


def process_otpw_version(version: str, spark: SparkSession, verbose: bool = VERBOSE) -> DataFrame:
    """
    Process OTPW data for a specific version.
    
    Steps:
    1. Load OTPW data
    2. Apply column mapping
    3. Filter to date range (if 60M, filter to 2015-2019)
    4. Validate date range
    5. Return processed DataFrame
    
    Args:
        version: Version string ("3M", "12M", or "60M")
        spark: SparkSession
        
    Returns:
        Processed DataFrame ready to save
    """
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*80}")
        print(f"[{timestamp}] Processing OTPW {version}")
        print(f"{'='*80}")
    
    # Step 1: Load OTPW data
    df_otpw = load_otpw_data(version, spark, verbose=verbose)
    
    # Step 2: Apply column mapping
    if verbose:
        print(f"\n  Applying column mapping...")
    df_mapped = map_otpw_columns_to_custom(df_otpw, verbose=VERBOSE)
    
    if verbose:
        print(f"  ✓ Column mapping applied: {len(df_mapped.columns)} columns")
        print(f"    Sample columns: {sorted(df_mapped.columns)[:10]}")
    
    # Step 3: Filter to date range (CRITICAL for 60M)
    if version == "60M":
        if verbose:
            print(f"\n  Filtering 60M to 2015-2019 (CRITICAL)...")
        df_mapped = filter_to_date_range(
            df_mapped, 
            date_col="fl_date",
            start_date=DATE_FILTER_START,
            end_date=DATE_FILTER_END,
            version=version,
            verbose=verbose
        )
    else:
        # For 3M and 12M, validate date range but don't filter (they should already be correct)
        if verbose:
            print(f"\n  Validating date range for {version}...")
        min_date, max_date, count = validate_date_range(df_mapped, version, "fl_date", verbose=verbose)
        
        # Warn if dates are outside expected range
        if min_date and min_date.strftime("%Y-%m-%d") < DATE_FILTER_START:
            print(f"  ⚠ WARNING: {version} has dates before {DATE_FILTER_START}")
        if max_date and max_date.strftime("%Y-%m-%d") >= DATE_FILTER_END:
            print(f"  ⚠ WARNING: {version} has dates after {DATE_FILTER_END}")
    
    # Step 4: Final validation
    if verbose:
        print(f"\n  Final validation:")
        print(f"    Rows: {df_mapped.count():,}")
        print(f"    Columns: {len(df_mapped.columns)}")
        
        # Check critical columns
        critical_cols = ["fl_date", "DEP_DELAY", "dep_delay", "origin", "dest", "op_carrier", "tail_num"]
        missing = [c for c in critical_cols if c not in df_mapped.columns]
        if missing:
            print(f"    ⚠ Missing critical columns: {missing}")
        else:
            print(f"    ✓ All critical columns present")
    
    if verbose:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n  ✓ Processing complete (took {duration:.1f}s / {duration/60:.1f}min)")
    
    return df_mapped


def save_processed_data(df: DataFrame, version: str, spark: SparkSession, verbose: bool = VERBOSE):
    """
    Save processed DataFrame to output path (matches Custom Join format).
    
    Args:
        df: Processed DataFrame
        version: Version string
        spark: SparkSession
    """
    if version not in OUTPUT_PATHS:
        raise ValueError(f"Unknown version: {version}. Available: {list(OUTPUT_PATHS.keys())}")
    
    output_path = OUTPUT_PATHS[version]
    
    if verbose:
        print(f"\n  Saving to: {output_path}")
        print(f"    Rows: {df.count():,}")
        print(f"    Columns: {len(df.columns)}")
        print(f"    Mode: {WRITE_MODE}")
    
    try:
        df.write.mode(WRITE_MODE).parquet(output_path)
        if verbose:
            print(f"  ✓ Saved successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to save {version} data to {output_path}: {e}")


# -------------------------
# MAIN PROCESSING
# -------------------------

def main():
    """Main processing function."""
    spark = SparkSession.builder.getOrCreate()
    
    print("="*80)
    print("OTPW DATA PROCESSING")
    print("="*80)
    print(f"Versions to process: {VERSIONS}")
    print(f"Date filter (60M): {DATE_FILTER_START} to {DATE_FILTER_END}")
    print(f"Output format: Custom Join compatible")
    print("="*80)
    
    for version in VERSIONS:
        try:
            # Process OTPW data
            df_processed = process_otpw_version(version, spark)
            
            # Save to Custom Join format
            save_processed_data(df_processed, version, spark, verbose=VERBOSE)
            
            print(f"\n✅ Successfully processed and saved {version}")
            
        except Exception as e:
            print(f"\n❌ ERROR processing {version}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next version
            continue
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\nProcessed versions: {VERSIONS}")
    print(f"\nOutput files (ready for split.py):")
    for version in VERSIONS:
        if version in OUTPUT_PATHS:
            print(f"  {version}: {OUTPUT_PATHS[version]}")
    
    print(f"\nNext steps:")
    print(f"  1. Verify output files exist")
    print(f"  2. Run split.py with SOURCE='OTPW' (or modify split.py to use OTPW paths)")
    print(f"  3. split.py will read from these paths and create folds")


if __name__ == "__main__":
    main()

