#!/usr/bin/env python3
"""
process_otpw_data.py - Process OTPW data and output in Custom Join format

This script processes OTPW data for 3M, 12M, and 60M, applies column mapping,
filters to correct date ranges, and outputs in the same format as Custom Join
so that split.py can seamlessly pick it up.

Output format matches Custom Join:
- dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m_otpw
- dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015_otpw
- dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M_otpw

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
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load column_mapping from {column_mapping_path}")
except (FileNotFoundError, Exception) as e:
    # Fallback to relative path for local development or if Databricks path doesn't work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    column_mapping_path = os.path.join(script_dir, "column_mapping.py")
    print(f"⚠ Using fallback path for column_mapping: {column_mapping_path}")
    spec = importlib.util.spec_from_file_location("column_mapping", column_mapping_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load column_mapping from either path. Tried:\n  - {column_mapping_path}\n  - {os.path.join(script_dir, 'column_mapping.py')}")

column_mapping = importlib.util.module_from_spec(spec)
sys.modules["column_mapping"] = column_mapping
spec.loader.exec_module(column_mapping)

map_otpw_columns_to_custom = column_mapping.map_otpw_columns_to_custom

# -------------------------
# CONFIGURATION
# -------------------------
# List of versions to process (e.g., ["3M", "12M", "60M"])
VERSIONS = ["3M","12M","60M"]  # <-- EDIT THIS LIST

# OTPW source paths (from Indri's analysis)
# NOTE: 60M uses OTPW_60M_Backup (as per Indri's code)
# However, this may be partitioned - if you see only ~800 rows, check partitions
OTPW_SOURCES = {
    "3M": "dbfs:/mnt/mids-w261/OTPW_3M_2015.csv",  # CSV format
    "12M": "dbfs:/mnt/mids-w261/OTPW_12M/OTPW_12M",  # CSV format (directory)
    "60M": "dbfs:/mnt/mids-w261/OTPW_60M_Backup/",  # Parquet format (as per Indri's code)
    # Alternative 60M paths if the above doesn't work:
    # "60M": "dbfs:/mnt/mids-w261/OTPW_60M/OTPW_60M/",  # Original path (incomplete)
    # "60M": [  # Raw airline data (would need weather join)
    #     "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/YEAR=2015",
    #     "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/YEAR=2016",
    #     "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/YEAR=2017",
    #     "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/YEAR=2018",
    #     "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/YEAR=2019"
    # ],
}

# Output paths (with OTPW suffix to avoid overwriting Custom Join)
OUTPUT_PATHS = {
    "3M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_3m_otpw",
    "12M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_2015_otpw",
    "60M": "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed/flights_weather_joined_60M_otpw",
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
            # 60M is in parquet format (OTPW_60M_Backup as per Indri's code)
            # Handle both single path and list of paths
            if isinstance(source_path, list):
                if verbose:
                    print(f"  Loading from {len(source_path)} paths...")
                    for path in source_path:
                        print(f"    - {path}")
                df = spark.read.parquet(*source_path)
            else:
                # Single path - read all partitions
                # Spark.read.parquet should automatically read all partitions
                # But if data is partitioned by PartitionId, we may need to read explicitly
                if verbose:
                    print(f"  Loading from: {source_path}")
                    print(f"  Note: Reading all partitions (Spark handles partition discovery automatically)")
                
                # Try reading the path - Spark will automatically discover and read all partitions
                df = spark.read.parquet(source_path)
                
                # Check if PartitionId column exists (indicates partitioned data)
                if "PartitionId" in df.columns and verbose:
                    partition_ids = df.select("PartitionId").distinct().collect()
                    partition_ids_list = [str(row["PartitionId"]) for row in partition_ids]
                    print(f"  Found PartitionId column with values: {sorted(partition_ids_list)}")
                    print(f"  This suggests data is partitioned - all partitions should be loaded")
            
            row_count = df.count()
            col_count = len(df.columns)
            if verbose:
                print(f"✓ Loaded {version} from parquet: {row_count:,} rows, {col_count} columns")
                # Warn if row count seems too low for 60M dataset
                if row_count < 10000:
                    print(f"\n  ⚠⚠⚠ CRITICAL WARNING ⚠⚠⚠")
                    print(f"  ⚠ Only {row_count:,} rows loaded for 60M - this is WAY too low!")
                    print(f"  ⚠ Expected: ~60,000,000 rows (60M)")
                    print(f"  ⚠ Current path: {source_path}")
                    print(f"  ⚠ This suggests:")
                    print(f"     1. Data is partitioned - only one partition loaded")
                    print(f"     2. Sample/test file instead of full dataset")
                    print(f"     3. Wrong path or incomplete dataset")
                    print(f"  ⚠ Try checking partitions: dbutils.fs.ls('{source_path}')")
                    print(f"  ⚠ Or try: dbfs:/mnt/mids-w261/OTPW_60M_Backup/PartitionId=*")
                    print(f"  ⚠⚠⚠")
                elif row_count < 1000000:
                    print(f"  ⚠ WARNING: Only {row_count:,} rows loaded for 60M")
                    print(f"  ⚠ Expected: ~60,000,000 rows (60M)")
                    print(f"  ⚠ This may indicate incomplete data or wrong path")
                elif row_count > 100000000:
                    print(f"  ⚠ WARNING: {row_count:,} rows seems very high - may include data outside 2015-2019")
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
        final_row_count = df_mapped.count()
        final_col_count = len(df_mapped.columns)
        print(f"    Rows: {final_row_count:,}")
        print(f"    Columns: {final_col_count}")
        
        # Check critical columns (dep_delay will be added after save/load to avoid ambiguity)
        critical_cols = ["fl_date", "DEP_DELAY", "origin", "dest", "op_carrier", "tail_num"]
        missing = [c for c in critical_cols if c not in df_mapped.columns]
        if missing:
            print(f"    ❌ Missing critical columns: {missing}")
            raise ValueError(f"Critical columns missing after processing: {missing}")
        else:
            print(f"    ✓ All critical columns present (dep_delay will be added after save/load)")
        
        # Validate critical column data quality
        print(f"\n  Data quality checks:")
        
        # Check for nulls in critical columns
        # Note: dep_delay will be added after save/load, so we only check DEP_DELAY here
        null_checks = {}
        for col_name in ["DEP_DELAY", "origin", "dest", "op_carrier", "tail_num"]:
            if col_name in df_mapped.columns:
                # Simple check - no ambiguity since we only have DEP_DELAY (not dep_delay yet)
                null_count = df_mapped.filter(F.col(col_name).isNull()).count()
                
                null_pct = (null_count / final_row_count * 100) if final_row_count > 0 else 0
                null_checks[col_name] = (null_count, null_pct)
                if null_pct > 50:
                    print(f"    ⚠ WARNING: {col_name} has {null_count:,} nulls ({null_pct:.1f}%)")
                elif null_count > 0:
                    print(f"    ℹ {col_name}: {null_count:,} nulls ({null_pct:.1f}%) - OK")
                else:
                    print(f"    ✓ {col_name}: No nulls")
        
        # Check date column
        if "fl_date" in df_mapped.columns:
            temp_df = df_mapped.select(F.col("fl_date").alias("_temp_fl_date"))
            null_dates = temp_df.filter(F.col("_temp_fl_date").isNull()).count()
            if null_dates > 0:
                print(f"    ⚠ WARNING: fl_date has {null_dates:,} nulls")
            else:
                print(f"    ✓ fl_date: No nulls")
        
        # Note: dep_delay will be added after save/load, so we skip the duplicate check here
        # The duplicate check will happen in save_processed_data() after both columns exist
        if "DEP_DELAY" in df_mapped.columns:
            print(f"    ✓ DEP_DELAY present (dep_delay will be added after save/load)")
        
        # Check row count is reasonable
        if final_row_count == 0:
            raise ValueError(f"ERROR: Processed DataFrame has 0 rows!")
        elif final_row_count < 1000:
            print(f"    ⚠ WARNING: Very low row count ({final_row_count:,}) - is this expected?")
        else:
            print(f"    ✓ Row count looks reasonable ({final_row_count:,})")
    
    if verbose:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n  ✓ Processing complete (took {duration:.1f}s / {duration/60:.1f}min)")
    
    return df_mapped


def save_processed_data(df: DataFrame, version: str, spark: SparkSession, verbose: bool = VERBOSE):
    """
    Save processed DataFrame to output path (matches Custom Join format).
    
    NOTE: We only save DEP_DELAY (uppercase). The flight_lineage_features.py script
    has been updated to handle both DEP_DELAY and dep_delay, so we don't need to
    create both columns. This simplifies the pipeline and avoids ambiguity issues.
    
    Args:
        df: Processed DataFrame (should have DEP_DELAY)
        version: Version string
        spark: SparkSession
    """
    if version not in OUTPUT_PATHS:
        raise ValueError(f"Unknown version: {version}. Available: {list(OUTPUT_PATHS.keys())}")
    
    output_path = OUTPUT_PATHS[version]
    
    # Get row count before saving
    row_count_before = df.count()
    
    # Check if DEP_DELAY exists
    if "DEP_DELAY" not in df.columns:
        raise ValueError("Critical column 'DEP_DELAY' missing before save!")
    
    # Drop dep_delay if it exists (we only need DEP_DELAY - flight_lineage_features.py handles both)
    df_to_save = df
    if "dep_delay" in df.columns:
        df_to_save = df.drop("dep_delay")
        if verbose:
            print(f"  Note: Dropped 'dep_delay' - only keeping 'DEP_DELAY' (flight_lineage_features.py handles both)")
    
    try:
        df_to_save.write.mode(WRITE_MODE).parquet(output_path)
        if verbose:
            print(f"  ✓ Write operation completed")
        
        # Verify the file was saved correctly by reading it back
        if verbose:
            print(f"  Verifying saved file...")
        df_verify = spark.read.parquet(output_path)
        row_count_after = df_verify.count()
        col_count_after = len(df_verify.columns)
        
        if verbose:
            print(f"    ✓ File exists and is readable")
            print(f"    ✓ Row count: {row_count_after:,} (expected: {row_count_before:,})")
            print(f"    ✓ Column count: {col_count_after} (expected: {len(df_to_save.columns)})")
        
        # Validate row count matches
        if row_count_after != row_count_before:
            raise RuntimeError(f"Row count mismatch! Before: {row_count_before:,}, After: {row_count_after:,}")
        
        # Validate column count matches
        if col_count_after != len(df_to_save.columns):
            print(f"  ⚠ WARNING: Column count mismatch! Expected: {len(df_to_save.columns)}, Got: {col_count_after}")
            print(f"    This may be OK if columns were merged/optimized during write")
        
        # Validate DEP_DELAY exists (we don't need dep_delay - flight_lineage_features.py handles both)
        if "DEP_DELAY" not in df_verify.columns:
            raise RuntimeError("DEP_DELAY missing in saved file!")
        
        # Verify critical columns exist in saved file
        # Note: We only check for DEP_DELAY (not dep_delay) since flight_lineage_features.py handles both
        critical_cols = ["fl_date", "DEP_DELAY", "origin", "dest", "op_carrier", "tail_num"]
        missing_in_saved = [c for c in critical_cols if c not in df_verify.columns]
        if missing_in_saved:
            raise RuntimeError(f"Critical columns missing in saved file: {missing_in_saved}")
        
        if verbose:
            print(f"    ✓ DEP_DELAY present in saved file")
            print(f"    ✓ All critical columns present")
        
        if verbose:
            print(f"  ✅ Saved and verified successfully!")
            print(f"    Note: Only 'DEP_DELAY' (uppercase) is saved - flight_lineage_features.py handles both cases")
            
    except Exception as e:
        raise RuntimeError(f"Failed to save {version} data to {output_path}: {e}")


# -------------------------
# MAIN PROCESSING
# -------------------------

def main():
    """Main processing function."""
    spark = SparkSession.builder.getOrCreate()
    
    overall_start_time = datetime.now()
    
    print("="*80)
    print("OTPW DATA PROCESSING")
    print("="*80)
    print(f"Versions to process: {VERSIONS}")
    print(f"Date filter (60M): {DATE_FILTER_START} to {DATE_FILTER_END}")
    print(f"Output format: Custom Join compatible")
    print(f"Start time: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    successful_versions = []
    failed_versions = []
    
    for version in VERSIONS:
        version_start_time = datetime.now()
        try:
            print(f"\n{'#'*80}")
            print(f"# Processing {version}")
            print(f"{'#'*80}")
            
            # Process OTPW data
            df_processed = process_otpw_version(version, spark)
            
            # Save to Custom Join format
            save_processed_data(df_processed, version, spark, verbose=VERBOSE)
            
            version_end_time = datetime.now()
            version_duration = (version_end_time - version_start_time).total_seconds()
            
            print(f"\n{'='*80}")
            print(f"✅ SUCCESS: {version} processed and saved")
            print(f"{'='*80}")
            print(f"  Duration: {version_duration:.1f}s ({version_duration/60:.1f}min)")
            print(f"  Output: {OUTPUT_PATHS[version]}")
            print(f"  Rows: {df_processed.count():,}")
            print(f"  Columns: {len(df_processed.columns)}")
            
            successful_versions.append(version)
            
        except Exception as e:
            version_end_time = datetime.now()
            version_duration = (version_end_time - version_start_time).total_seconds()
            
            print(f"\n{'='*80}")
            print(f"❌ ERROR processing {version}")
            print(f"{'='*80}")
            print(f"  Duration before failure: {version_duration:.1f}s")
            print(f"  Error: {e}")
            print(f"\n  Full traceback:")
            import traceback
            traceback.print_exc()
            
            failed_versions.append(version)
            # Continue with next version
            continue
    
    overall_end_time = datetime.now()
    overall_duration = (overall_end_time - overall_start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total duration: {overall_duration:.1f}s ({overall_duration/60:.1f}min)")
    print(f"Start time: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {overall_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Successful: {len(successful_versions)}/{len(VERSIONS)}")
    if successful_versions:
        print(f"   Versions: {', '.join(successful_versions)}")
    print(f"\n❌ Failed: {len(failed_versions)}/{len(VERSIONS)}")
    if failed_versions:
        print(f"   Versions: {', '.join(failed_versions)}")
    
    if successful_versions:
        print(f"\n{'='*80}")
        print("OUTPUT FILES (ready for split.py)")
        print(f"{'='*80}")
        for version in successful_versions:
            if version in OUTPUT_PATHS:
                output_path = OUTPUT_PATHS[version]
                # Try to verify file exists and get row count
                try:
                    df_check = spark.read.parquet(output_path)
                    row_count = df_check.count()
                    col_count = len(df_check.columns)
                    print(f"  ✅ {version}:")
                    print(f"     Path: {output_path}")
                    print(f"     Rows: {row_count:,}")
                    print(f"     Columns: {col_count}")
                except Exception as e:
                    print(f"  ⚠ {version}: {output_path}")
                    print(f"     ⚠ Could not verify file: {e}")
    
    if len(successful_versions) == len(VERSIONS):
        print(f"\n{'='*80}")
        print("✅ ALL VERSIONS PROCESSED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nNext steps:")
        print(f"  1. ✅ Output files verified and ready")
        print(f"  2. Run split.py with SOURCE='OTPW' (or modify split.py to use OTPW paths)")
        print(f"  3. split.py will read from these paths and create folds")
    else:
        print(f"\n{'='*80}")
        print("⚠ SOME VERSIONS FAILED")
        print(f"{'='*80}")
        print(f"  Please review errors above and fix issues before proceeding")


if __name__ == "__main__":
    main()

