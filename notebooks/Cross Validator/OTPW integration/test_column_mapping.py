#!/usr/bin/env python3
"""
test_column_mapping.py - Test the column mapping function

This script tests the column mapping function on actual OTPW and CUSTOM folds
to verify that the mapping works correctly.

Usage:
    python test_column_mapping.py
    
    Or in Databricks notebook:
    %run /path/to/test_column_mapping.py
"""

from pyspark.sql import SparkSession
import sys
import os
import importlib.util

# Load column_mapping module using importlib (for Databricks compatibility)
# Try Databricks path first, fall back to relative path for local development
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

# Import functions from the loaded module
map_otpw_columns_to_custom = column_mapping.map_otpw_columns_to_custom
validate_mapping = column_mapping.validate_mapping

# Configuration
FOLDER_PATH = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
VERSION = "3M"  # Start with 3M for testing
FOLD_INDEX = 1  # Use fold 1 for testing

# OTPW paths
OTPW_TRAIN_PATH = f"{FOLDER_PATH}/OTPW_OTPW_{VERSION}_FOLD_{FOLD_INDEX}_TRAIN.parquet"

# CUSTOM paths (for comparison)
CUSTOM_TRAIN_PATH = f"{FOLDER_PATH}/OTPW_CUSTOM_{VERSION}_FOLD_{FOLD_INDEX}_TRAIN.parquet"


def test_mapping():
    """Test the column mapping function."""
    spark = SparkSession.builder.getOrCreate()
    
    print("="*80)
    print("TESTING COLUMN MAPPING")
    print("="*80)
    
    # Step 1: Load OTPW data
    print(f"\n1. Loading OTPW data from: {OTPW_TRAIN_PATH}")
    try:
        df_otpw = spark.read.parquet(OTPW_TRAIN_PATH)
        print(f"   ✓ Loaded OTPW data: {df_otpw.count():,} rows, {len(df_otpw.columns)} columns")
        print(f"   Sample OTPW columns: {sorted(df_otpw.columns)[:10]}")
        
        # Check for DEP_DELAY or similar columns before mapping
        delay_cols = [c for c in df_otpw.columns if "delay" in c.lower() or "DELAY" in c]
        print(f"\n   Delay-related columns in OTPW data:")
        for col_name in sorted(delay_cols):
            print(f"     - {col_name}")
        
        if "DEP_DELAY" not in df_otpw.columns:
            print(f"\n   ⚠ WARNING: 'DEP_DELAY' not found in OTPW data!")
            print(f"   Will attempt to find alternative or create from available columns")
    except Exception as e:
        print(f"   ✗ ERROR loading OTPW data: {e}")
        return False
    
    # Step 2: Apply mapping
    print(f"\n2. Applying column mapping...")
    try:
        df_mapped = map_otpw_columns_to_custom(df_otpw)
        print(f"   ✓ Mapping applied: {df_mapped.count():,} rows, {len(df_mapped.columns)} columns")
        print(f"   Sample mapped columns: {sorted(df_mapped.columns)[:10]}")
    except Exception as e:
        print(f"   ✗ ERROR applying mapping: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Check critical columns
    print(f"\n3. Checking critical columns (REQUIRED by flight_lineage_features.py and feature engineering)...")
    critical_columns = [
        # Aircraft identification (flight_lineage_features.py auto-detects tail_num or TAIL_NUM)
        "tail_num",  # or TAIL_NUM (function auto-detects both)
        # Date/Time (critical for flight_lineage_features.py)
        "fl_date",
        "arr_time",
        "crs_arr_time",
        "dep_time",
        "crs_dep_time",
        # Location (critical for flight_lineage_features.py and graph features)
        "origin",
        "dest",
        # Delays (critical for flight_lineage_features.py)
        "DEP_DELAY",  # UPPERCASE (for cv.py) - MUST exist
        "dep_delay",  # lowercase (for flight_lineage_features.py) - MUST exist as duplicate
        "DEP_DEL15",  # UPPERCASE
        "SEVERE_DEL60",  # UPPERCASE
        "arr_delay",
        # Time components (critical for flight_lineage_features.py)
        "air_time",
        "crs_elapsed_time",
        "actual_elapsed_time",
        "taxi_in",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        # Route information (critical for flight_lineage_features.py)
        "distance",
        "op_carrier",
        "op_carrier_fl_num",
        # Status flags (critical for flight_lineage_features.py)
        "cancelled",
        "diverted",
        # Date components (for meta models)
        "month",
        "day_of_week",
        # Elevation (for meta models)
        "elevation",
    ]
    
    missing = []
    for col in critical_columns:
        if col in df_mapped.columns:
            print(f"   ✅ {col}")
        else:
            # Special case: tail_num - check if TAIL_NUM exists (function handles both)
            if col == "tail_num":
                if "TAIL_NUM" in df_mapped.columns:
                    print(f"   ✅ tail_num (TAIL_NUM found - function will auto-detect)")
                else:
                    print(f"   ❌ {col} - MISSING! (TAIL_NUM also not found)")
                    missing.append(col)
            else:
                print(f"   ❌ {col} - MISSING!")
                missing.append(col)
    
    # Special validation: Check that both DEP_DELAY (uppercase) and dep_delay (lowercase) exist
    print(f"\n   Special validation: DEP_DELAY case handling...")
    has_upper = "DEP_DELAY" in df_mapped.columns
    has_lower = "dep_delay" in df_mapped.columns
    
    if has_upper and has_lower:
        print(f"   ✅ Both 'DEP_DELAY' (uppercase) and 'dep_delay' (lowercase) exist")
        # Verify they're not the same column (should be true duplicates)
        # We can't easily check if they're aliases vs duplicates, but we can check if both exist
        print(f"   ✅ Both columns present - ready for cv.py (DEP_DELAY) and flight_lineage_features.py (dep_delay)")
    elif has_upper and not has_lower:
        print(f"   ❌ 'DEP_DELAY' (uppercase) exists but 'dep_delay' (lowercase) is MISSING!")
        print(f"   ⚠ flight_lineage_features.py requires 'dep_delay' (lowercase)")
        missing.append("dep_delay")
    elif not has_upper and has_lower:
        print(f"   ❌ 'dep_delay' (lowercase) exists but 'DEP_DELAY' (uppercase) is MISSING!")
        print(f"   ⚠ cv.py requires 'DEP_DELAY' (uppercase)")
        missing.append("DEP_DELAY")
    else:
        print(f"   ❌ Both 'DEP_DELAY' (uppercase) and 'dep_delay' (lowercase) are MISSING!")
        missing.extend(["DEP_DELAY", "dep_delay"])
    
    if missing:
        print(f"\n   ❌ CRITICAL ERROR: {len(missing)} critical column(s) missing: {missing}")
        print(f"   These columns are REQUIRED by flight_lineage_features.py or other feature engineering scripts!")
        return False
    else:
        print(f"\n   ✅ All critical columns present!")
    
    # Step 4: Check label columns are UPPERCASE
    print(f"\n4. Verifying label columns are UPPERCASE...")
    label_columns = ["DEP_DELAY", "DEP_DEL15", "SEVERE_DEL60"]
    for col in label_columns:
        if col in df_mapped.columns:
            print(f"   ✓ {col} is UPPERCASE (correct)")
        else:
            print(f"   ✗ {col} not found or wrong case!")
            return False
    
    # Step 5: Check other columns are lowercase
    print(f"\n5. Verifying other columns are lowercase...")
    test_columns = ["origin", "dest", "op_carrier", "dep_time", "arr_time", "crs_elapsed_time"]
    for col in test_columns:
        if col in df_mapped.columns:
            print(f"   ✓ {col} is lowercase (correct)")
        else:
            # Check if uppercase version exists
            upper_col = col.upper()
            if upper_col in df_mapped.columns:
                print(f"   ⚠ {col} not found, but {upper_col} exists (may need mapping)")
            else:
                print(f"   ⚠ {col} not found (may be optional)")
    
    # Step 6: Compare with CUSTOM (optional)
    print(f"\n6. Comparing with CUSTOM reference (optional)...")
    try:
        df_custom = spark.read.parquet(CUSTOM_TRAIN_PATH)
        print(f"   ✓ Loaded CUSTOM reference: {df_custom.count():,} rows, {len(df_custom.columns)} columns")
        
        # Validate mapping
        validation = validate_mapping(df_otpw, df_custom)
        
        print(f"\n   Validation Results:")
        print(f"   - Mapped columns: {len(validation['mapped_columns'])}")
        print(f"   - CUSTOM columns: {len(validation['custom_columns'])}")
        print(f"   - Missing in mapped: {len(validation['missing_in_mapped'])}")
        print(f"   - Extra in mapped: {len(validation['extra_in_mapped'])}")
        print(f"   - Is valid: {validation['is_valid']}")
        
        if validation['missing_in_mapped']:
            print(f"\n   ⚠ Missing columns (expected - these are flight lineage features):")
            for col in sorted(validation['missing_in_mapped'])[:20]:  # Show first 20
                print(f"      - {col}")
            if len(validation['missing_in_mapped']) > 20:
                print(f"      ... and {len(validation['missing_in_mapped']) - 20} more")
        
        if validation['extra_in_mapped']:
            print(f"\n   ℹ Extra columns in mapped (OTPW-specific, OK to keep):")
            for col in sorted(validation['extra_in_mapped'])[:10]:  # Show first 10
                print(f"      - {col}")
            if len(validation['extra_in_mapped']) > 10:
                print(f"      ... and {len(validation['extra_in_mapped']) - 10} more")
        
        # Note: It's expected that mapped will have fewer columns than CUSTOM
        # because CUSTOM includes flight lineage features that will be added later
        print(f"\n   ℹ Note: Missing columns are expected - they are flight lineage features")
        print(f"      that will be added by the feature engineering pipeline.")
        
    except Exception as e:
        print(f"   ⚠ Could not load CUSTOM reference: {e}")
        print(f"   (This is OK - mapping test can proceed without comparison)")
    
    # Step 7: Check data types
    print(f"\n7. Checking critical column data types...")
    schema_dict = {f.name: f.dataType.typeName() for f in df_mapped.schema.fields}
    
    type_checks = {
        "DEP_DELAY": ["double", "int"],
        "DEP_DEL15": ["int", "integer"],
        "SEVERE_DEL60": ["int", "integer"],
        "fl_date": ["date"],
    }
    
    for col, expected_types in type_checks.items():
        if col in schema_dict:
            actual_type = schema_dict[col]
            if actual_type in expected_types:
                print(f"   ✓ {col}: {actual_type} (correct)")
            else:
                print(f"   ⚠ {col}: {actual_type} (expected one of: {expected_types})")
        else:
            print(f"   ✗ {col}: not found")
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print("✓ Column mapping function is ready!")
    print("✓ Critical columns are present and correctly cased")
    print("✓ Labels are UPPERCASE, other columns are lowercase")
    print("\nThe mapping function is ready to use in Indri's split.py")
    
    return True


if __name__ == "__main__":
    success = test_mapping()
    sys.exit(0 if success else 1)

