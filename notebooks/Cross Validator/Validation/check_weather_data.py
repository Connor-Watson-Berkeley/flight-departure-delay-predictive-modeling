"""
Check weather column nulls in fold 3 validation data.
Usage: Run in Databricks notebook. If you have cv_obj loaded, use Option 1. Otherwise use Option 2.
"""

import importlib.util
from pyspark.sql import functions as F

# Weather columns to check
WEATHER_COLS = [
    'hourlyprecipitation', 'hourlysealevelpressure', 'hourlyaltimetersetting',
    'hourlywetbulbtemperature', 'hourlystationpressure', 'hourlywinddirection',
    'hourlyrelativehumidity', 'hourlywindspeed', 'hourlydewpointtemperature',
    'hourlydrybulbtemperature', 'hourlyvisibility', 'elevation'
]

# ============================================================================
# Load data - try to use existing data first, then load if needed
# ============================================================================
VERSION = "60M"  # Change to "3M" or "12M" as needed

df = None

# Try to use cv_obj if it exists (already loaded with permissions)
try:
    _, df = cv_obj.folds[-2]  # Fold 3 validation data
    print(f"✓ Using data from cv_obj (version: {cv_obj.version if hasattr(cv_obj, 'version') else 'unknown'})")
except NameError:
    # Try to use existing data_loader if it exists
    try:
        folds = data_loader.get_version(VERSION)
        _, df = folds[2]  # Fold 3 is at index 2
        print(f"✓ Using data from existing data_loader (version: {VERSION})")
    except NameError:
        # Need to load data - this may have permission issues
        print(f"📥 Loading folds for {VERSION}...")
        print("⚠️  Note: This requires file read permissions. If you get permission errors,")
        print("   try using cv_obj or data_loader that's already loaded in your notebook.")
        
        # Import cv module
        cv_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/cv.py"
        spec = importlib.util.spec_from_file_location("cv", cv_path)
        cv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv)
        
        # Load folds
        loader = cv.FlightDelayDataLoader()
        loader.load()
        folds = loader.get_version(VERSION)
        _, df = folds[2]  # Fold 3 is at index 2
        print(f"✓ Loaded fold 3 validation data")

if df is None:
    raise ValueError("Could not load data. Please ensure cv_obj or data_loader is available, or check permissions.")

# ============================================================================
# ANALYSIS (works with either option above)
# ============================================================================

print("=" * 80)
print("Weather Column Null Analysis")
print("=" * 80)

total = df.count()
print(f"\nTotal rows: {total:,}")

# Check date range
date_info = df.select(
    F.min("FL_DATE").alias("min_date"),
    F.max("FL_DATE").alias("max_date"),
    F.countDistinct("FL_DATE").alias("unique_dates")
).first()
print(f"Date range: {date_info['min_date']} to {date_info['max_date']} ({date_info['unique_dates']} unique dates)")

# Check nulls
print(f"\n{'Column':<30} {'Null Count':<15} {'Total':<15} {'% Null':<10} {'Status'}")
print("-" * 80)

all_null_cols = []
for col in WEATHER_COLS:
    if col in df.columns:
        nulls = df.filter(F.col(col).isNull() | F.isnan(F.col(col))).count()
        pct = (nulls / total) * 100
        if nulls == total:
            status = "❌ ALL NULL"
            all_null_cols.append(col)
        elif nulls > 0:
            status = f"⚠️  {pct:.1f}% null"
        else:
            status = "✓ OK"
        print(f"{col:<30} {nulls:>10,} {total:>10,} {pct:>6.2f}% {status}")
    else:
        print(f"{col:<30} {'MISSING COLUMN':>30}")

# Summary
print("\n" + "=" * 80)
if all_null_cols:
    print(f"❌ CRITICAL: {len(all_null_cols)} columns are 100% null:")
    for col in all_null_cols:
        print(f"   - {col}")
    print("\nThis will cause Imputer to fail!")
    print("Action: Check weather data join for this date range.")
else:
    print("✓ All weather columns have at least some non-null values")

print("=" * 80)

