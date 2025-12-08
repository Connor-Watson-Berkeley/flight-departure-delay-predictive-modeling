"""
precompute_conditional_expected_values.py

Pre-computation script for conditional expected values per training fold.

This script generates and saves conditional expected value lookup tables for a single fold.
It should be run once per training fold to pre-compute values that will be used during
cross-validation model training.

Usage in Databricks Notebook:
    1. Load the module:
       %run /Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/flight_lineage/precompute_conditional_expected_values
    
    2. Modify the configuration variables in the __main__ block (VERSION, FOLD_INDEX, etc.)
    
    3. Run the cell - it will automatically execute
    
    OR call the function directly:
    
    from precompute_conditional_expected_values import precompute_conditional_expected_values
    
    results = precompute_conditional_expected_values(
        version="3M",
        fold_index=0,  # 0-based (will be converted to 1-based for file paths)
        lookup_base_path="dbfs:/path/to/conditional_expected_values",
        use_prophet=True
    )
"""

import os
from datetime import datetime
from conditional_expected_values_experiment import generate_conditional_expected_values

# Load modules from Databricks repo
import importlib.util


def precompute_conditional_expected_values(
    version,
    fold_index,  # 0-based (0, 1, 2, 3)
    lookup_base_path,
    use_prophet=True,
    add_lineage_features=True,
    verbose=True
):
    """
    Pre-compute conditional expected values for a single training fold.
    
    Parameters:
    -----------
    version : str
        Data version: "3M", "12M", or "60M"
    fold_index : int
        Fold index (0-based: 0, 1, 2, 3). Will be converted to 1-based for file paths.
    lookup_base_path : str
        Base path where conditional expected values will be stored.
        Files will be saved to: {lookup_base_path}/{version}/fold_{fold_index_1based}/
    use_prophet : bool
        Whether to generate temporal conditional means using Prophet (default: True)
    add_lineage_features : bool
        Whether to add flight lineage features before computing conditional values (default: True)
        Required for turnover time conditional means
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    dict : Results dictionary with generated conditional expected values and file paths
    """
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.getOrCreate()
    
    # Convert 0-based fold_index to 1-based for file paths
    fold_index_1based = fold_index + 1
    fold_path = f"{lookup_base_path}/{version}/fold_{fold_index_1based}"
    
    if verbose:
        print("=" * 80)
        print(f"PRE-COMPUTING CONDITIONAL EXPECTED VALUES")
        print("=" * 80)
        print(f"Version: {version}")
        print(f"Fold Index: {fold_index} (0-based) → fold_{fold_index_1based} (1-based for files)")
        print(f"Output Path: {fold_path}")
        print(f"Use Prophet: {use_prophet}")
        print(f"Add Lineage Features: {add_lineage_features}")
        print("=" * 80)
        print()
    
    # Load cv module
    try:
        cv_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Cross Validator/cv.py"
        spec = importlib.util.spec_from_file_location("cv", cv_path)
        cv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv)
        
        if verbose:
            print("✓ Loaded cv module")
    except Exception as e:
        if verbose:
            print(f"✗ Could not load cv module: {str(e)}")
        raise
    
    # Load data loader
    data_loader = cv.FlightDelayDataLoader()
    data_loader.load()
    
    # Get folds for this version
    folds = data_loader.get_version(version)
    
    if not folds or len(folds) <= fold_index:
        raise ValueError(
            f"No fold at index {fold_index} for version {version}. "
            f"Available folds: {len(folds) if folds else 0}"
        )
    
    # Get training data for this fold
    train_df, val_df = folds[fold_index]
    
    if verbose:
        train_count = train_df.count()
        val_count = val_df.count()
        print(f"✓ Loaded fold {fold_index} (fold_{fold_index_1based}):")
        print(f"  Training: {train_count:,} rows")
        print(f"  Validation: {val_count:,} rows")
        print()
    
    # Add flight lineage features if needed
    if add_lineage_features:
        if verbose:
            print("Adding flight lineage features...")
        
        try:
            # Load flight_lineage_features module
            lineage_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/flight_lineage_features.py"
            spec = importlib.util.spec_from_file_location("flight_lineage_features", lineage_path)
            flight_lineage_features = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(flight_lineage_features)
            
            # Add lineage features
            train_df = flight_lineage_features.add_flight_lineage_features(train_df)
            
            if verbose:
                print("✓ Flight lineage features added")
                print()
        except Exception as e:
            if verbose:
                print(f"⚠ Warning: Could not add flight lineage features: {str(e)}")
                print("  Continuing anyway - turnover time conditional means may not be generated")
            # Continue anyway - air time conditional means don't require lineage features
    
    # Create output directory if it doesn't exist
    # Note: In Databricks, you typically don't need to explicitly create directories
    # But we can verify the path structure
    if verbose:
        print(f"Output directory: {fold_path}")
        print("(Directory will be created automatically when files are written)")
        print()
    
    # Generate conditional expected values
    if verbose:
        print("Generating conditional expected values...")
        print()
    
    results = generate_conditional_expected_values(
        train_df,
        output_path=fold_path,
        use_prophet=use_prophet,
        verbose=verbose,
        save_results=True
    )
    
    if verbose:
        print()
        print("=" * 80)
        print("✅ PRE-COMPUTATION COMPLETE!")
        print("=" * 80)
        print(f"Conditional expected values saved to: {fold_path}")
        print()
        print("Generated files:")
        for name, path in results.items():
            if isinstance(path, str) and path.endswith('.parquet'):
                print(f"  ✓ {name}: {path}")
    
    return results


if __name__ == "__main__":
    """
    Run pre-computation directly in Databricks notebook.
    
    Modify the parameters below and run the script in a notebook cell.
    """
    print("=" * 80)
    print("PRE-COMPUTE CONDITIONAL EXPECTED VALUES")
    print("=" * 80)
    print("\n📝 Configure parameters below, then run this cell in Databricks")
    print("=" * 80)
    
    # ============================================================================
    # CONFIGURATION - Modify these parameters as needed
    # ============================================================================
    
    VERSION = "3M"  # Options: "3M", "12M", "60M"
    FOLD_INDEX = 0  # 0-based index: 0, 1, 2, or 3 (will save to fold_1, fold_2, etc.)
    LOOKUP_BASE_PATH = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/conditional_expected_values"
    USE_PROPHET = True  # Set to False to skip Prophet (faster but less accurate)
    ADD_LINEAGE_FEATURES = True  # Set to False if lineage features already added
    VERBOSE = True  # Set to False for less output
    
    # ============================================================================
    # PRE-COMPUTATION - Run automatically when script is executed
    # ============================================================================
    
    try:
        print(f"\n🚀 Starting pre-computation...")
        print(f"   Version: {VERSION}")
        print(f"   Fold Index: {FOLD_INDEX} → fold_{FOLD_INDEX + 1}")
        print(f"   Output Path: {LOOKUP_BASE_PATH}")
        print(f"   Use Prophet: {USE_PROPHET}")
        print(f"   Add Lineage Features: {ADD_LINEAGE_FEATURES}")
        print()
        
        results = precompute_conditional_expected_values(
            version=VERSION,
            fold_index=FOLD_INDEX,
            lookup_base_path=LOOKUP_BASE_PATH,
            use_prophet=USE_PROPHET,
            add_lineage_features=ADD_LINEAGE_FEATURES,
            verbose=VERBOSE
        )
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Pre-computation complete!")
        print("=" * 80)
        print(f"\nConditional expected values saved to:")
        print(f"  {LOOKUP_BASE_PATH}/{VERSION}/fold_{FOLD_INDEX + 1}/")
        print("\nYou can now use these pre-computed values in your pipeline by setting:")
        print(f"  ConditionalExpectedValuesEstimator(lookup_base_path='{LOOKUP_BASE_PATH}')")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR DURING PRE-COMPUTATION")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check that VERSION is one of: '3M', '12M', '60M'")
        print("  2. Check that FOLD_INDEX is between 0 and 3")
        print("  3. Verify LOOKUP_BASE_PATH is accessible in Databricks")
        print("  4. Ensure cv.py and flight_lineage_features.py are accessible")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise

