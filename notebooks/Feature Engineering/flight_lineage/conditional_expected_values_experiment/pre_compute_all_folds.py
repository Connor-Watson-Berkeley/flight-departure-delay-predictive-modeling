"""
precompute_all_folds.py

Batch pre-computation script for all folds and versions.

This script generates conditional expected values for all folds across all versions:
- 3M: folds 0, 1, 2, 3 (4 folds)
- 12M: folds 0, 1, 2, 3 (4 folds)
- 60M: folds 0, 1, 2, 3 (4 folds)

Total: 12 fold pre-computations

Usage in Databricks Notebook:
    1. Load the module:
       %run /Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/flight_lineage/precompute_all_folds
    
    2. Modify configuration if needed (VERSIONS, FOLD_INDICES, etc.)
    
    3. Run the cell - it will automatically execute all pre-computations
"""

from datetime import datetime
import time

# Load modules from Databricks repo (matching pattern from Demo notebook)
import importlib.util

# Load precompute_conditional_expected_values module
precompute_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering/flight_lineage/precompute_conditional_expected_values.py"
spec = importlib.util.spec_from_file_location("precompute_conditional_expected_values", precompute_path)
precompute_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute_module)
precompute_conditional_expected_values = precompute_module.precompute_conditional_expected_values

# Configuration
# RECOMMENDED: Start with 3M only to validate, then run 12M and 60M
# VERSIONS = ["3M"]  # Start with 3M (~53 minutes for all 4 folds)
VERSIONS = ["3M"] # NEXT: ["12M", "60M"]  # All versions (~20-30 hours total)

FOLD_INDICES = [0, 1, 2, 3]  # 0-based fold indices (will be converted to 1-based for files)
# Note: Fold 0 already completed, so you can skip it: FOLD_INDICES = [1, 2, 3]

LOOKUP_BASE_PATH = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/conditional_expected_values"
USE_PROPHET = True
ADD_LINEAGE_FEATURES = True
VERBOSE = True

# Execution Time Estimates:
# - 3M per fold: ~13 minutes → Total for 4 folds: ~53 minutes
# - 12M per fold: ~60-90 minutes → Total for 4 folds: ~4-6 hours
# - 60M per fold: ~4-6 hours → Total for 4 folds: ~16-24 hours
# Total if running all: ~20-30 hours (can parallelize across folds/versions)

# Execution tracking
RESULTS = []


def precompute_all_folds(
    versions=None,
    fold_indices=None,
    lookup_base_path=None,
    use_prophet=True,
    add_lineage_features=True,
    verbose=True
):
    """
    Pre-compute conditional expected values for all specified folds and versions.
    
    Parameters:
    -----------
    versions : list of str
        Versions to pre-compute: ["3M", "12M", "60M"]
    fold_indices : list of int
        Fold indices to pre-compute: [0, 1, 2, 3] (0-based)
    lookup_base_path : str
        Base path for storing conditional expected values
    use_prophet : bool
        Whether to use Prophet for temporal conditional means
    add_lineage_features : bool
        Whether to add lineage features before computation
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    list : List of results dictionaries (one per fold/version combination)
    """
    if versions is None:
        versions = VERSIONS
    if fold_indices is None:
        fold_indices = FOLD_INDICES
    if lookup_base_path is None:
        lookup_base_path = LOOKUP_BASE_PATH
    
    total_combinations = len(versions) * len(fold_indices)
    current = 0
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    if verbose:
        print("=" * 80)
        print("BATCH PRE-COMPUTATION: ALL FOLDS AND VERSIONS")
        print("=" * 80)
        print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Versions: {versions}")
        print(f"Fold indices: {fold_indices} (0-based)")
        print(f"Total combinations: {total_combinations}")
        print(f"Lookup base path: {lookup_base_path}")
        print("=" * 80)
        print()
    
    results = []
    
    for version in versions:
        if verbose:
            print("\n" + "=" * 80)
            print(f"PROCESSING VERSION: {version}")
            print("=" * 80)
        
        for fold_index in fold_indices:
            current += 1
            
            if verbose:
                print(f"\n[{current}/{total_combinations}] {version} - Fold {fold_index} (fold_{fold_index+1})")
                print("-" * 80)
            
            try:
                fold_start_time = time.time()
                
                result = precompute_conditional_expected_values(
                    version=version,
                    fold_index=fold_index,
                    lookup_base_path=lookup_base_path,
                    use_prophet=use_prophet,
                    add_lineage_features=add_lineage_features,
                    verbose=verbose
                )
                
                fold_elapsed = time.time() - fold_start_time
                result['execution_time_seconds'] = fold_elapsed
                result['version'] = version
                result['fold_index'] = fold_index
                results.append(result)
                
                if verbose:
                    print(f"\n✓ Completed {version}/fold_{fold_index+1} in {fold_elapsed:.2f} seconds ({fold_elapsed/60:.2f} minutes)")
                    
            except Exception as e:
                error_msg = f"ERROR processing {version}/fold_{fold_index+1}: {str(e)}"
                if verbose:
                    print(f"\n❌ {error_msg}")
                results.append({
                    'version': version,
                    'fold_index': fold_index,
                    'status': 'error',
                    'error': str(e)
                })
    
    total_elapsed = time.time() - start_time
    end_datetime = datetime.now()
    
    if verbose:
        print("\n" + "=" * 80)
        print("BATCH PRE-COMPUTATION COMPLETE")
        print("=" * 80)
        print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total elapsed time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes, {total_elapsed/3600:.2f} hours)")
        print(f"\nCompleted: {len([r for r in results if 'status' not in r or r.get('status') != 'error'])}/{total_combinations}")
        print(f"Errors: {len([r for r in results if r.get('status') == 'error'])}/{total_combinations}")
        
        # Summary by version
        print("\nSummary by version:")
        for version in versions:
            version_results = [r for r in results if r.get('version') == version]
            version_completed = [r for r in version_results if 'status' not in r or r.get('status') != 'error']
            version_errors = [r for r in version_results if r.get('status') == 'error']
            
            if version_completed:
                avg_time = sum(r.get('execution_time_seconds', 0) for r in version_completed) / len(version_completed)
                print(f"  {version}: {len(version_completed)}/{len(version_results)} folds, avg time: {avg_time:.2f}s ({avg_time/60:.2f} min)")
            if version_errors:
                print(f"    Errors: {len(version_errors)} folds")
        
        print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Execute batch pre-computation with configured parameters
    RESULTS = precompute_all_folds(
        versions=VERSIONS,
        fold_indices=FOLD_INDICES,
        lookup_base_path=LOOKUP_BASE_PATH,
        use_prophet=USE_PROPHET,
        add_lineage_features=ADD_LINEAGE_FEATURES,
        verbose=VERBOSE
    )
    
    print(f"\n✓ Pre-computation complete! Processed {len(RESULTS)} fold/version combinations.")

