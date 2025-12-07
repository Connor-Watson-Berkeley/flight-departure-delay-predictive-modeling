# Sanity Check Findings - Custom Join Refined

## Summary

Initial analysis of the Custom Join Refined notebook execution reveals critical findings about flight data loss.

## Key Findings

### 1. MASSIVE DROP at dropDuplicates Step (CRITICAL)

**Location**: After dropDuplicates

**Row Counts**:
- **Initial Load**: 2,806,942 flights
- **After dropDuplicates**: 1,403,471 flights
- **Loss**: 1,403,471 flights (exactly 50.00%)

**Analysis**:
- The `dropDuplicates()` operation is removing exactly **half** of all flights
- This is a **50% data loss** at this single step
- The exact 50% suggests either:
  1. True duplicates (same flight appearing twice in source data)
  2. Deduplication logic issue (removing legitimate unique flights)
  3. Data quality issue in source dataset

**Recommendation**:
- **URGENT**: Investigate why dropDuplicates is removing 50% of flights
- Check if duplicates are truly duplicates or represent separate records
- Consider using `dropDuplicates(subset=['key_columns'])` with specific identifying columns
- Verify source data quality - are there intentional duplicate entries?
- This is likely the **primary source** of the ~20% overall loss (if combined with other drops)

### 2. Airport-Timezones File Read Error (BLOCKING)

**Location**: Airport Join step

**Error**: 
```
Error while reading file dbfs:/tmp/airport-timezones.csv
RemoteFileChangedException: The file might have been updated during query execution
```

**Impact**:
- Prevents Airport Join from completing
- Blocks all subsequent sanity checks
- Cannot proceed with station matching or weather join analysis

**Root Cause**:
- File is read from temporary location `/tmp/airport-timezones.csv`
- File may be modified/deleted during execution
- DataFrame is not cached, causing re-reads that fail

**Fix Applied**:
- Added `.cache()` to `df_airport_timezones` after reading
- Added `.count()` to materialize cache immediately
- This prevents re-reading the file during joins

**Status**: Fixed in notebook

### 3. Duplicate Sanity Check Cells

**Observation**: 
- "After dropDuplicates" sanity check appears twice in output
- This suggests duplicate cells in the notebook

**Impact**: Minor - just redundant output

**Recommendation**: Remove duplicate sanity check cells

## Data Loss Summary

Based on current findings:

| Step | Row Count | Loss | Loss % |
|------|-----------|------|--------|
| Initial Load | 2,806,942 | - | - |
| After dropDuplicates | 1,403,471 | -1,403,471 | -50.00% |
| After Airport Join | ERROR | - | - |
| After Station Join | N/A | - | - |
| After Station Filter | N/A | - | - |
| After Weather Join | N/A | - | - |
| Final | N/A | - | - |

**Total Loss So Far**: 1,403,471 flights (50%) at dropDuplicates step alone

## Next Steps

1. **IMMEDIATE**: Fix airport-timezones caching (DONE)
2. **CRITICAL**: Investigate dropDuplicates - why is it removing 50% of flights?
3. **HIGH PRIORITY**: Re-run notebook after fix to get complete pipeline analysis
4. **ANALYSIS**: Once pipeline completes, identify all drop points:
   - Station filter (expected to drop flights without stations)
   - Weather join (may drop flights without weather data)
   - Any other filters or joins

## Recommendations

### For dropDuplicates Issue:

1. **Verify Duplicates**:
   ```python
   # Check if duplicates are truly duplicates
   duplicate_count = df_flights.count() - df_flights.dropDuplicates().count()
   print(f"True duplicates: {duplicate_count}")
   
   # Check what makes rows "duplicate"
   df_flights.groupBy(df_flights.columns).count().filter(col("count") > 1).show()
   ```

2. **Use Key-Based Deduplication**:
   ```python
   # Instead of dropDuplicates() on all columns
   # Use specific key columns that uniquely identify a flight
   key_cols = ['fl_date', 'op_carrier', 'op_carrier_fl_num', 'origin', 'dest', 'crs_dep_time']
   df_flights = df_flights.dropDuplicates(subset=key_cols)
   ```

3. **Preserve All Rows Initially**:
   - Consider NOT deduplicating at this stage
   - Let downstream processes handle duplicates if needed
   - Or use a more selective deduplication strategy

### For Complete Analysis:

1. Re-run notebook after airport-timezones fix
2. Collect row counts at each checkpoint
3. Identify all drop points
4. Quantify losses at each step
5. Determine if 50% loss at dropDuplicates is acceptable or needs fixing

## Expected Remaining Drops

Based on design document analysis, we expect additional drops at:
- **Station Filter**: Flights from airports without weather stations (estimated 5-20%)
- **Weather Join**: Flights without matching weather data (estimated 1-5%)

**Combined with 50% drop at deduplication**, this could explain the ~20% overall loss if:
- 50% drop at deduplication is intentional/correct
- Additional 10-15% drops at station/weather steps
- Final retention: ~40-45% of original

However, **50% loss at deduplication seems excessive** and should be investigated.

