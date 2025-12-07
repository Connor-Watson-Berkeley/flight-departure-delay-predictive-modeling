# Sanity Check Results - Custom Join Refined

## Executive Summary

Analysis of the Custom Join Refined notebook execution reveals critical findings about data loss and data quality issues in the pipeline.

## Row Count Summary

| Step | Row Count | Change | Change % |
|------|-----------|--------|----------|
| **Initial Load** | 2,806,942 | - | - |
| **After dropDuplicates** | 1,403,471 | -1,403,471 | **-50.00%** |
| **After Airport Join** | 1,403,471 | 0 | 0.00% |
| **After Station Join (Left)** | 1,403,471 | 0 | 0.00% |
| **After Station Filter** | 1,402,397 | -1,074 | -0.08% |
| **After Weather Join** | **2,066,856** | **+664,459** | **+47.37%** ⚠️ |

**Total Loss from Initial**: 740,086 flights (26.36% loss)
**Final Retention Rate**: 73.64%

## Critical Findings

### 1. MASSIVE DROP at dropDuplicates (50% Loss) ⚠️ CRITICAL

**Location**: After dropDuplicates step

**Impact**:
- **Loss**: 1,403,471 flights (exactly 50.00%)
- **From**: 2,806,942 → 1,403,471

**Analysis**:
- The exact 50% suggests either:
  1. **True duplicates** - Same flight appearing twice in source data
  2. **Deduplication logic issue** - Removing legitimate unique flights
  3. **Data quality issue** - Source dataset has intentional duplicate entries

**Recommendation**: **URGENT INVESTIGATION REQUIRED**
- Verify if duplicates are truly duplicates or represent separate records
- Check deduplication logic - consider using key-based deduplication
- Review source data quality
- This is the **primary source** of data loss

### 2. WEATHER JOIN CREATING DUPLICATES ⚠️ CRITICAL

**Location**: After Weather Join step

**Impact**:
- **Row count INCREASES**: 1,402,397 → 2,066,856
- **Gain**: +664,459 rows (+47.37%)
- This is **NOT data loss, but data duplication**

**Root Cause**:
- Weather join is likely a **one-to-many join**
- Multiple weather records per station/time combination
- Each flight is being matched to multiple weather records

**Analysis**:
- This suggests the weather data has:
  - Multiple records per station per time period
  - Or the join key is not unique enough
  - Or the join logic is creating cartesian products

**Recommendation**: **URGENT FIX REQUIRED**
- Review weather join logic
- Ensure weather data is deduplicated before join
- Use proper join keys to prevent one-to-many relationships
- Consider aggregating weather data before joining
- This is creating **artificial data inflation** which will affect model training

### 3. Station Filter Drop (Expected)

**Location**: After Station Filter step

**Impact**:
- **Loss**: 1,074 flights (0.08%)
- **From**: 1,403,471 → 1,402,397

**Analysis**:
- This is **expected behavior** - flights from airports without weather stations are filtered out
- 4 airports without stations: PSE, PPG, GUM, ISN
- This is a **small, acceptable loss** compared to other issues

**Status**: ✅ Expected and acceptable

### 4. Airport Join (No Loss)

**Location**: After Airport Join step

**Impact**:
- **No row loss** - LEFT JOIN preserves all flights
- 765 flights (0.05%) have NULL latitude/longitude
- These are likely airports not in the airport codes dataset

**Status**: ✅ Working as expected

### 5. Station Join (No Loss)

**Location**: After Station Join (Left) step

**Impact**:
- **No row loss** - LEFT JOIN preserves all flights
- 1,074 flights (0.08%) have NULL station_id
- These are from 4 airports without stations (PSE, PPG, GUM, ISN)

**Status**: ✅ Working as expected

## Data Quality Issues

### Missing Airport Data
- **765 flights** (0.05%) missing origin latitude/longitude
- These flights have airports not in the airport codes dataset

### Missing Station Data
- **1,074 flights** (0.08%) from 4 airports without weather stations:
  - PSE (Ponce, Puerto Rico)
  - PPG (Pago Pago, American Samoa)
  - GUM (Guam)
  - ISN (Williston, North Dakota)

## Recommendations

### Priority 1: CRITICAL - Fix dropDuplicates (50% Loss)

1. **Investigate Duplicates**:
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

### Priority 2: CRITICAL - Fix Weather Join Duplication

1. **Review Weather Join Logic**:
   - Check if weather data is deduplicated before join
   - Verify join keys are unique
   - Ensure one-to-one relationship between flights and weather

2. **Deduplicate Weather Data**:
   ```python
   # Before joining, ensure weather data is unique per station/time
   df_weather_dedup = df_weather.dropDuplicates(subset=['station_id', 'timestamp'])
   ```

3. **Use Proper Join Keys**:
   - Ensure join creates one-to-one relationship
   - Consider time window matching instead of exact match
   - Aggregate weather data if multiple records exist per time period

4. **Add Deduplication After Weather Join**:
   ```python
   # If weather join creates duplicates, deduplicate after join
   df_joined = df_joined.dropDuplicates(subset=['flight_key_columns'])
   ```

### Priority 3: Handle Missing Data

1. **Missing Airport Coordinates**:
   - Consider imputing with airport lookup
   - Or exclude flights from unknown airports if acceptable

2. **Missing Weather Stations**:
   - Consider using nearest station for airports without stations
   - Or exclude these flights if acceptable (currently 0.08% loss)

## Expected vs Actual Behavior

| Step | Expected Behavior | Actual Behavior | Status |
|------|------------------|-----------------|--------|
| dropDuplicates | Remove true duplicates only | Removes 50% of flights | ❌ **ISSUE** |
| Airport Join | LEFT JOIN - no loss | No loss | ✅ OK |
| Station Join | LEFT JOIN - no loss | No loss | ✅ OK |
| Station Filter | Filter NULL stations | Drops 0.08% | ✅ OK |
| Weather Join | One-to-one join | Creates 47% duplicates | ❌ **ISSUE** |

## Next Steps

1. **IMMEDIATE**: Investigate dropDuplicates - why is it removing 50% of flights?
2. **IMMEDIATE**: Fix weather join to prevent duplication
3. **HIGH PRIORITY**: Re-run pipeline after fixes
4. **ANALYSIS**: Verify final dataset quality after fixes
5. **DOCUMENTATION**: Update pipeline documentation with findings

## Conclusion

The pipeline has two critical issues:
1. **50% data loss at dropDuplicates** - needs urgent investigation
2. **47% data duplication at weather join** - needs urgent fix

These issues must be resolved before using the dataset for model training, as they will significantly impact model performance and validity.

