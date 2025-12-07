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

### 1. dropDuplicates (50% Reduction) ✅ EXPECTED

**Location**: After dropDuplicates step

**Impact**:
- **Reduction**: 1,403,471 flights (exactly 50.00%)
- **From**: 2,806,942 → 1,403,471

**Analysis**:
- **WORKING AS INTENDED (WAI)**
- Source data contains **two rows per flight** (likely due to data structure or processing)
- The 50% reduction is expected and correct
- Deduplication is functioning properly

**Status**: ✅ No action needed - this is expected behavior

### 2. WEATHER JOIN CREATING DUPLICATES ⚠️ CRITICAL - PRIMARY ISSUE

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

### Priority 1: CRITICAL - Fix Weather Join Duplication

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

### Priority 2: Handle Missing Data

1. **Missing Airport Coordinates**:
   - Consider imputing with airport lookup
   - Or exclude flights from unknown airports if acceptable

2. **Missing Weather Stations**:
   - Consider using nearest station for airports without stations
   - Or exclude these flights if acceptable (currently 0.08% loss)

## Expected vs Actual Behavior

| Step | Expected Behavior | Actual Behavior | Status |
|------|------------------|-----------------|--------|
| dropDuplicates | Remove duplicates (2 rows per flight) | Removes 50% (expected) | ✅ **WAI** |
| Airport Join | LEFT JOIN - no loss | No loss | ✅ OK |
| Station Join | LEFT JOIN - no loss | No loss | ✅ OK |
| Station Filter | Filter NULL stations | Drops 0.08% | ✅ OK |
| Weather Join | One-to-one join | Creates 47% duplicates | ❌ **ISSUE** |

## Next Steps

1. **IMMEDIATE**: Fix weather join to prevent duplication (47% increase)
2. **HIGH PRIORITY**: Re-run pipeline after weather join fix
3. **ANALYSIS**: Verify final dataset quality after fixes
4. **DOCUMENTATION**: Update pipeline documentation with findings

## Conclusion

The pipeline has one critical issue:
1. **47% data duplication at weather join** - needs urgent fix

The 50% reduction at dropDuplicates is **expected and working correctly** (source data has 2 rows per flight).

The weather join duplication must be resolved before using the dataset for model training, as it will:
- Inflate the dataset artificially
- Create duplicate training examples
- Skew model performance metrics
- Potentially cause data leakage if not handled correctly

