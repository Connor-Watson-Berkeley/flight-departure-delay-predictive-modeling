# Flight Lineage Join Design Document

## Problem Statement

We need to join each flight to its previous flight in the lineage (same aircraft, regardless of day) to enable flight lineage feature engineering. The previous flight is simply the most recent flight by the same aircraft (tail number), ordered by actual arrival time.

This allows us to compute features such as:

- **Taxi time**: Time between previous flight's arrival and current flight's departure
- **Cumulative delays**: How delays compound as an aircraft travels through multiple flights
- **Sequence information**: Position of flight in the aircraft's sequence
- **Jump detection**: Whether the aircraft was repositioned between flights (e.g., maintenance) - detected when previous flight's destination does not match current flight's origin
- **Imputation considerations**: First flight (no previous flight) requires imputation strategy - delays from prior flights should be zero

**Key Insight**: If there's a jump/repositioning (previous flight didn't arrive at current origin), the large time delta between flights will be predictive of on-time performance, so we don't need to explicitly filter for route continuity.

## ⚠️ CRITICAL REQUIREMENT: NO ROWS DROPPED

**This is a fundamental requirement for the flight lineage join implementation:**

- **ALL flights in the input dataset MUST be preserved in the output**
- Window functions naturally preserve all rows - they do NOT drop any flights
- Flights without previous flight data (first flight, missing tail_num, etc.) will have NULL values for `prev_flight_*` columns
- NULL values are handled via imputation (see Imputation Strategy section)
- **No filtering should occur** - canceled flights, flights with missing data, etc. are all preserved
- The only exception is if flights are already filtered upstream (e.g., canceled flights already removed from dataset), but we do NOT add additional filtering in this step

**Why this matters:**
- Preserves data integrity and completeness
- Ensures downstream analysis has access to all flights
- Window functions are designed to preserve all rows - filtering would break this natural behavior
- Imputation handles missing previous flight data appropriately

## Key Requirements

1. **Join Key**: Same aircraft (`tail_num`)
2. **Ordering**: Flights ordered by `actual_arrival_time` (or scheduled if actual unavailable)
3. **Previous Flight**: The flight with rank-1 (most recent flight before current one)
4. **Canceled Flight Handling**: Handle canceled flights appropriately (use scheduled time as fallback for ranking)
5. **Efficiency**: Must be performant on large datasets (millions of flights)
6. **⚠️ CRITICAL: NO ROWS DROPPED**: All flights in the input dataset must be preserved in the output. Window functions naturally preserve all rows - flights without previous flight data will have NULL values for `prev_flight_*` columns, which is expected and handled via imputation.

## Approach: Window Functions with Ranking

### Recommended: Window Functions (Simple and Efficient)

**Why Window Functions?**
- ✅ **Simple**: Just rank by `tail_num` by `actual_arrival_time`
- ✅ **Efficient**: Single pass over data, no expensive self-join
- ✅ **Cross-day**: Window functions work across all dates when partitioned only by `tail_num`
- ✅ **Automatic sequencing**: LAG function gets the previous row in the ordered partition
- ✅ **Handles jumps naturally**: Large time deltas from jumps/repositioning are predictive features
- ✅ **Preserves all rows**: Window functions do NOT drop any rows - flights without previous flight data get NULL values (handled via imputation)

**How it works:**
```python
# Window specification: partition by tail_num, order by actual arrival time (ASCENDING)
# ASCENDING order ensures LAG gets the previous flight in chronological sequence
# Default is ascending, but explicit is clearer
window_spec = Window.partitionBy('tail_num').orderBy(F.col('arrival_timestamp').asc())
# Or simply: Window.partitionBy('tail_num').orderBy('arrival_timestamp')  # default is ascending

# Rank flights by arrival time (1 = earliest, higher = more recent)
# lineage_rank is highly predictive: indicates how many flights aircraft has completed
# Later flights (higher rank) often have more delays due to cumulative effects
df = df.withColumn('lineage_rank', F.row_number().over(window_spec))

# Get previous flight data using LAG (gets value from 1 row before in ordered partition)
# LAG returns NULL for the first flight (no previous flight)

# Core Previous Flight Information (Required for Feature Engineering)
df = df.withColumn('prev_flight_origin', F.lag('origin', 1).over(window_spec))
df = df.withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))
# Note: prev_flight_dest is particularly useful for jump detection:
#   - If prev_flight_dest != current_origin → could indicate:
#     * Aircraft repositioned (jump/maintenance) - actual repositioning
#     * Data gap - missing flights in between (data quality issue)
#   - Either way, this is a predictive feature (large time gaps affect delays)
df = df.withColumn('prev_flight_actual_dep_time', F.lag('dep_time', 1).over(window_spec))
df = df.withColumn('prev_flight_actual_arr_time', F.lag('arr_time', 1).over(window_spec))
df = df.withColumn('prev_flight_dep_delay', F.lag('DEP_DELAY', 1).over(window_spec))
df = df.withColumn('prev_flight_arr_delay', F.lag('ARR_DELAY', 1).over(window_spec))
df = df.withColumn('prev_flight_air_time', F.lag('air_time', 1).over(window_spec))

# Scheduled Times (for fallback when actual times unavailable)
df = df.withColumn('prev_flight_crs_dep_time', F.lag('crs_dep_time', 1).over(window_spec))
df = df.withColumn('prev_flight_crs_arr_time', F.lag('crs_arr_time', 1).over(window_spec))
df = df.withColumn('prev_flight_crs_elapsed_time', F.lag('crs_elapsed_time', 1).over(window_spec))

# Time Components (for turn time and taxi time calculations)
df = df.withColumn('prev_flight_taxi_in', F.lag('taxi_in', 1).over(window_spec))
df = df.withColumn('prev_flight_taxi_out', F.lag('taxi_out', 1).over(window_spec))
df = df.withColumn('prev_flight_wheels_off', F.lag('wheels_off', 1).over(window_spec))
df = df.withColumn('prev_flight_wheels_on', F.lag('wheels_on', 1).over(window_spec))
df = df.withColumn('prev_flight_actual_elapsed_time', F.lag('actual_elapsed_time', 1).over(window_spec))

# Route and Flight Information
df = df.withColumn('prev_flight_distance', F.lag('distance', 1).over(window_spec))
df = df.withColumn('prev_flight_op_carrier', F.lag('op_carrier', 1).over(window_spec))
df = df.withColumn('prev_flight_op_carrier_fl_num', F.lag('op_carrier_fl_num', 1).over(window_spec))
df = df.withColumn('prev_flight_FL_DATE', F.lag('FL_DATE', 1).over(window_spec))

# Status Flags (for understanding previous flight context)
df = df.withColumn('prev_flight_cancelled', F.lag('cancelled', 1).over(window_spec))
df = df.withColumn('prev_flight_diverted', F.lag('diverted', 1).over(window_spec))
```

**Note**: We get ALL the previous flight's information, not just destination. The previous flight's destination (`prev_flight_dest`) is useful for:
- Detecting jumps (if `prev_flight_dest != current_origin`, it's a jump/repositioning)
- Understanding route continuity
- But we don't REQUIRE it to match - jumps are fine and predictive

**Key Points:**
1. **Partition**: Only by `tail_num` (not by date) - allows cross-day matching
2. **Order**: By `arrival_timestamp` **ASCENDING** (earliest flights first) - critical for LAG to work correctly
3. **Previous Flight**: Simply the row with rank-1 (the flight immediately before in the ordered sequence)
4. **No route matching**: Don't require `prev.dest == current.origin` - jumps are fine and predictive

**Why ASCENDING Order?**
- With ASCENDING: Flight 1 (8 AM) → Flight 2 (12 PM) → Flight 3 (4 PM)
  - For Flight 3, LAG(1) correctly gets Flight 2 (the previous flight) ✓
- With DESCENDING: Flight 3 (4 PM) → Flight 2 (12 PM) → Flight 1 (8 AM)
  - For Flight 3, LAG(1) would get... nothing (it's first in the partition) ✗
- **Default**: Spark's `orderBy()` defaults to ASCENDING, but explicit is clearer

### Canceled Flight Handling

**⚠️ IMPORTANT: We do NOT filter out canceled flights - all rows must be preserved.**

**Approach:**
- **Use scheduled time as fallback**: For canceled flights (or any flights with missing actual arrival time), use scheduled arrival time (`crs_arr_time`) for ranking
- This ensures canceled flights are still included in the lineage sequence and can serve as previous flights for subsequent flights
- Canceled flights will have NULL values for actual times, but scheduled times are still available for ranking

**Implementation:**
```python
# Use scheduled time if actual unavailable (preserves all rows, including canceled flights)
df = df.withColumn(
    'arrival_time_for_ranking',
    F.coalesce(col('actual_arr_time'), col('crs_arr_time'))
)

# Note: Canceled flights should already be filtered from our dataset, but if they exist,
# we handle them gracefully without dropping rows
```

## Features to Compute

### 1. Previous Flight Information

**Core Previous Flight Data (Required):**
- `prev_flight_origin`: Previous flight's origin airport
- `prev_flight_dest`: Previous flight's destination airport
  - **Key for jump detection**: If `prev_flight_dest != current_origin`, could indicate:
    * Aircraft repositioned (jump/maintenance) - actual repositioning
    * Data gap - missing flights in between (data quality issue)
  - **Predictive feature**: Large time gaps (from jumps or data gaps) are predictive of delays
- `prev_flight_actual_dep_time`: Previous flight's actual departure time (HHMM format)
- `prev_flight_actual_arr_time`: Previous flight's actual arrival time (HHMM format)
- `prev_flight_dep_delay`: Previous flight's departure delay (minutes)
- `prev_flight_arr_delay`: Previous flight's arrival delay (minutes)
- `prev_flight_air_time`: Previous flight's air time (minutes)

**Scheduled Times (Fallback when actual unavailable):**
- `prev_flight_crs_dep_time`: Previous flight's scheduled departure time (HHMM format)
- `prev_flight_crs_arr_time`: Previous flight's scheduled arrival time (HHMM format)
- `prev_flight_crs_elapsed_time`: Previous flight's scheduled elapsed time (minutes)

**Time Components (for turn time/taxi time calculations):**
- `prev_flight_taxi_in`: Previous flight's taxi-in time (minutes) - from wheels on to gate
- `prev_flight_taxi_out`: Previous flight's taxi-out time (minutes) - from gate to wheels off
- `prev_flight_wheels_off`: Previous flight's wheels off time (HHMM format)
- `prev_flight_wheels_on`: Previous flight's wheels on time (HHMM format)
- `prev_flight_actual_elapsed_time`: Previous flight's actual elapsed time (minutes)

**Route and Flight Information:**
- `prev_flight_distance`: Previous flight's distance (miles)
- `prev_flight_op_carrier`: Previous flight's operating carrier code
- `prev_flight_op_carrier_fl_num`: Previous flight's flight number
- `prev_flight_FL_DATE`: Previous flight's date (for cross-day analysis)

**Status Flags:**
- `prev_flight_cancelled`: Previous flight's cancellation status (0/1)
- `prev_flight_diverted`: Previous flight's diversion status (0/1)

### 2. Sequence Information

- `lineage_rank`: Rank of flight in aircraft's sequence (1 = earliest flight, higher = more recent)
  - **Highly predictive**: Indicates how many flights the aircraft has already completed
  - Later flights (higher rank) often have more delays due to cumulative effects throughout the day
  - Cross-day ranking allows tracking aircraft fatigue/maintenance cycles
  - **Note**: Rank is always non-null (row_number never returns NULL), but rank=1 indicates first flight (no previous flight data - will have NULLs for prev_flight_* columns)
- `lineage_is_jump`: Boolean flag indicating if this is a jump (aircraft repositioning/maintenance) or data gap
  - `True` if previous flight's destination ≠ current flight's origin (could be repositioning OR data gap)
  - `True` if previous flight data is missing (no previous flight found - could be first flight OR data gap)
  - `False` if previous flight's destination == current flight's origin (normal sequence)
  - **Note**: Cannot distinguish between actual repositioning and missing data - both are treated as jumps

### 3. Turnover Time Features (Time from Arrival to Departure)

**Turnover Time** (also called "turn time" or "taxi time"): Time between previous flight's arrival and current flight's departure at the same airport. This is the time the aircraft spends at the airport between flights.

**Expected (Scheduled) Turnover Time:**
- `lineage_turnover_time_minutes`: Time between previous flight's scheduled arrival and current flight's scheduled departure
  - Formula: `current_crs_dep_time_minutes - prev_flight_crs_arr_time_minutes`
  - Handles day rollover (if current < previous, add 1440 minutes = 24 hours)
  - **Data Leakage**: ✅ SAFE - uses scheduled times (predetermined)
  - **Alias**: Also available as `lineage_taxi_time_minutes` and `lineage_turn_time_minutes` (same value)

**Actual Turnover Time (when available):**
- `lineage_actual_turnover_time_minutes`: Time between previous flight's actual arrival and current flight's actual departure
  - Formula: `current_actual_dep_time_minutes - prev_flight_actual_arr_time_minutes`
  - Only computed when both actual times are available AND `prev_arr_time_safe_to_use == True`
  - Handles day rollover
  - **Data Leakage**: ⚠️ RISKY - requires checking if `prev_flight_actual_arr_time <= current_crs_dep_time - 2 hours`
  - **Alias**: Also available as `lineage_actual_taxi_time_minutes` and `lineage_actual_turn_time_minutes` (same value)

**Note**: Turnover time = Turn time = Taxi time (all refer to the same metric: time from arrival to departure)

### 4. Expected Flight Time

**Expected Flight Time:**
- `lineage_expected_flight_time_minutes`: Expected duration of current flight
  - Formula: `crs_arr_time_minutes - crs_dep_time_minutes` (scheduled arrival - scheduled departure)
  - Used to estimate actual arrival time: `actual_dep_time + expected_flight_time = estimated_arrival_time`
  - **Data Leakage**: ✅ SAFE - uses scheduled times (predetermined)

### 5. Cumulative Delay Features

- `lineage_cumulative_delay`: Total delay accumulated by previous flights
- `lineage_num_previous_flights`: Number of flights the aircraft has already completed
- `lineage_avg_delay_previous_flights`: Average delay across previous flights
- `lineage_max_delay_previous_flights`: Maximum delay in previous flights

## Implementation Details

### Step 1: Identify Tail Number Column

**Challenge**: Column name may vary (`tail_num`, `TAIL_NUM`, `tail_number`, etc.)

**Solution**: Check for common variations, fallback to pattern matching
```python
tail_num_candidates = ['tail_num', 'TAIL_NUM', 'tail_number', 'TAIL_NUMBER', 'op_unique_carrier_tail_num']
# Try each candidate, then fallback to pattern matching
```

### Step 2: Prepare Data (NO FILTERING - Preserve All Rows)

**⚠️ CRITICAL: Do NOT filter out any flights. All rows must be preserved.**

**Required columns:**
- `tail_num` (or variant) - **Note**: All flights should have tail_num. If missing, lineage features will be NULL (handled via imputation)
- `FL_DATE`
- `crs_dep_time`
- `origin`
- `dest`

**Handling missing data:**
- If `tail_num` is NULL: Flights cannot be matched to previous flights, so all `prev_flight_*` columns will be NULL (handled via imputation)
- If `arrival_time` is NULL: Use scheduled arrival time (`crs_arr_time`) as fallback for ranking
- Window functions naturally handle NULLs - they don't drop rows, they just produce NULL values for LAG operations

### Step 3: Handle Canceled Flights and Create Arrival Time for Ranking

**⚠️ IMPORTANT: Do NOT filter out canceled flights. Preserve all rows.**

**Handle canceled/missing actual times:**
```python
# Use scheduled time as fallback for canceled/missing actual times (preserves all rows)
df_ranked = df_final.withColumn(
    'arrival_time_for_ranking',
    F.coalesce(
        col('actual_arr_time'),  # Use actual if available
        col('crs_arr_time')      # Fallback to scheduled
    )
)
```

**Create timestamp for proper ordering:**
```python
# Convert arrival time to timestamp for proper temporal ordering
df_ranked = df_ranked.withColumn(
    'arrival_timestamp',
    F.to_timestamp(
        F.concat(
            col('FL_DATE'), 
            F.lpad(col('arrival_time_for_ranking').cast('string'), 4, '0')
        ),
        'yyyy-MM-ddHHmm'
    )
)
```

### Step 4: Rank Flights and Get Previous Flight Data

**Window Specification:**
```python
# Window: partition by tail_num, order by arrival timestamp (ASCENDING)
# ASCENDING order is critical: earliest flights first, so LAG gets the previous flight
window_spec = Window.partitionBy(tail_num_col).orderBy(F.col('arrival_timestamp').asc())
# Note: Default is ascending, but explicit is clearer
```

**Rank Flights:**
```python
# Rank flights by arrival time (1 = earliest, higher = more recent)
# lineage_rank is highly predictive: indicates how many flights aircraft has completed
# Later flights (higher rank) often have more delays due to cumulative effects
df_lineage = df_ranked.withColumn('lineage_rank', F.row_number().over(window_spec))
```

**Get Previous Flight Data Using LAG:**
```python
# LAG gets the value from 1 row before in the ordered partition
df_lineage = df_lineage.withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))
df_lineage = df_lineage.withColumn('prev_flight_origin', F.lag('origin', 1).over(window_spec))
df_lineage = df_lineage.withColumn('prev_flight_arr_time', F.lag('arrival_time_for_ranking', 1).over(window_spec))
df_lineage = df_lineage.withColumn('prev_flight_arr_timestamp', F.lag('arrival_timestamp', 1).over(window_spec))
df_lineage = df_lineage.withColumn('prev_flight_dep_time', F.lag('dep_time', 1).over(window_spec))
df_lineage = df_lineage.withColumn('prev_flight_dep_delay', F.lag('DEP_DELAY', 1).over(window_spec))
df_lineage = df_lineage.withColumn('prev_flight_arr_delay', F.lag('ARR_DELAY', 1).over(window_spec))
# ... etc for other previous flight columns
```

**Imputation for First Flight (NULL Values):**
```python
# Apply imputation to all prev_flight_* and lineage_* features when LAG returns NULL
# This happens for first flight (lineage_rank == 1) 

# Raw Previous Flight Data
# Delays: -10 minutes (anti-delay, early departure/arrival)
df_lineage = df_lineage.withColumn(
    'prev_flight_dep_delay',
    F.coalesce(col('prev_flight_dep_delay'), F.lit(-10.0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_arr_delay',
    F.coalesce(col('prev_flight_arr_delay'), F.lit(-10.0))
)

# Scheduled times: For first flight, calculate based on current flight's scheduled departure
# Assume previous flight arrived 4 hours (240 minutes) before current scheduled departure
df_lineage = df_lineage.withColumn(
    'crs_dep_time_minutes_temp',
    (F.floor(col('crs_dep_time') / 100) * 60 + (col('crs_dep_time') % 100))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_crs_arr_time_minutes_imputed',
    when(
        col('prev_flight_crs_arr_time').isNotNull(),
        (F.floor(col('prev_flight_crs_arr_time') / 100) * 60 + (col('prev_flight_crs_arr_time') % 100))
    ).otherwise(
        # Impute: current scheduled dep - 240 minutes (4 hours)
        when(
            col('crs_dep_time_minutes_temp') >= 240,
            col('crs_dep_time_minutes_temp') - 240
        ).otherwise(
            col('crs_dep_time_minutes_temp') + 1440 - 240  # Day rollover
        )
    )
)
# Convert back to HHMM format for prev_flight_crs_arr_time
df_lineage = df_lineage.withColumn(
    'prev_flight_crs_arr_time',
    F.coalesce(
        col('prev_flight_crs_arr_time'),
        (F.floor(col('prev_flight_crs_arr_time_minutes_imputed') / 60) * 100 + 
         (col('prev_flight_crs_arr_time_minutes_imputed') % 60)).cast('int')
    )
)
# Similar for scheduled departure time
df_lineage = df_lineage.withColumn(
    'prev_flight_crs_dep_time',
    F.coalesce(
        col('prev_flight_crs_dep_time'),
        (F.floor((col('prev_flight_crs_arr_time_minutes_imputed') - 60) / 60) * 100 + 
         ((col('prev_flight_crs_arr_time_minutes_imputed') - 60) % 60)).cast('int')
    )
)
df_lineage = df_lineage.withColumn(
    'prev_flight_crs_elapsed_time',
    F.coalesce(col('prev_flight_crs_elapsed_time'), col('crs_elapsed_time'))
)

# Actual times: Use scheduled times as fallback
df_lineage = df_lineage.withColumn(
    'prev_flight_actual_dep_time',
    F.coalesce(col('prev_flight_actual_dep_time'), col('prev_flight_crs_dep_time'))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_actual_arr_time',
    F.coalesce(col('prev_flight_actual_arr_time'), col('prev_flight_crs_arr_time'))
)

# Route: Assume no jump for first flight (prev_dest = current_origin)
df_lineage = df_lineage.withColumn(
    'prev_flight_dest',
    F.coalesce(col('prev_flight_dest'), col('origin'))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_origin',
    F.coalesce(col('prev_flight_origin'), col('origin'))
)

# Time components: Use scheduled or defaults
df_lineage = df_lineage.withColumn(
    'prev_flight_air_time',
    F.coalesce(col('prev_flight_air_time'), col('prev_flight_crs_elapsed_time'))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_taxi_in',
    F.coalesce(col('prev_flight_taxi_in'), F.lit(10.0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_taxi_out',
    F.coalesce(col('prev_flight_taxi_out'), F.lit(15.0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_distance',
    F.coalesce(col('prev_flight_distance'), F.lit(0.0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_cancelled',
    F.coalesce(col('prev_flight_cancelled'), F.lit(0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_diverted',
    F.coalesce(col('prev_flight_diverted'), F.lit(0))
)

# Engineered Lineage Features
# Taxi/Turn Time: 240 minutes (4 hours) - overnight/maintenance gap
df_lineage = df_lineage.withColumn(
    'lineage_taxi_time_minutes',
    F.coalesce(col('lineage_taxi_time_minutes'), F.lit(240.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_turn_time_minutes',
    F.coalesce(col('lineage_turn_time_minutes'), F.lit(240.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_actual_taxi_time_minutes',
    F.coalesce(col('lineage_actual_taxi_time_minutes'), F.lit(240.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_actual_turn_time_minutes',
    F.coalesce(col('lineage_actual_turn_time_minutes'), F.lit(240.0))
)

# Cumulative delays: 0 (no previous flights)
df_lineage = df_lineage.withColumn(
    'lineage_cumulative_delay',
    F.coalesce(col('lineage_cumulative_delay'), F.lit(0.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_avg_delay_previous_flights',
    F.coalesce(col('lineage_avg_delay_previous_flights'), F.lit(-10.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_max_delay_previous_flights',
    F.coalesce(col('lineage_max_delay_previous_flights'), F.lit(-10.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_num_previous_flights',
    F.coalesce(col('lineage_num_previous_flights'), F.lit(0))
)
df_lineage = df_lineage.withColumn(
    'lineage_expected_flight_time_minutes',
    F.coalesce(col('lineage_expected_flight_time_minutes'), col('crs_elapsed_time'))
)
```

**Jump Detection (After Imputation):**
```python
# Jump = previous flight didn't arrive at current origin
# Note: After imputation, prev_flight_dest will never be NULL (imputed to origin for first flight)
# So jump detection is based on route mismatch only
df_lineage = df_lineage.withColumn(
    'lineage_is_jump',
    when(col('lineage_rank') == 1, F.lit(False))  # First flight is not a jump (imputed to match)
    .when(col('prev_flight_dest') != col('origin'), F.lit(True))  # Route mismatch = jump
    .otherwise(F.lit(False))
)
```

**Taxi Time Calculation:**
```python
# Convert times to minutes since midnight for calculation
df_lineage = df_lineage.withColumn(
    'crs_dep_time_minutes',
    (F.floor(col('crs_dep_time') / 100) * 60 + (col('crs_dep_time') % 100))
)

df_lineage = df_lineage.withColumn(
    'prev_flight_crs_arr_time_minutes',
    when(
        col('prev_flight_crs_arr_time').isNotNull(),
        (F.floor(col('prev_flight_crs_arr_time') / 100) * 60 + (col('prev_flight_crs_arr_time') % 100))
    ).otherwise(None)
)

# Scheduled taxi time (SAFE - no data leakage)
df_lineage = df_lineage.withColumn(
    'lineage_taxi_time_minutes',
    when(
        (col('prev_flight_crs_arr_time_minutes').isNotNull()) &
        (col('crs_dep_time_minutes').isNotNull()),
        when(
            col('crs_dep_time_minutes') >= col('prev_flight_crs_arr_time_minutes'),
            col('crs_dep_time_minutes') - col('prev_flight_crs_arr_time_minutes')
        ).otherwise(
            col('crs_dep_time_minutes') + 1440 - col('prev_flight_crs_arr_time_minutes')  # Day rollover
        )
    ).otherwise(None)
)

# Actual taxi time (RISKY - check data leakage)
# Only use if prev_flight_actual_arr_time <= current_crs_dep_time - 2 hours
df_lineage = df_lineage.withColumn(
    'prev_flight_actual_arr_time_minutes',
    when(
        col('prev_flight_actual_arr_time').isNotNull(),
        (F.floor(col('prev_flight_actual_arr_time') / 100) * 60 + (col('prev_flight_actual_arr_time') % 100))
    ).otherwise(None)
)

df_lineage = df_lineage.withColumn(
    'actual_dep_time_minutes',
    when(
        col('dep_time').isNotNull(),
        (F.floor(col('dep_time') / 100) * 60 + (col('dep_time') % 100))
    ).otherwise(None)
)

# Check data leakage: prev_arr_time must be <= crs_dep_time - 2 hours (120 minutes)
df_lineage = df_lineage.withColumn(
    'prev_arr_time_safe_to_use',
    when(
        (col('prev_flight_actual_arr_time_minutes').isNotNull()) &
        (col('crs_dep_time_minutes').isNotNull()),
        col('prev_flight_actual_arr_time_minutes') <= (col('crs_dep_time_minutes') - 120)
    ).otherwise(False)
)

df_lineage = df_lineage.withColumn(
    'lineage_actual_taxi_time_minutes',
    when(
        (col('prev_arr_time_safe_to_use') == True) &
        (col('actual_dep_time_minutes').isNotNull()) &
        (col('prev_flight_actual_arr_time_minutes').isNotNull()),
        when(
            col('actual_dep_time_minutes') >= col('prev_flight_actual_arr_time_minutes'),
            col('actual_dep_time_minutes') - col('prev_flight_actual_arr_time_minutes')
        ).otherwise(
            col('actual_dep_time_minutes') + 1440 - col('prev_flight_actual_arr_time_minutes')
        )
    ).otherwise(None)
)
```

**Turnover Time Calculation:**
```python
# Turnover time = Turn time = Taxi time (all the same: time from arrival to departure)
# Create explicit turnover time columns (aliases for clarity)

# Expected (Scheduled) Turnover Time
df_lineage = df_lineage.withColumn(
    'lineage_turnover_time_minutes',
    col('lineage_taxi_time_minutes')  # Same as scheduled taxi/turn time (SAFE)
)

# Actual Turnover Time (with data leakage check)
df_lineage = df_lineage.withColumn(
    'lineage_actual_turnover_time_minutes',
    when(
        col('prev_arr_time_safe_to_use') == True,
        col('lineage_actual_taxi_time_minutes')
    ).otherwise(None)  # Only use if safe (RISKY - already checked for data leakage)
)

# Keep aliases for backward compatibility
df_lineage = df_lineage.withColumn(
    'lineage_turn_time_minutes',
    col('lineage_turnover_time_minutes')  # Alias: scheduled turnover time
)

df_lineage = df_lineage.withColumn(
    'lineage_actual_turn_time_minutes',
    col('lineage_actual_turnover_time_minutes')  # Alias: actual turnover time
)
```

**Expected Flight Time:**
```python
# Expected flight time = scheduled arrival - scheduled departure
df_lineage = df_lineage.withColumn(
    'crs_arr_time_minutes',
    when(
        col('crs_arr_time').isNotNull(),
        (F.floor(col('crs_arr_time') / 100) * 60 + (col('crs_arr_time') % 100))
    ).otherwise(None)
)

df_lineage = df_lineage.withColumn(
    'lineage_expected_flight_time_minutes',
    when(
        (col('crs_arr_time_minutes').isNotNull()) &
        (col('crs_dep_time_minutes').isNotNull()),
        when(
            col('crs_arr_time_minutes') >= col('crs_dep_time_minutes'),
            col('crs_arr_time_minutes') - col('crs_dep_time_minutes')
        ).otherwise(
            col('crs_arr_time_minutes') + 1440 - col('crs_dep_time_minutes')  # Day rollover
        )
    ).otherwise(None)
)
```

**Taxi Time Calculation:**
```python
# Convert HHMM format to minutes since midnight
crs_dep_time_minutes = (floor(crs_dep_time / 100) * 60 + (crs_dep_time % 100))

# Taxi time with day rollover handling
taxi_time = when(
    crs_dep_time_minutes >= prev_arr_time_minutes,
    crs_dep_time_minutes - prev_arr_time_minutes
).otherwise(
    crs_dep_time_minutes + 1440 - prev_arr_time_minutes  # Add 24 hours
)
```

**Cumulative Features:**
```python
# Cumulative delay (exclude current row)
cumulative_delay = F.sum('DEP_DELAY').over(
    window_spec.rowsBetween(Window.unboundedPreceding, -1)
)
```

### Step 5: Features Are Already Added (No Separate Join Needed)

**Important**: Window functions add columns directly to the DataFrame - **no separate join is needed**. All lineage features are computed in-place using `withColumn()` operations.

**How it works:**
- Window functions operate on the entire DataFrame
- `LAG()` and other window functions preserve all rows automatically
- Features are added directly to `df_final` (or whatever your main DataFrame is named)
- No need to create a separate `df_lineage` and join back

**Implementation:**
```python
# All operations happen directly on df_final
# Window functions preserve all rows automatically

# Step 1: Add ranking
df_final = df_final.withColumn('lineage_rank', F.row_number().over(window_spec))

# Step 2: Add previous flight data (LAG operations)
df_final = df_final.withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))
# ... etc for all prev_flight_* columns

# Step 3: Compute engineered features
df_final = df_final.withColumn('lineage_turnover_time_minutes', ...)
# ... etc for all lineage_* features

# Step 4: Apply imputation
df_final = df_final.withColumn('prev_flight_dep_delay', 
    F.coalesce(col('prev_flight_dep_delay'), F.lit(-10.0)))
# ... etc for all imputation

# Result: df_final now has all lineage features - ready to save!
```

**Result**: **⚠️ CRITICAL: All flights preserved** - lineage features are NULL for flights without valid lineage data (handled via imputation). Window functions naturally preserve all rows - no filtering or dropping occurs. **No separate join step needed** - features are added directly to the DataFrame.

## Edge Cases & Handling

### 1. Missing Tail Number
- **Handling**: **DO NOT filter out** - Keep all flights. Flights without `tail_num` cannot determine lineage, so all `prev_flight_*` columns will be NULL
- **Impact**: These flights will have NULL lineage features (handled via imputation)
- **Note**: All flights should have `tail_num` in our dataset, but if missing, we preserve the row with NULL lineage features

### 2. First Flight Ever / No Previous Flight
- **Handling**: All previous flight columns are NULL (LAG returns NULL for first row in partition)
- **Jump Detection**: Marked as `True` (no previous flight = jump/repositioning)

### 3. Cross-Day Sequences
- **Problem**: Flight on Day 1 at 11:30 PM, next flight on Day 2 at 1:00 AM
- **Handling**: 
  - Window function partitioned only by `tail_num` (not date) handles cross-day automatically
  - Temporal ordering by `arrival_timestamp` correctly sequences flights across days
  - Taxi time calculation handles day boundaries automatically

### 4. Aircraft Repositioning (Jumps)
- **Detection**: `prev_flight_dest != origin` OR large time delta between flights
- **Handling**: 
  - Previous flight data is still available (LAG gets it regardless of route match)
  - Large time delta is a predictive feature (aircraft had time to reposition)
  - Jump flag can be set based on route mismatch or time delta threshold
- **Note**: We don't filter out jumps - they're informative features

### 5. Canceled Flights
- **Problem**: Canceled flight has no actual arrival time
- **Handling**: 
  - **⚠️ DO NOT filter out canceled flights** - Preserve all rows
  - Use scheduled arrival time (`crs_arr_time`) as fallback for ranking
  - Canceled flights can still serve as previous flights for subsequent flights (using scheduled times)
  - Canceled flights will have NULL values for actual times, but scheduled times are available
- **Recommendation**: Use scheduled time as fallback so canceled flights can still be ranked and included in lineage

### 5. Missing Previous Flight Data
- **Causes**: First flight, data quality issues, jump
- **Handling**: Previous flight columns are NULL, jump flag is True

### 6. Multiple Flights with Same Arrival Time
- **Problem**: Same aircraft may have multiple flights with same arrival timestamp
- **Handling**: Window function will order by arrival_timestamp, then by row order
- **Note**: This is rare but possible - LAG will get the immediately preceding row in the partition

### 7. Column Name Variations
- **Handling**: Check for common variations, pattern matching fallback
- **Error**: If no tail_num column found, raise clear error with available columns

## Performance Considerations

### Window Function Performance

**Partitioning Strategy:**
- Window functions partitioned by `tail_num` only (not by date)
- Requires a shuffle operation to group flights by aircraft
- Spark optimizes this, but it's still expensive on large datasets

**Optimization Tips:**
1. **⚠️ NO FILTERING**: All flights should have `tail_num`. If arrival time is NULL, use scheduled arrival time as fallback. **DO NOT DROP ANY ROWS.**
2. **Repartition before window**: Repartition by `tail_num` to reduce shuffling
3. **Handle canceled/missing times efficiently**: Use `coalesce(actual_arr_time, crs_arr_time)` for ranking - canceled flights should already be filtered from our dataset, but we should **NOT DROP ANY ROWS** for our analysis
4. **Cache intermediate results**: If lineage features are used multiple times
5. **Partition output**: Save lineage features partitioned by date for efficient joins later

### Memory Considerations

**Window Functions:**
- Window functions keep partition data in memory
- Large partitions (aircraft with many flights) may cause memory issues
- Spark handles this with spill-to-disk, but performance may degrade

**Mitigation:**
- Monitor partition sizes (some aircraft may have thousands of flights)
- **⚠️ Note**: Do NOT filter flights for performance - preserve all rows. If needed, filter only in downstream model pipelines, not in the custom join step.
- Use checkpointing to break lineage if needed
- Adaptive query execution (AQE) is already enabled and helps with this

## Integration with Custom Join Pipeline

### ⚠️ CRITICAL: Pipeline Placement Strategy

**IMPORTANT**: The flight lineage join should be implemented as a **single, consolidated final step** in the custom join pipeline. **DO NOT distribute the code throughout the pipeline** - keep all lineage feature engineering code together in one section for clarity and maintainability.

### Placement in Pipeline

**Recommended Location**: **Final step** - After all other joins (weather, etc.), immediately before final save

**Target Notebook**: `Custom Joins/Custom Join (all).ipynb` (official final custom join notebook)

**Pipeline Flow:**
1. Custom Join → Flights + Weather (`df_final`)
2. **Flight Lineage Join** → Add previous flight features (ALL code in one section)
3. Save Final Result → `flights_weather_joined_{version}.parquet`

**Why this location?**
- All required columns are available after weather join
- Lineage features are added before saving, so they're available for model training
- Can be computed once and reused
- **Keeps all lineage logic in one place** - easier to understand, debug, and maintain

**Implementation Structure:**
```python
# ============================================================================
# FLIGHT LINEAGE JOIN - FINAL STEP
# ============================================================================
# All flight lineage feature engineering code should be grouped together here
# DO NOT distribute this code throughout the pipeline

# Step 1: Identify tail_num column
# Step 2: Prepare arrival timestamp for ranking
# Step 3: Create window specification
# Step 4: Rank flights and get previous flight data (LAG)
# Step 5: Compute engineered features (turnover time, cumulative delays, etc.)
# Step 6: Apply imputation for NULL values
# Step 7: Add data leakage flags

# Result: df_final now has all lineage features added
# ============================================================================
```

**Insertion Point**: 
- **After all other joins are complete** (weather, etc.)
- **Immediately before "Save Final Result" cell**
- **All lineage code should be in one contiguous section** - use markdown headers to clearly separate it

**Code Organization:**
- Create a clear markdown header: `## Flight Lineage Join`
- Group all lineage-related code cells together
- Add comments explaining each step
- Keep it self-contained - don't mix with other feature engineering

### Output Schema

**New Columns Added:**

**Previous Flight Raw Data:**
- `lineage_rank`: int - Rank of flight in aircraft's sequence
- `prev_flight_origin`: string (nullable)
- `prev_flight_dest`: string (nullable)
- `prev_flight_FL_DATE`: date (nullable)
- `prev_flight_op_carrier`: string (nullable)
- `prev_flight_op_carrier_fl_num`: string (nullable)
- `prev_flight_crs_dep_time`: int (nullable)
- `prev_flight_crs_arr_time`: int (nullable)
- `prev_flight_actual_dep_time`: int (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_actual_arr_time`: int (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_dep_delay`: double (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_arr_delay`: double (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_air_time`: double (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_taxi_in`: double (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_taxi_out`: double (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_wheels_off`: int (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_wheels_on`: int (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_actual_elapsed_time`: double (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_crs_elapsed_time`: double (nullable) - ✅ **SAFE** (predetermined)
- `prev_flight_distance`: double (nullable) - ✅ **SAFE** (predetermined)
- `prev_flight_cancelled`: int (nullable) - ⚠️ **DATA LEAKAGE RISK**
- `prev_flight_diverted`: int (nullable) - ⚠️ **DATA LEAKAGE RISK**

**Lineage Engineered Features:**
- `lineage_is_jump`: boolean
- `lineage_turnover_time_minutes`: double (nullable) - ✅ **SAFE** (uses scheduled times) - Expected turnover time
- `lineage_actual_turnover_time_minutes`: double (nullable) - ⚠️ **DATA LEAKAGE RISK** (only safe if `prev_arr_time_safe_to_use == True`) - Actual turnover time
- `lineage_taxi_time_minutes`: double (nullable) - ✅ **SAFE** (alias for `lineage_turnover_time_minutes`)
- `lineage_actual_taxi_time_minutes`: double (nullable) - ⚠️ **DATA LEAKAGE RISK** (alias for `lineage_actual_turnover_time_minutes`)
- `lineage_turn_time_minutes`: double (nullable) - ✅ **SAFE** (alias for `lineage_turnover_time_minutes`)
- `lineage_actual_turn_time_minutes`: double (nullable) - ⚠️ **DATA LEAKAGE RISK** (alias for `lineage_actual_turnover_time_minutes`)
- `lineage_expected_flight_time_minutes`: double (nullable) - ✅ **SAFE** (predetermined)
- `lineage_cumulative_delay`: double (nullable) - ⚠️ **DATA LEAKAGE RISK** (derived from actual delays)
- `lineage_num_previous_flights`: long (nullable) - ✅ **SAFE** (count only)
- `lineage_avg_delay_previous_flights`: double (nullable) - ⚠️ **DATA LEAKAGE RISK** (derived from actual delays)
- `lineage_max_delay_previous_flights`: double (nullable) - ⚠️ **DATA LEAKAGE RISK** (derived from actual delays)
- `prev_arr_time_safe_to_use`: boolean - Flag indicating if previous arrival time is safe to use

**Naming Convention:**
- Previous flight columns: `prev_flight_{column_name}`
- Lineage-specific features: `lineage_{feature_name}`
- Data leakage flags: `{feature_name}_safe_to_use` or documented in comments

## Data Leakage Prevention

### Data Leakage Criteria

**Rule**: A feature has data leakage if it uses information that would not be available at prediction time.

**For Flight Lineage Features:**
- **Cutoff Time**: `current_flight_scheduled_departure_time - 2 hours`
- **Safe to Use**: Any actual time/event that occurred **at or before** the cutoff time
- **Unsafe to Use**: Any actual time/event that occurred **after** the cutoff time

### Data Leakage Classification

#### ✅ SAFE - No Data Leakage (Predetermined/Scheduled)

These features are safe because they use predetermined information (schedules) that would be known in advance:

- `prev_flight_crs_dep_time` - Previous flight's scheduled departure (predetermined)
- `prev_flight_crs_arr_time` - Previous flight's scheduled arrival (predetermined)
- `prev_flight_crs_elapsed_time` - Previous flight's scheduled elapsed time (predetermined)
- `prev_flight_distance` - Previous flight's distance (predetermined)
- `prev_flight_origin` - Previous flight's origin (predetermined)
- `prev_flight_dest` - Previous flight's destination (predetermined)
- `prev_flight_op_carrier` - Previous flight's carrier (predetermined)
- `prev_flight_op_carrier_fl_num` - Previous flight's flight number (predetermined)
- `prev_flight_FL_DATE` - Previous flight's date (predetermined)
- `lineage_turnover_time_minutes` - Expected turnover time using scheduled times (predetermined) ✅ SAFE
- `lineage_taxi_time_minutes` - Alias for `lineage_turnover_time_minutes` (predetermined) ✅ SAFE
- `lineage_turn_time_minutes` - Alias for `lineage_turnover_time_minutes` (predetermined) ✅ SAFE
- `lineage_expected_flight_time_minutes` - Uses scheduled times (predetermined)
- `lineage_num_previous_flights` - Count only (no actual times)

#### ⚠️ RISKY - Potential Data Leakage (Actual Times/Events)

These features use actual times/events that may have occurred after the cutoff time. **Must check timestamp before use:**

- `prev_flight_actual_dep_time` - ⚠️ **RISKY** - Only safe if `prev_flight_actual_dep_time <= current_crs_dep_time - 2 hours`
- `prev_flight_actual_arr_time` - ⚠️ **RISKY** - Only safe if `prev_flight_actual_arr_time <= current_crs_dep_time - 2 hours`
- `prev_flight_dep_delay` - ⚠️ **RISKY** - Derived from actual departure time
- `prev_flight_arr_delay` - ⚠️ **RISKY** - Derived from actual arrival time
- `prev_flight_air_time` - ⚠️ **RISKY** - Actual air time
- `prev_flight_taxi_in` - ⚠️ **RISKY** - Actual taxi-in time
- `prev_flight_taxi_out` - ⚠️ **RISKY** - Actual taxi-out time
- `prev_flight_wheels_off` - ⚠️ **RISKY** - Actual wheels off time
- `prev_flight_wheels_on` - ⚠️ **RISKY** - Actual wheels on time
- `prev_flight_actual_elapsed_time` - ⚠️ **RISKY** - Actual elapsed time
- `prev_flight_cancelled` - ⚠️ **RISKY** - Cancellation status (may be known late)
- `prev_flight_diverted` - ⚠️ **RISKY** - Diversion status (may be known late)
- `lineage_actual_turnover_time_minutes` - ⚠️ **RISKY** - Actual turnover time using actual times (only safe if `prev_arr_time_safe_to_use == True`)
- `lineage_actual_taxi_time_minutes` - ⚠️ **RISKY** - Alias for `lineage_actual_turnover_time_minutes` (only safe if `prev_arr_time_safe_to_use == True`)
- `lineage_actual_turn_time_minutes` - ⚠️ **RISKY** - Alias for `lineage_actual_turnover_time_minutes` (only safe if `prev_arr_time_safe_to_use == True`)
- `lineage_cumulative_delay` - ⚠️ **RISKY** - Derived from actual delays
- `lineage_avg_delay_previous_flights` - ⚠️ **RISKY** - Derived from actual delays
- `lineage_max_delay_previous_flights` - ⚠️ **RISKY** - Derived from actual delays

### Data Leakage Check Implementation

**Check if Previous Arrival Time is Safe:**
```python
# Previous arrival time is safe if it occurred at least 2 hours before current scheduled departure
df_lineage = df_lineage.withColumn(
    'prev_arr_time_safe_to_use',
    when(
        (col('prev_flight_actual_arr_time_minutes').isNotNull()) &
        (col('crs_dep_time_minutes').isNotNull()),
        col('prev_flight_actual_arr_time_minutes') <= (col('crs_dep_time_minutes') - 120)  # 2 hours = 120 minutes
    ).otherwise(False)
)
```

**Check if Previous Departure Time is Safe:**
```python
# Previous departure time is safe if it occurred at least 2 hours before current scheduled departure
df_lineage = df_lineage.withColumn(
    'prev_dep_time_safe_to_use',
    when(
        (col('prev_flight_actual_dep_time_minutes').isNotNull()) &
        (col('crs_dep_time_minutes').isNotNull()),
        col('prev_flight_actual_dep_time_minutes') <= (col('crs_dep_time_minutes') - 120)
    ).otherwise(False)
)
```

### Using Safe Features Only

**Best Practice**: When using risky features in models:
1. **⚠️ Note**: This filtering is for MODEL TRAINING/EVALUATION only, NOT for the custom join step. The custom join step preserves ALL rows.
2. In model pipelines, you may filter to only rows where `prev_arr_time_safe_to_use == True` (or `prev_dep_time_safe_to_use == True`) for training
3. Set risky features to NULL when unsafe (already done in custom join)
4. Use scheduled times as fallback when actual times are unsafe

**Example:**
```python
# Only use actual taxi time if previous arrival time is safe
df_lineage = df_lineage.withColumn(
    'lineage_actual_taxi_time_minutes_safe',
    when(col('prev_arr_time_safe_to_use') == True, col('lineage_actual_taxi_time_minutes'))
    .otherwise(None)
)
```

## Validation & Testing

### Validation Checks

1. **Uniqueness**: Each flight should have at most one previous flight
2. **Sequence**: `lineage_seq_num` should be sequential (1, 2, 3, ...) for same aircraft/day
3. **Jump Detection**: First flight should have `lineage_is_jump = False`
4. **Taxi Time**: Should be positive and reasonable (e.g., 10-300 minutes)
5. **Cumulative Delay**: Should increase (or stay same) as sequence number increases

### Sample Queries

```python
# Check sequence numbers
df.groupBy('tail_num', 'FL_DATE').agg(
    F.min('lineage_seq_num').alias('min_seq'),
    F.max('lineage_seq_num').alias('max_seq'),
    F.count('*').alias('num_flights')
).filter(col('max_seq') != col('num_flights')).show()  # Should be empty

# Check taxi times
# Note: This filter is for validation/testing only - NOT for the actual join step
# The join step preserves ALL rows, even if lineage_taxi_time_minutes is NULL
df.filter(col('lineage_taxi_time_minutes').isNotNull()).select(
    F.min('lineage_taxi_time_minutes'),
    F.max('lineage_taxi_time_minutes'),
    F.avg('lineage_taxi_time_minutes')
).show()

# Check jump detection
df.groupBy('lineage_is_jump').count().show()
```

## Future Enhancements

### Potential Additions

1. **Multi-hop Features**: Previous 2-3 flights (not just immediate previous)
2. **Expected Values**: Join with conditional expected turn time/air time tables
3. **Temporal Features**: Time since first flight, time until next flight
4. **Route Patterns**: Detect common route patterns for aircraft
5. **Maintenance Windows**: Detect maintenance/repositioning patterns

### Performance Optimizations

1. **Materialized Lineage Table**: Pre-compute and save lineage features separately
2. **Partitioning**: Partition lineage table by date for efficient joins
3. **Caching**: Cache lineage features if used multiple times
4. **Incremental Updates**: Only recompute lineage for new flights

## Imputation Strategy for First Flight (NULL Values)

**When LAG Returns NULL:**
- **First flight in sequence**: No previous flight exists (lineage_rank == 1)
- **Data gaps**: Missing flights in the sequence (treated same as first flight)
- **Missing tail_num**: Flights without tail_num cannot determine lineage (all prev_flight_* columns NULL)
- **Note**: We do NOT filter out any flights - all rows are preserved, NULLs are handled via imputation

**Imputation Rationale:**
- **First flight assumption**: 
  - **Aircraft is at airport well ahead of schedule**: For the first flight of a plane (or after a gap), we assume the aircraft arrived at the airport well in advance (4 hours before scheduled departure)
  - This represents overnight parking, maintenance, or repositioning time
  - The aircraft has plenty of time to prepare, so delays from prior flights don't propagate
- **Long time gap**: Impute 4 hours (240 minutes) between previous arrival and current departure
  - Represents overnight/maintenance gap before first flight
  - Large enough to be clearly distinguishable from normal turn times (typically 30-90 minutes)
  - **Assumption**: Aircraft was at the airport 4 hours before scheduled departure
- **Anti-delay**: Use -10 minutes (early departure/arrival) to represent on-time/early performance
  - Negative delays indicate early departures/arrivals (good performance)
  - **Assumption**: First flight starts fresh with no delay propagation, aircraft is ready early
  - Aircraft being at airport well ahead of schedule means it's ready to depart on-time or early

### Imputation Values by Feature

**Raw Previous Flight Data (prev_flight_*):**

```python
# Delays: -10 minutes (anti-delay, early departure/arrival)
df_lineage = df_lineage.withColumn(
    'prev_flight_dep_delay',
    F.coalesce(col('prev_flight_dep_delay'), F.lit(-10.0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_arr_delay',
    F.coalesce(col('prev_flight_arr_delay'), F.lit(-10.0))
)

# Times: Use scheduled times as fallback (predetermined, safe)
df_lineage = df_lineage.withColumn(
    'prev_flight_actual_dep_time',
    F.coalesce(col('prev_flight_actual_dep_time'), col('prev_flight_crs_dep_time'))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_actual_arr_time',
    F.coalesce(col('prev_flight_actual_arr_time'), col('prev_flight_crs_arr_time'))
)

# Scheduled times: Keep as-is (already from LAG, may be NULL for first flight)
# If NULL, these indicate no previous flight - leave as NULL or use current flight's origin?
# Option: Set to current origin for prev_flight_dest (makes jump detection work)
df_lineage = df_lineage.withColumn(
    'prev_flight_dest',
    F.coalesce(col('prev_flight_dest'), col('origin'))  # Assume no jump for first flight
)
df_lineage = df_lineage.withColumn(
    'prev_flight_origin',
    F.coalesce(col('prev_flight_origin'), col('origin'))  # Same airport
)

# Time components: Use scheduled elapsed time or reasonable defaults
df_lineage = df_lineage.withColumn(
    'prev_flight_air_time',
    F.coalesce(col('prev_flight_air_time'), col('prev_flight_crs_elapsed_time'))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_taxi_in',
    F.coalesce(col('prev_flight_taxi_in'), F.lit(10.0))  # Default 10 minutes
)
df_lineage = df_lineage.withColumn(
    'prev_flight_taxi_out',
    F.coalesce(col('prev_flight_taxi_out'), F.lit(15.0))  # Default 15 minutes
)
df_lineage = df_lineage.withColumn(
    'prev_flight_actual_elapsed_time',
    F.coalesce(col('prev_flight_actual_elapsed_time'), col('prev_flight_crs_elapsed_time'))
)

# Route information: Use current flight's origin (assume no jump)
df_lineage = df_lineage.withColumn(
    'prev_flight_distance',
    F.coalesce(col('prev_flight_distance'), F.lit(0.0))  # No distance for first flight
)

# Status flags: Assume not canceled/diverted
df_lineage = df_lineage.withColumn(
    'prev_flight_cancelled',
    F.coalesce(col('prev_flight_cancelled'), F.lit(0))
)
df_lineage = df_lineage.withColumn(
    'prev_flight_diverted',
    F.coalesce(col('prev_flight_diverted'), F.lit(0))
)
```

**Engineered Lineage Features (lineage_*):**

```python
# Turnover Time: 240 minutes (4 hours) - represents overnight/maintenance gap
# Expected (Scheduled) Turnover Time
df_lineage = df_lineage.withColumn(
    'lineage_turnover_time_minutes',
    F.coalesce(col('lineage_turnover_time_minutes'), col('lineage_taxi_time_minutes'), F.lit(240.0))
)

# Actual Turnover Time: Same as scheduled (240 minutes) when imputing
df_lineage = df_lineage.withColumn(
    'lineage_actual_turnover_time_minutes',
    F.coalesce(col('lineage_actual_turnover_time_minutes'), col('lineage_actual_taxi_time_minutes'), F.lit(240.0))
)

# Keep aliases for backward compatibility
df_lineage = df_lineage.withColumn(
    'lineage_taxi_time_minutes',
    F.coalesce(col('lineage_taxi_time_minutes'), col('lineage_turnover_time_minutes'), F.lit(240.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_turn_time_minutes',
    F.coalesce(col('lineage_turn_time_minutes'), col('lineage_turnover_time_minutes'), F.lit(240.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_actual_taxi_time_minutes',
    F.coalesce(col('lineage_actual_taxi_time_minutes'), col('lineage_actual_turnover_time_minutes'), F.lit(240.0))
)
df_lineage = df_lineage.withColumn(
    'lineage_actual_turn_time_minutes',
    F.coalesce(col('lineage_actual_turn_time_minutes'), col('lineage_actual_turnover_time_minutes'), F.lit(240.0))
)

# Cumulative delays: 0 (no previous flights to accumulate delays from)
df_lineage = df_lineage.withColumn(
    'lineage_cumulative_delay',
    F.coalesce(col('lineage_cumulative_delay'), F.lit(0.0))
)

# Average delay: -10 (anti-delay, same as individual delays)
df_lineage = df_lineage.withColumn(
    'lineage_avg_delay_previous_flights',
    F.coalesce(col('lineage_avg_delay_previous_flights'), F.lit(-10.0))
)

# Max delay: -10 (anti-delay)
df_lineage = df_lineage.withColumn(
    'lineage_max_delay_previous_flights',
    F.coalesce(col('lineage_max_delay_previous_flights'), F.lit(-10.0))
)

# Number of previous flights: 0 (no previous flights)
df_lineage = df_lineage.withColumn(
    'lineage_num_previous_flights',
    F.coalesce(col('lineage_num_previous_flights'), F.lit(0))
)

# Expected flight time: Use scheduled elapsed time
df_lineage = df_lineage.withColumn(
    'lineage_expected_flight_time_minutes',
    F.coalesce(col('lineage_expected_flight_time_minutes'), col('crs_elapsed_time'))
)

# Jump flag: False for first flight (assume no jump, just first flight)
# Note: This will be set based on prev_flight_dest != origin, but for first flight
# we imputed prev_flight_dest = origin, so jump will be False
```

### Summary of Imputation Values

**Key Assumption**: For first flight, aircraft is at the airport well ahead of schedule (4 hours before scheduled departure), allowing it to depart on-time or early with no prior delay propagation.

| Feature | Imputation Value | Rationale |
|---------|-----------------|-----------|
| `prev_flight_dep_delay` | -10 minutes | Anti-delay (early departure) - aircraft ready early |
| `prev_flight_arr_delay` | -10 minutes | Anti-delay (early arrival) - aircraft ready early |
| `prev_flight_crs_dep_time` | `current_crs_dep_time - 4 hours` | Calculate from current scheduled dep minus 4 hours |
| `prev_flight_crs_arr_time` | `current_crs_dep_time - 4 hours` | Calculate from current scheduled dep minus 4 hours |
| `prev_flight_crs_elapsed_time` | `current_crs_elapsed_time` | Use current flight's scheduled elapsed time |
| `prev_flight_actual_dep_time` | `prev_flight_crs_dep_time` | Use scheduled time (predetermined) |
| `prev_flight_actual_arr_time` | `prev_flight_crs_arr_time` | Use scheduled time (predetermined) |
| `prev_flight_dest` | `current_origin` | Assume no jump for first flight |
| `prev_flight_origin` | `current_origin` | Same airport |
| `prev_flight_air_time` | `prev_flight_crs_elapsed_time` | Use scheduled elapsed time |
| `prev_flight_taxi_in` | 10 minutes | Default reasonable value |
| `prev_flight_taxi_out` | 15 minutes | Default reasonable value |
| `prev_flight_distance` | 0.0 | No distance for first flight |
| `prev_flight_cancelled` | 0 | Assume not canceled |
| `prev_flight_diverted` | 0 | Assume not diverted |
| `prev_flight_wheels_off` | `prev_flight_crs_dep_time` | Use scheduled departure time |
| `prev_flight_wheels_on` | `prev_flight_crs_arr_time` | Use scheduled arrival time |
| `prev_flight_actual_elapsed_time` | `prev_flight_crs_elapsed_time` | Use scheduled elapsed time |
| `prev_flight_op_carrier` | `current_op_carrier` | Use current flight's carrier |
| `prev_flight_op_carrier_fl_num` | `current_op_carrier_fl_num` | Use current flight's flight number |
| `prev_flight_FL_DATE` | `current_FL_DATE` | Use current flight's date |
| `lineage_turnover_time_minutes` | 240 minutes | 4 hours (overnight/maintenance gap) - Expected turnover time |
| `lineage_actual_turnover_time_minutes` | 240 minutes | 4 hours (same as scheduled) - Actual turnover time |
| `lineage_taxi_time_minutes` | 240 minutes | 4 hours (alias for turnover time) |
| `lineage_turn_time_minutes` | 240 minutes | 4 hours (alias for turnover time) |
| `lineage_actual_taxi_time_minutes` | 240 minutes | 4 hours (alias for actual turnover time) |
| `lineage_actual_turn_time_minutes` | 240 minutes | 4 hours (alias for actual turnover time) |
| `lineage_cumulative_delay` | 0.0 | No previous flights |
| `lineage_avg_delay_previous_flights` | -10.0 | Anti-delay |
| `lineage_max_delay_previous_flights` | -10.0 | Anti-delay |
| `lineage_num_previous_flights` | 0 | No previous flights |
| `lineage_expected_flight_time_minutes` | `crs_elapsed_time` | Use scheduled elapsed time |

### Implementation Note

**All imputation happens in custom join step** using `F.coalesce()` to replace NULLs with imputed values. This ensures:
- No NULLs in output features (simplifies model pipelines)
- Consistent imputation strategy across all models
- First flight features are clearly distinguishable (240 min gap, -10 delay)

## References

- **Flight Lineage Design Doc**: `notebooks/Feature Engineering/FLIGHT_LINEAGE_DESIGN.md`
- **Window Functions Guide**: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.html
- **LAG Function**: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lag.html

---

**Last Updated**: 2025-01-XX
**Author**: Team 4_2
**Status**: Design Proposal

