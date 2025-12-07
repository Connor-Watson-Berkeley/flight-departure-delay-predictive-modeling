# Flight Lineage Feature Engineering - Design Document

## Problem Statement

We need to engineer features that track how delays compound as planes travel through multiple flights in a day (flight sequences). For a sequence like A→B→A→C [Break] D→E, we need to:

1. Track previous flight information
2. Compute cumulative delays
3. Calculate conditional expected values (turn time, air time)
4. Detect "impossible on-time" scenarios
5. Handle jumps (maintenance, repositioning, data gaps)

## Key Question: GraphFrames vs. Window Functions?

### What We Actually Need (from Flight Lineage Features Experiment):

**1. Previous Flight Information**
- `prev_flight_arr_delay`, `prev_flight_dep_delay`
- `prev_flight_actual_dep_time`, `prev_flight_actual_arr_time`
- `prev_flight_origin`, `prev_flight_dest`

**2. Cumulative Delay Features**
- `cumulative_delay_since_3am`
- `num_previous_flights_today`
- `avg_delay_per_previous_flight`

**3. Conditional Expected Values**
- `expected_turn_time_carrier_airport`
- `expected_air_time_route`
- `expected_taxi_time_airport_time`

**4. Deterministic Calculations**
- `expected_departure_time_current`
- `impossible_on_time_flag`
- `time_buffer`

### Analysis: Are These Graph Operations?

**NO.** All of these are:
- **Window functions** (LAG, SUM, AVG over partitions)
- **Joins** (with pre-computed conditional expected value tables)
- **Conditional logic** (IF-THEN calculations)

**Graph algorithms** (PageRank, shortest paths, community detection) are NOT needed.

## Recommended Approach: Window Functions + Joins

### Structure

**Nodes as Conceptual Model:**
- Each flight is a "node" with attributes: `(tail_num, FL_DATE, crs_dep_time, origin, dest, ...)`
- Sequences are tracked via window functions, not graph edges

**Implementation:**
```python
# Window to order flights by tail_num, date, scheduled departure time
window_spec = Window.partitionBy('tail_num', 'FL_DATE').orderBy('crs_dep_time')

# Previous flight info using LAG (window function, NOT a join!)
# LAG gets the value from the previous row in the ordered partition
df = df.withColumn('prev_dest', F.lag('dest', 1).over(window_spec))
df = df.withColumn('prev_arr_delay', F.lag('ARR_DELAY', 1).over(window_spec))
df = df.withColumn('prev_air_time', F.lag('air_time', 1).over(window_spec))
```

**How LAG Works**:
- **Partition**: Groups rows by `(tail_num, FL_DATE)` - all flights by same plane on same day
- **Order**: Orders by `crs_dep_time` - earliest to latest  
- **LAG(column, 1)**: Gets the value from 1 row before in the ordered partition
- **No join needed**: All data is in the same DataFrame, LAG just looks at the previous row

# Cumulative delays using SUM
df = df.withColumn('cumulative_delay', 
    F.sum('DEP_DELAY').over(window_spec.rowsBetween(Window.unboundedPreceding, -1)))

# Sequence number
df = df.withColumn('seq_num', F.row_number().over(window_spec))

# Jump detection
df = df.withColumn('is_jump',
    when(col('seq_num') == 1, False)
    .when(col('prev_dest').isNull(), True)
    .otherwise(col('prev_dest') != col('origin')))
```

**Conditional Expected Values:**
- Pre-compute lookup tables (from Time-Series Features Experiment)
- Join back to flight data:
```python
# Join with conditional expected values
df = df.join(
    expected_turn_time_carrier_airport,
    ['op_carrier', 'origin'],
    'left'
)
df = df.join(
    expected_air_time_route,
    ['origin', 'dest'],
    'left'
)
```

## Why NOT GraphFrames?

1. **Self-loops problem**: GraphFrames doesn't handle self-loops well (tail_num → tail_num)
2. **No graph algorithms needed**: We don't need PageRank, shortest paths, etc.
3. **Extra complexity**: Building a graph just to extract features adds unnecessary steps
4. **Performance**: Window functions are optimized for sequential operations
5. **Data leakage prevention**: Window functions naturally handle temporal ordering

## When GraphFrames WOULD Be Useful

- If we needed **PageRank** to identify "important" aircraft/airports
- If we needed **shortest paths** between airports
- If we needed **community detection** to find aircraft clusters
- If we needed **graph-based embeddings** for representation learning

**None of these are required for Flight Lineage features.**

## Sequential Graph Structure (If We Still Want It)

If we want a graph for visualization/analysis (not feature engineering), nodes should be:

**Node ID**: `{tail_num}_{FL_DATE}_{seq_num}` or `{tail_num}_{FL_DATE}_{crs_dep_time}`
- Ensures uniqueness
- Preserves temporal ordering
- No self-loops

**Edges**: `(node_i, node_i+1)` where nodes are consecutive flights
- `src`: `{tail_num}_{date}_{seq_num}`
- `dst`: `{tail_num}_{date}_{seq_num+1}`
- Edge attributes: `dest_airport`, `air_time`, `is_jump`, etc.

**But again**: This is just for visualization. Feature engineering should use window functions.

## Implementation Plan

### Phase 1: Window Function Features (Primary)
1. Order flights by (tail_num, FL_DATE, crs_dep_time)
2. Use LAG to get previous flight info
3. Use SUM/AVG to get cumulative statistics
4. Detect jumps

### Phase 2: Conditional Expected Values (Joins)
1. Load pre-computed tables from Time-Series Features Experiment
2. Join on (carrier, airport), (origin, dest), etc.
3. Use for deterministic calculations

### Phase 3: Deterministic Features
1. Calculate expected departure time
2. Compute time buffer
3. Flag impossible on-time scenarios

### Phase 4: Optional Graph (Visualization Only)
1. Build sequential graph with unique node IDs
2. Use for visualization/analysis
3. NOT for feature engineering

## Key Question: How to Pull Engineered Features from Previous Flight?

**Answer**: Use LAG (window function) on computed columns in a multi-pass approach.

**Important**: LAG is NOT a join or primary key. It's a window function that gets the previous row's value within the same DataFrame.

### The Challenge

Some features depend on **engineered features from the previous flight**, not just raw values:
- `expected_arrival_time_prev_flight` (computed for flight 2)
- Then used to compute `expected_departure_time_current` for flight 3

### Solution: Multi-Pass Feature Engineering

In Spark, `withColumn` operations are evaluated sequentially, so you can:

1. **First pass**: Pull raw values from previous flight
2. **Second pass**: Compute features using those raw values
3. **Third pass**: Use LAG to pull those computed features for the next flight

**Example**:
```python
window_spec = Window.partitionBy('tail_num', 'FL_DATE').orderBy('crs_dep_time')

# Pass 1: Pull raw values from previous flight
df = df.withColumn('prev_actual_dep_time', F.lag('dep_time', 1).over(window_spec))
df = df.withColumn('prev_flight_origin', F.lag('origin', 1).over(window_spec))
df = df.withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))

# Pass 2: Join expected values for PREVIOUS flight's route
# IMPORTANT: Join on (prev_flight_origin, prev_flight_dest), not (origin, dest)!
df = df.join(
    expected_air_time_route.alias('prev_route'),
    (col('prev_flight_origin') == col('prev_route.origin')) &
    (col('prev_flight_dest') == col('prev_route.dest')),
    'left'
).select(
    col('*'),
    col('prev_route.expected_air_time_route').alias('prev_expected_air_time_route')
)

# Pass 3: Compute features for CURRENT flight using previous flight's values
df = df.withColumn('expected_arrival_time_prev_flight',
    when(
        (col('prev_actual_dep_time').isNotNull()) &
        (col('prev_expected_air_time_route').isNotNull()),
        col('prev_actual_dep_time') + col('prev_expected_air_time_route')
    )
    .otherwise(None)
)

# Pass 4: Pull computed features from previous flight for NEXT flight
df = df.withColumn('prev_expected_arrival_time', 
    F.lag('expected_arrival_time_prev_flight', 1).over(window_spec)
)

# Pass 5: Join expected values for CURRENT flight's airport
df = df.join(expected_turn_time_carrier_airport, ['op_carrier', 'origin'], 'left')

# Pass 6: Use prev_expected_arrival_time to compute features for current flight
df = df.withColumn('expected_departure_time_current',
    when(
        (col('prev_expected_arrival_time').isNotNull()) &
        (col('expected_turn_time_carrier_airport').isNotNull()),
        col('prev_expected_arrival_time') + col('expected_turn_time_carrier_airport')
    )
    .otherwise(None)
)
```

**Key Point**: When joining expected values for the previous flight, join on `(prev_flight_origin, prev_flight_dest)`, not `(origin, dest)`!

### Important Considerations

1. **Data Leakage Prevention**: Only use `prev_actual_dep_time` if it's >= 2 hours before current scheduled departure
2. **Null Handling**: Previous flight features may be NULL (first flight, jump, etc.)
3. **Order Matters**: Compute features in dependency order
4. **Window Specification**: Must be consistent across all LAG operations

### Alternative: Self-Join Approach

If you need more complex dependencies, you could also:
1. Compute all features for all flights
2. Self-join on `(tail_num, FL_DATE, seq_num)` to get previous flight's features
3. This is more explicit but less efficient than LAG

**Recommendation**: Use LAG with multi-pass approach for efficiency.

## Performance Optimization: Pre-Compute and Materialize

### The Problem

Computing window functions over the entire dataset on every model run is **extremely expensive**:
- Window functions require partitioning by `(tail_num, FL_DATE)` and ordering by `crs_dep_time`
- This requires a full shuffle and sort of the entire dataset
- No natural join key exists to match Flight 1 → Flight 2

### Solution: Materialized Lineage Features Table

**Pre-compute lineage features ONCE after the custom join, save as a lookup table.**

#### Step 1: Create Unique Flight Key

Create a unique identifier for each flight that can be used for joins.

**Important**: The flight key must be unique. If `{tail_num}_{FL_DATE}_{crs_dep_time}_{origin}_{dest}` has duplicates, add more fields:

```python
# Option A: Composite key with flight number (recommended for uniqueness)
flight_key = F.concat(
    col('tail_num'), F.lit('_'),
    col('FL_DATE'), F.lit('_'),
    col('crs_dep_time'), F.lit('_'),
    col('op_carrier_fl_num'), F.lit('_'),  # Add flight number for uniqueness
    col('origin'), F.lit('_'),
    col('dest')
)

# Option B: Hash-based key (shorter, but not human-readable)
flight_key = F.sha2(
    F.concat(
        col('tail_num'), col('FL_DATE'), 
        col('crs_dep_time'), col('op_carrier_fl_num'),
        col('origin'), col('dest')
    ), 
    256
)
```

**Recommendation**: 
- Use composite key with `op_carrier_fl_num` for debuggability and uniqueness
- Always verify uniqueness: `df.select('flight_key').distinct().count() == df.count()`
- If duplicates exist, add more fields (e.g., `op_carrier`) until unique

#### Step 2: Pre-Compute All Lineage Features

Run window functions ONCE on the full dataset after custom join:

```python
# After custom join, compute lineage features
window_spec = Window.partitionBy('tail_num', 'FL_DATE').orderBy('crs_dep_time')

lineage_features = (
    df_after_custom_join
    .withColumn('flight_key', create_flight_key(...))
    .withColumn('seq_num', F.row_number().over(window_spec))
    .withColumn('prev_flight_dest', F.lag('dest', 1).over(window_spec))
    .withColumn('prev_actual_dep_time', F.lag('dep_time', 1).over(window_spec))
    # ... all other lineage features ...
)

# Save as materialized table
lineage_features.write.mode("overwrite").parquet(
    "dbfs:/mnt/.../lineage_features_materialized.parquet"
)
```

#### Step 3: Join Back to Main Dataset

When needed for model training/prediction:

```python
# Load materialized lineage features
lineage_features = spark.read.parquet(".../lineage_features_materialized.parquet")

# Join back to main dataset using flight_key
df_with_lineage = (
    df_main
    .withColumn('flight_key', create_flight_key(...))
    .join(lineage_features, 'flight_key', 'left')
)
```

### Benefits

1. **Performance**: Window functions computed ONCE, not on every model run
2. **Efficiency**: Simple join operation instead of expensive window computation
3. **Scalability**: Can partition lineage table by date for faster joins
4. **Maintainability**: Clear separation between feature computation and model training

### Implementation Location

**Best place**: After the custom join, as a post-processing step:

```
Pipeline Flow:
1. Custom Join → df_joined (with weather, etc.)
2. Compute Lineage Features → df_with_lineage
   - Create flight_key
   - Compute window functions (expensive, one-time)
   - Join with conditional expected value tables
3. Save → lineage_features_materialized.parquet
4. Model Training → Join lineage features using flight_key (fast!)
```

**Key Point**: The expensive window function computation happens ONCE, not on every model run.

### Storage Strategy

Partition the materialized table by date for efficient joins:

```python
lineage_features.write.mode("overwrite").partitionBy("FL_DATE").parquet(
    "dbfs:/mnt/.../lineage_features_materialized/"
)
```

This allows Spark to skip entire date partitions when joining.

## Conclusion

**Use Window Functions + Joins for feature engineering.**
**Pre-compute and materialize lineage features as a lookup table.**
**Join back to main dataset using flight_key for efficient model training.**
**Use GraphFrames only if you need graph algorithms (which we don't).**
**Build a sequential graph only for visualization/analysis purposes.**

