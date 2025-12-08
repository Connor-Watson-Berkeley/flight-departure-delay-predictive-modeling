"""
conditional_expected_values_mvp.py - MVP Pipeline Estimator for Conditional Expected Values

Simplified Spark ML Estimator that computes conditional expected values on-the-fly from training data.
Uses simple averages with month-based temporal features (no Prophet).

Features added:
- expected_air_time_route_minutes: Average air time by route (origin, dest)
- expected_air_time_route_month_minutes: Average air time by route and month (temporal)
    - expected_turnover_time_carrier_airport_minutes: Average turnover time by carrier and airport
    - expected_turnover_time_carrier_airport_month_minutes: Average turnover time by carrier, airport, and month (temporal, most specific)
    - expected_turnover_time_airport_minutes: Average turnover time by airport (fallback)
    - expected_turnover_time_airport_month_minutes: Average turnover time by airport and month (temporal fallback)
- expected_turnover_time_airport_minutes: Average turnover time by airport (fallback)
- expected_turnover_time_airport_month_minutes: Average turnover time by airport and month (temporal fallback)
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, to_timestamp, broadcast, avg, count as spark_count
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from datetime import datetime


class ConditionalExpectedValuesMVPModel(Model):
    """Model returned by ConditionalExpectedValuesMVPEstimator after fitting"""
    
    def __init__(self, 
                 lookup_tables,  # Pre-computed lookup tables (dict of DataFrames)
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 dest_col="dest"):
        super(ConditionalExpectedValuesMVPModel, self).__init__()
        self._lookup_tables = lookup_tables
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.dest_col = dest_col
        self._spark = SparkSession.builder.getOrCreate()
    
    def _transform(self, df):
        """Join conditional expected values to input DataFrame."""
        df_with_features = df
        
        # Prepare date components for temporal joins
        df_with_features = df_with_features.withColumn(
            "date",
            to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
        ).withColumn(
            "month",
            F.month("date")
        )
        
        # ============================================================================
        # AIR TIME CONDITIONAL EXPECTED VALUES
        # ============================================================================
        
        # 1. Route-based (non-temporal)
        if 'expected_air_time_route' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_air_time_route']),
                [self.origin_col, self.dest_col],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_air_time_route_minutes': 0.0})
        else:
            df_with_features = df_with_features.withColumn('expected_air_time_route_minutes', F.lit(0.0))
        
        # 2. Route × Month (temporal)
        if 'expected_air_time_route_month' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_air_time_route_month']),
                [self.origin_col, self.dest_col, 'month'],
                'left'
            )
            # Fallback hierarchy: temporal → non-temporal → 0.0
            df_with_features = df_with_features.withColumn(
                "expected_air_time_route_month_minutes",
                F.coalesce(
                    F.col("expected_air_time_route_month_minutes"),
                    F.col("expected_air_time_route_minutes"),
                    F.lit(0.0)
                )
            )
        else:
            df_with_features = df_with_features.withColumn(
                'expected_air_time_route_month_minutes',
                F.coalesce(F.col("expected_air_time_route_minutes"), F.lit(0.0))
            )
        
        # 3. Route-based scheduled elapsed time (non-temporal)
        if 'expected_scheduled_elapsed_time_route' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_scheduled_elapsed_time_route']),
                [self.origin_col, self.dest_col],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_scheduled_elapsed_time_route_minutes': 0.0})
        else:
            df_with_features = df_with_features.withColumn('expected_scheduled_elapsed_time_route_minutes', F.lit(0.0))
        
        # 4. Route × Month scheduled elapsed time (temporal)
        if 'expected_scheduled_elapsed_time_route_month' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_scheduled_elapsed_time_route_month']),
                [self.origin_col, self.dest_col, 'month'],
                'left'
            )
            # Fallback hierarchy: temporal → non-temporal → 0.0
            df_with_features = df_with_features.withColumn(
                "expected_scheduled_elapsed_time_route_month_minutes",
                F.coalesce(
                    F.col("expected_scheduled_elapsed_time_route_month_minutes"),
                    F.col("expected_scheduled_elapsed_time_route_minutes"),
                    F.lit(0.0)
                )
            )
        else:
            df_with_features = df_with_features.withColumn(
                'expected_scheduled_elapsed_time_route_month_minutes',
                F.coalesce(F.col("expected_scheduled_elapsed_time_route_minutes"), F.lit(0.0))
            )
        
        # ============================================================================
        # SCHEDULED ELAPSED TIME CONDITIONAL EXPECTED VALUES
        # ============================================================================
        
        # (Already added above in Air Time section - features 3 and 4)
        
        # ============================================================================
        # TURNOVER TIME CONDITIONAL EXPECTED VALUES
        # ============================================================================
        
        # 5. Carrier-Airport (non-temporal)
        if 'expected_turnover_time_carrier_airport' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_turnover_time_carrier_airport']),
                [self.carrier_col, self.origin_col],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_turnover_time_carrier_airport_minutes': 0.0})
        else:
            df_with_features = df_with_features.withColumn('expected_turnover_time_carrier_airport_minutes', F.lit(0.0))
        
        # 6. Airport-only (non-temporal, fallback)
        if 'expected_turnover_time_airport' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_turnover_time_airport']),
                [self.origin_col],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_turnover_time_airport_minutes': 0.0})
        else:
            df_with_features = df_with_features.withColumn('expected_turnover_time_airport_minutes', F.lit(0.0))
        
        # 7. Airport × Month (temporal, fallback)
        if 'expected_turnover_time_airport_month' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_turnover_time_airport_month']),
                [self.origin_col, 'month'],
                'left'
            )
            # Fallback hierarchy: temporal → non-temporal → 0.0
            df_with_features = df_with_features.withColumn(
                "expected_turnover_time_airport_month_minutes",
                F.coalesce(
                    F.col("expected_turnover_time_airport_month_minutes"),
                    F.col("expected_turnover_time_airport_minutes"),
                    F.lit(0.0)
                )
            )
        else:
            df_with_features = df_with_features.withColumn(
                'expected_turnover_time_airport_month_minutes',
                F.coalesce(F.col("expected_turnover_time_airport_minutes"), F.lit(0.0))
            )
        
        # 8. Carrier-Airport × Month (temporal, most specific)
        if 'expected_turnover_time_carrier_airport_month' in self._lookup_tables:
            df_with_features = df_with_features.join(
                broadcast(self._lookup_tables['expected_turnover_time_carrier_airport_month']),
                [self.carrier_col, self.origin_col, 'month'],
                'left'
            )
            # Fallback hierarchy: carrier-airport-month → carrier-airport → airport-month → airport → 0.0
            df_with_features = df_with_features.withColumn(
                "expected_turnover_time_carrier_airport_month_minutes",
                F.coalesce(
                    F.col("expected_turnover_time_carrier_airport_month_minutes"),
                    F.col("expected_turnover_time_carrier_airport_minutes"),
                    F.col("expected_turnover_time_airport_month_minutes"),
                    F.col("expected_turnover_time_airport_minutes"),
                    F.lit(0.0)
                )
            )
        else:
            # Fallback hierarchy without carrier-airport-month: carrier-airport → airport-month → airport → 0.0
            df_with_features = df_with_features.withColumn(
                'expected_turnover_time_carrier_airport_month_minutes',
                F.coalesce(
                    F.col("expected_turnover_time_carrier_airport_minutes"),
                    F.col("expected_turnover_time_airport_month_minutes"),
                    F.col("expected_turnover_time_airport_minutes"),
                    F.lit(0.0)
                )
            )
        
        # Drop temporary columns
        df_with_features = df_with_features.drop("date", "month")
        
        return df_with_features


class ConditionalExpectedValuesMVPEstimator(Estimator, HasInputCol, HasOutputCol):
    """
    Spark ML Estimator that computes conditional expected values on-the-fly from training data.
    
    Computes conditional means during `.fit()` from training data only (CV-safe).
    Fast: ~30 seconds per fold for simple aggregations.
    No pre-computation step needed - just add to pipeline!
    
    Uses simple averages with month-based temporal features (no Prophet).
    
    Features added:
    - expected_air_time_route_minutes: Average actual air time by route (origin, dest)
    - expected_air_time_route_month_minutes: Average actual air time by route and month (temporal)
    - expected_scheduled_elapsed_time_route_minutes: Average scheduled elapsed time by route (origin, dest)
    - expected_scheduled_elapsed_time_route_month_minutes: Average scheduled elapsed time by route and month (temporal)
    - expected_turnover_time_carrier_airport_minutes: Average turnover time by carrier and airport
    - expected_turnover_time_carrier_airport_month_minutes: Average turnover time by carrier, airport, and month (temporal, most specific)
    - expected_turnover_time_airport_minutes: Average turnover time by airport (fallback)
    - expected_turnover_time_airport_month_minutes: Average turnover time by airport and month (temporal fallback)
    """
    
    def __init__(self,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 dest_col="dest",
                 air_time_col="air_time",
                 min_observations=10):
        super(ConditionalExpectedValuesMVPEstimator, self).__init__()
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.air_time_col = air_time_col
        self.min_observations = min_observations
        self._spark = SparkSession.builder.getOrCreate()
    
    def _fit(self, df):
        """
        Compute conditional expected values from training data.
        
        Note: CV-safe - only uses training data passed to `.fit()`.
        """
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing conditional expected values (MVP) on-the-fly from training data...")
        
        lookup_tables = self._compute_conditional_means(df)
        
        return ConditionalExpectedValuesMVPModel(
            lookup_tables=lookup_tables,
            date_col=self.date_col,
            carrier_col=self.carrier_col,
            origin_col=self.origin_col,
            dest_col=self.dest_col
        )
    
    def _compute_conditional_means(self, train_df):
        """Compute conditional expected values from training data."""
        # Prepare data with date components
        df_prep = (
            train_df
            .withColumn(
                "date",
                to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
            )
            .withColumn(
                "month",
                F.month("date")
            )
            .filter(col("date").isNotNull())
        )
        
        lookup_tables = {}
        
        # 1. Route-based air time (non-temporal)
        if self.air_time_col in df_prep.columns:
            print("  Computing expected air time by route...")
            route_air = (
                df_prep
                .filter(col(self.air_time_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .filter(col(self.dest_col).isNotNull())
                .groupBy(self.origin_col, self.dest_col)
                .agg(
                    avg(self.air_time_col).alias("expected_air_time_route_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_air_time_route'] = route_air
            route_count = route_air.count()
            print(f"    ✓ Generated expected air time for {route_count} routes")
        
        # 2. Route × Month air time (temporal)
        if self.air_time_col in df_prep.columns:
            print("  Computing expected air time by route and month...")
            route_month_air = (
                df_prep
                .filter(col(self.air_time_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .filter(col(self.dest_col).isNotNull())
                .groupBy(self.origin_col, self.dest_col, "month")
                .agg(
                    avg(self.air_time_col).alias("expected_air_time_route_month_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_air_time_route_month'] = route_month_air
            route_month_count = route_month_air.select(self.origin_col, self.dest_col).distinct().count()
            print(f"    ✓ Generated temporal expected air time for {route_month_count} routes")
        
        # 2b. Route-based scheduled elapsed time (non-temporal)
        # Note: crs_elapsed_time is scheduled total elapsed time (includes air + taxi + etc.)
        scheduled_elapsed_col = 'crs_elapsed_time'
        if scheduled_elapsed_col in df_prep.columns:
            print("  Computing expected scheduled elapsed time by route...")
            route_scheduled = (
                df_prep
                .filter(col(scheduled_elapsed_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .filter(col(self.dest_col).isNotNull())
                .groupBy(self.origin_col, self.dest_col)
                .agg(
                    avg(scheduled_elapsed_col).alias("expected_scheduled_elapsed_time_route_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_scheduled_elapsed_time_route'] = route_scheduled
            scheduled_count = route_scheduled.count()
            print(f"    ✓ Generated expected scheduled elapsed time for {scheduled_count} routes")
        
        # 2c. Route × Month scheduled elapsed time (temporal)
        if scheduled_elapsed_col in df_prep.columns:
            print("  Computing expected scheduled elapsed time by route and month...")
            route_month_scheduled = (
                df_prep
                .filter(col(scheduled_elapsed_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .filter(col(self.dest_col).isNotNull())
                .groupBy(self.origin_col, self.dest_col, "month")
                .agg(
                    avg(scheduled_elapsed_col).alias("expected_scheduled_elapsed_time_route_month_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_scheduled_elapsed_time_route_month'] = route_month_scheduled
            scheduled_month_count = route_month_scheduled.select(self.origin_col, self.dest_col).distinct().count()
            print(f"    ✓ Generated temporal expected scheduled elapsed time for {scheduled_month_count} routes")
        
        # 5. Carrier-Airport turnover time (non-temporal)
        # Note: Requires lineage features to be computed first
        turnover_col = None
        for col_name in ['lineage_actual_turnover_time_minutes', 'scheduled_lineage_turnover_time_minutes']:
            if col_name in df_prep.columns:
                turnover_col = col_name
                break
        
        if turnover_col:
            print("  Computing expected turnover time by carrier and airport...")
            carrier_airport_turnover = (
                df_prep
                .filter(col(turnover_col).isNotNull())
                .filter(col(self.carrier_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.carrier_col, self.origin_col)
                .agg(
                    avg(turnover_col).alias("expected_turnover_time_carrier_airport_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_turnover_time_carrier_airport'] = carrier_airport_turnover
            ca_count = carrier_airport_turnover.count()
            print(f"    ✓ Generated expected turnover time for {ca_count} carrier-airport pairs")
        
        # 6. Airport-only turnover time (non-temporal, fallback)
        if turnover_col:
            print("  Computing expected turnover time by airport...")
            airport_turnover = (
                df_prep
                .filter(col(turnover_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.origin_col)
                .agg(
                    avg(turnover_col).alias("expected_turnover_time_airport_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_turnover_time_airport'] = airport_turnover
            airport_count = airport_turnover.count()
            print(f"    ✓ Generated expected turnover time for {airport_count} airports")
        
        # 7. Airport × Month turnover time (temporal, fallback)
        if turnover_col:
            print("  Computing expected turnover time by airport and month...")
            airport_month_turnover = (
                df_prep
                .filter(col(turnover_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.origin_col, "month")
                .agg(
                    avg(turnover_col).alias("expected_turnover_time_airport_month_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_turnover_time_airport_month'] = airport_month_turnover
            airport_month_count = airport_month_turnover.select(self.origin_col).distinct().count()
            print(f"    ✓ Generated temporal expected turnover time for {airport_month_count} airports")
        
        # 8. Carrier-Airport × Month turnover time (temporal, most specific)
        if turnover_col:
            print("  Computing expected turnover time by carrier, airport, and month...")
            carrier_airport_month_turnover = (
                df_prep
                .filter(col(turnover_col).isNotNull())
                .filter(col(self.carrier_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.carrier_col, self.origin_col, "month")
                .agg(
                    avg(turnover_col).alias("expected_turnover_time_carrier_airport_month_minutes"),
                    spark_count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            lookup_tables['expected_turnover_time_carrier_airport_month'] = carrier_airport_month_turnover
            ca_month_count = carrier_airport_month_turnover.select(self.carrier_col, self.origin_col).distinct().count()
            print(f"    ✓ Generated temporal expected turnover time for {ca_month_count} carrier-airport pairs")
        else:
            print("  ⚠ Warning: Turnover time column not found. Skipping turnover time conditional means.")
            print("    (Make sure flight lineage features are computed before this estimator)")
        
        print(f"✓ Conditional expected values computation complete!")
        return lookup_tables