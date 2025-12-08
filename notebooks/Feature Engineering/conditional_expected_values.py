"""
conditional_expected_values.py - Conditional expected value feature engineering for flight delay prediction

Provides ConditionalExpectedValuesEstimator for use in Spark ML pipelines.
Generates conditional expected values for air time and turnover time using time-series models
(Prophet) and aggregations conditional on route, carrier, airport, time, aircraft type, etc.

Key Strategy: Multiple Conditional Means
- Compute expected values in multiple different ways (different conditional means)
- Provide all conditional means to the model as features
- Model learns to weight between them based on predictive power
- Weather can be incorporated into conditional means (e.g., route + weather, carrier + airport + weather)

Key Features:
- Expected air time conditional on route, time of year (jet streams), aircraft type, time of day
- Expected turnover time conditional on carrier, airport, time of day, day of week
- Non-temporal conditional means (aircraft, airport, carrier only)
- Future: Weather-conditioned expected values (wind, precipitation affect both air_time and turnover_time)
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, to_timestamp, when, avg, count
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
import pandas as pd
from datetime import datetime

# Suppress verbose cmdstanpy output BEFORE importing Prophet
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

from prophet import Prophet


class ConditionalExpectedValuesModel(Model):
    """Model returned by ConditionalExpectedValuesEstimator after fitting"""
    
    def __init__(self, 
                 version=None,
                 fold_index=None,
                 lookup_base_path=None,
                 expected_air_time_route=None,
                 expected_turnover_time_carrier_airport=None,
                 expected_air_time_route_temporal=None,
                 expected_turnover_time_temporal=None,
                 expected_air_time_aircraft=None,
                 expected_turnover_time_aircraft=None,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 dest_col="dest"):
        super(ConditionalExpectedValuesModel, self).__init__()
        self.version = version
        self.fold_index = fold_index
        self.lookup_base_path = lookup_base_path
        self.expected_air_time_route = expected_air_time_route
        self.expected_turnover_time_carrier_airport = expected_turnover_time_carrier_airport
        self.expected_air_time_route_temporal = expected_air_time_route_temporal
        self.expected_turnover_time_temporal = expected_turnover_time_temporal
        self.expected_air_time_aircraft = expected_air_time_aircraft
        self.expected_turnover_time_aircraft = expected_turnover_time_aircraft
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.dest_col = dest_col
        self._spark = SparkSession.builder.getOrCreate()
        self._lookup_tables = None  # Cache for lookup tables
    
    def _load_lookup_tables(self):
        """Lazy load lookup tables from parquet files."""
        if self._lookup_tables is not None:
            return self._lookup_tables
        
        if not (self.lookup_base_path and self.version and self.fold_index is not None):
            # Not using lookup tables, return None
            return None
        
        # Convert 0-based fold_index to 1-based for file path (files are named fold_1, fold_2, etc.)
        fold_index_1based = self.fold_index + 1
        fold_path = f"{self.lookup_base_path}/{self.version}/fold_{fold_index_1based}"
        
        try:
            self._lookup_tables = {
                'expected_air_time_route': self._spark.read.parquet(f"{fold_path}/expected_air_time_route.parquet"),
                'expected_air_time_route_temporal': self._spark.read.parquet(f"{fold_path}/expected_air_time_route_temporal.parquet"),
                'expected_air_time_aircraft': self._spark.read.parquet(f"{fold_path}/expected_air_time_aircraft.parquet"),
                'expected_turnover_time_carrier_airport': self._spark.read.parquet(f"{fold_path}/expected_turnover_time_carrier_airport.parquet"),
                'expected_turnover_time_temporal': self._spark.read.parquet(f"{fold_path}/expected_turnover_time_temporal.parquet"),
                'expected_turnover_time_aircraft': self._spark.read.parquet(f"{fold_path}/expected_turnover_time_aircraft.parquet"),
            }
            return self._lookup_tables
        except Exception as e:
            raise ValueError(
                f"Failed to load lookup tables from {fold_path}. "
                f"Error: {str(e)}\n"
                f"Please ensure pre-computation has been run for {self.version}/fold_{fold_index_1based}."
            )
    
    def _transform(self, df):
        """Join conditional expected values to input DataFrame"""
        # Try to load from lookup tables first
        lookup_tables = self._load_lookup_tables()
        
        if lookup_tables is not None:
            return self._transform_from_lookup_tables(df, lookup_tables)
        else:
            # Fall back to in-memory pandas DataFrames (backward compatibility)
            return self._transform_from_memory(df)
    
    def _transform_from_lookup_tables(self, df, lookup_tables):
        """Join pre-computed lookup tables to DataFrame."""
        df_with_features = df
        
        # Add date_str column if needed for temporal joins
        if 'date_str' not in df_with_features.columns:
            df_with_features = df_with_features.withColumn(
                "date_str",
                F.date_format(
                    F.to_date(F.col(self.date_col), "yyyy-MM-dd"),
                    "yyyy-MM-dd"
                )
            )
        
        # Join non-temporal expected air time by route
        if 'expected_air_time_route' in lookup_tables:
            df_with_features = df_with_features.join(
                lookup_tables['expected_air_time_route'],
                [self.origin_col, self.dest_col],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_air_time_route_minutes': 0.0})
        
        # Join temporal expected air time by route and date
        if 'expected_air_time_route_temporal' in lookup_tables:
            df_with_features = df_with_features.join(
                lookup_tables['expected_air_time_route_temporal'],
                [self.origin_col, self.dest_col, 'date_str'],
                'left'
            )
            # Fallback to non-temporal if temporal not available
            df_with_features = df_with_features.withColumn(
                "expected_air_time_route_temporal_minutes",
                F.coalesce(
                    F.col("expected_air_time_route_temporal_minutes"),
                    F.col("expected_air_time_route_minutes"),
                    F.lit(0.0)
                )
            )
        
        # Join non-temporal expected turnover time by carrier and airport
        if 'expected_turnover_time_carrier_airport' in lookup_tables:
            df_with_features = df_with_features.join(
                lookup_tables['expected_turnover_time_carrier_airport'],
                [self.carrier_col, self.origin_col],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_turnover_time_carrier_airport_minutes': 0.0})
        
        # Join temporal expected turnover time by carrier, airport, and date
        if 'expected_turnover_time_temporal' in lookup_tables:
            df_with_features = df_with_features.join(
                lookup_tables['expected_turnover_time_temporal'],
                [self.carrier_col, self.origin_col, 'date_str'],
                'left'
            )
            # Fallback to non-temporal if temporal not available
            df_with_features = df_with_features.withColumn(
                "expected_turnover_time_temporal_minutes",
                F.coalesce(
                    F.col("expected_turnover_time_temporal_minutes"),
                    F.col("expected_turnover_time_carrier_airport_minutes"),
                    F.lit(0.0)
                )
            )
        
        # Join aircraft-based expected values if available
        if 'expected_air_time_aircraft' in lookup_tables:
            df_with_features = df_with_features.join(
                lookup_tables['expected_air_time_aircraft'],
                ['tail_num'],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_air_time_aircraft_minutes': 0.0})
        
        if 'expected_turnover_time_aircraft' in lookup_tables:
            df_with_features = df_with_features.join(
                lookup_tables['expected_turnover_time_aircraft'],
                ['tail_num'],
                'left'
            )
            df_with_features = df_with_features.fillna({'expected_turnover_time_aircraft_minutes': 0.0})
        
        return df_with_features
    
    def _transform_from_memory(self, df):
        """Join conditional expected values from in-memory pandas DataFrames (backward compatibility)."""
        df_with_features = df
        
        # Join non-temporal expected air time by route
        if self.expected_air_time_route is not None:
            route_spark = self._spark.createDataFrame(self.expected_air_time_route)
            df_with_features = df_with_features.join(
                route_spark,
                [self.origin_col, self.dest_col],
                'left'
            )
            if 'expected_air_time_route_minutes' in df_with_features.columns:
                df_with_features = df_with_features.fillna({'expected_air_time_route_minutes': 0.0})
        
        # Join temporal expected air time by route and date
        if self.expected_air_time_route_temporal is not None:
            route_temporal_spark = self._spark.createDataFrame(self.expected_air_time_route_temporal)
            df_with_features = df_with_features.join(
                route_temporal_spark,
                [self.origin_col, self.dest_col, 'date_str'],
                'left'
            )
            if 'expected_air_time_route_temporal_minutes' in df_with_features.columns:
                df_with_features = df_with_features.fillna({'expected_air_time_route_temporal_minutes': 0.0})
        
        # Join non-temporal expected turnover time by carrier and airport
        if self.expected_turnover_time_carrier_airport is not None:
            turnover_spark = self._spark.createDataFrame(self.expected_turnover_time_carrier_airport)
            df_with_features = df_with_features.join(
                turnover_spark,
                [self.carrier_col, self.origin_col],
                'left'
            )
            if 'expected_turnover_time_carrier_airport_minutes' in df_with_features.columns:
                df_with_features = df_with_features.fillna({'expected_turnover_time_carrier_airport_minutes': 0.0})
        
        # Join temporal expected turnover time by carrier, airport, and date
        if self.expected_turnover_time_temporal is not None:
            turnover_temporal_spark = self._spark.createDataFrame(self.expected_turnover_time_temporal)
            df_with_features = df_with_features.join(
                turnover_temporal_spark,
                [self.carrier_col, self.origin_col, 'date_str'],
                'left'
            )
            if 'expected_turnover_time_temporal_minutes' in df_with_features.columns:
                df_with_features = df_with_features.fillna({'expected_turnover_time_temporal_minutes': 0.0})
        
        # Join aircraft-based expected values if available
        if self.expected_air_time_aircraft is not None:
            aircraft_air_spark = self._spark.createDataFrame(self.expected_air_time_aircraft)
            df_with_features = df_with_features.join(
                aircraft_air_spark,
                ['tail_num'],
                'left'
            )
            if 'expected_air_time_aircraft_minutes' in df_with_features.columns:
                df_with_features = df_with_features.fillna({'expected_air_time_aircraft_minutes': 0.0})
        
        if self.expected_turnover_time_aircraft is not None:
            aircraft_turnover_spark = self._spark.createDataFrame(self.expected_turnover_time_aircraft)
            df_with_features = df_with_features.join(
                aircraft_turnover_spark,
                ['tail_num'],
                'left'
            )
            if 'expected_turnover_time_aircraft_minutes' in df_with_features.columns:
                df_with_features = df_with_features.fillna({'expected_turnover_time_aircraft_minutes': 0.0})
        
        return df_with_features


class ConditionalExpectedValuesEstimator(Estimator, HasInputCol, HasOutputCol):
    """
    Spark ML Estimator that adds conditional expected values for air time and turnover time.
    
    Strategy: Multiple Conditional Means
    - Computes expected values in multiple different ways (different conditional means)
    - All conditional means are provided as features - model learns to weight between them
    - Different conditional means are informative in different scenarios:
      * Route-based: Good baseline, captures distance/routing
      * Temporal (Prophet): Captures seasonal patterns, jet streams, time-of-day effects
      * Aircraft-based: Captures performance characteristics
      * Future: Weather-conditioned (wind, precipitation affect both air_time and turnover_time)
    
    In fit(): Loads pre-computed conditional expected values from lookup tables (based on version and fold_index)
    Returns a ConditionalExpectedValuesModel that can transform DataFrames
    
    Features added:
    - expected_air_time_route_minutes: Average air time for route (origin, dest) - baseline conditional
    - expected_air_time_route_temporal_minutes: Temporal expected air time (route + date, Prophet) - captures seasonal/jet stream effects
    - expected_air_time_aircraft_minutes: Average air time by aircraft - captures performance differences
    - expected_turnover_time_carrier_airport_minutes: Average turnover time (carrier, airport) - baseline conditional
    - expected_turnover_time_temporal_minutes: Temporal expected turnover time (carrier, airport, date, Prophet) - captures time-of-day/week effects
    - expected_turnover_time_aircraft_minutes: Average turnover time by aircraft - captures aircraft-specific turn times
    
    Future Enhancements:
    - Weather-conditioned expected values (route + weather, carrier + airport + weather)
    - Combined conditional means (route + time + weather + aircraft)
    - Previous departure to next departure conditional means (lower variance alternative)
    """
    
    # Spark ML Params for fold_index and version
    version = Param(Params._dummy(), "version", "Data version: '3M', '12M', or '60M'", TypeConverters.toString)
    fold_index = Param(Params._dummy(), "fold_index", "Fold index (0-based: 0, 1, 2, or 3)", TypeConverters.toInt)
    lookup_base_path = Param(Params._dummy(), "lookup_base_path", "Base path for conditional expected value lookup tables", TypeConverters.toString)
    
    def __init__(self,
                 version=None,
                 fold_index=None,
                 lookup_base_path=None,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 dest_col="dest",
                 air_time_col="air_time",
                 min_observations=10,
                 min_days_required=14,
                 changepoint_prior_scale=0.05,
                 use_prophet=True):
        super(ConditionalExpectedValuesEstimator, self).__init__()
        
        # Set Spark ML params
        if version is not None:
            self._setDefault(version=version)
        if fold_index is not None:
            self._setDefault(fold_index=fold_index)
        if lookup_base_path is not None:
            self._setDefault(lookup_base_path=lookup_base_path)
        
        # Regular attributes (for backward compatibility if not using params)
        self._version = version
        self._fold_index = fold_index
        self._lookup_base_path = lookup_base_path
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.air_time_col = air_time_col
        self.min_observations = min_observations
        self.min_days_required = min_days_required
        self.changepoint_prior_scale = changepoint_prior_scale
        self.use_prophet = use_prophet
        self._spark = SparkSession.builder.getOrCreate()
    
    def setVersion(self, value):
        """Set the data version."""
        return self._set(version=value)
    
    def getVersion(self):
        """Get the data version."""
        if self.isSet("version"):
            return self.getOrDefault("version")
        return self._version
    
    def setFoldIndex(self, value):
        """
        Set the fold index dynamically (called by FlightDelayCV before each fit).
        
        Args:
            value: Fold index (0-based: 0, 1, 2, or 3)
                   - 0, 1, 2: CV folds
                   - 3: Test fold
        """
        return self._set(fold_index=value)
    
    def getFoldIndex(self):
        """Get the fold index."""
        if self.isSet("fold_index"):
            return self.getOrDefault("fold_index")
        return self._fold_index
    
    def setLookupBasePath(self, value):
        """Set the base path for lookup tables."""
        return self._set(lookup_base_path=value)
    
    def getLookupBasePath(self):
        """Get the base path for lookup tables."""
        if self.isSet("lookup_base_path"):
            return self.getOrDefault("lookup_base_path")
        return self._lookup_base_path
    
    def _fit_prophet_model(self, data, y_col='y'):
        """Fit a Prophet model on time series data."""
        if len(data) < self.min_days_required:
            return None
        
        prophet_data = data[['ds', y_col]].copy()
        prophet_data = prophet_data.rename(columns={y_col: 'y'})
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) < self.min_days_required:
            return None
        
        has_enough_for_yearly = len(prophet_data) >= 365
        has_enough_for_weekly = len(prophet_data) >= 14
        
        prophet_model = Prophet(
            yearly_seasonality=has_enough_for_yearly,
            weekly_seasonality=has_enough_for_weekly,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        prophet_model.fit(prophet_data)
        
        forecast = prophet_model.predict(prophet_data[['ds']])
        return forecast
    
    def _fit(self, df):
        """
        Load pre-computed conditional expected values from lookup tables.
        
        Note: If version/fold_index/lookup_base_path are not set, this will attempt
        to compute conditional values on-the-fly (backward compatibility mode).
        """
        # Get version, fold_index, and lookup_base_path (from params or attributes)
        version = self.getVersion()
        fold_index = self.getFoldIndex()
        lookup_base_path = self.getLookupBasePath()
        
        # If lookup_base_path is set, use pre-computed lookup tables
        # Note: Check fold_index is not None (not just truthy) since fold_index=0 is valid
        if lookup_base_path and version and fold_index is not None:
            return self._fit_from_lookup_tables(df, version, fold_index, lookup_base_path)
        else:
            # Backward compatibility: compute on-the-fly
            return self._fit_compute_on_the_fly(df)
    
    def _fit_from_lookup_tables(self, df, version, fold_index, lookup_base_path):
        """Load pre-computed conditional expected values from lookup tables."""
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        # Convert 0-based to 1-based for display (files use 1-based naming)
        fold_index_1based = fold_index + 1
        print(f"[{timestamp}] Loading pre-computed conditional expected values for {version}/fold_{fold_index_1based}...")
        
        # Verify lookup tables exist (will raise error if not found)
        # The actual loading happens lazily in _transform
        
        return ConditionalExpectedValuesModel(
            version=version,
            fold_index=fold_index,
            lookup_base_path=lookup_base_path,
            date_col=self.date_col,
            carrier_col=self.carrier_col,
            origin_col=self.origin_col,
            dest_col=self.dest_col
        )
    
    def _fit_compute_on_the_fly(self, df):
        """Compute conditional expected values on-the-fly (backward compatibility)."""
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Generating conditional expected values on-the-fly...")
        
        # Prepare date column
        df_prep = df.withColumn(
            "date",
            to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
        ).withColumn(
            "date_str",
            F.date_format(col("date"), "yyyy-MM-dd")
        ).filter(
            col("date").isNotNull()
        )
        
        expected_air_time_route = None
        expected_turnover_time_carrier_airport = None
        expected_air_time_route_temporal = None
        expected_turnover_time_temporal = None
        expected_air_time_aircraft = None
        expected_turnover_time_aircraft = None
        
        # 1. Non-temporal expected air time by route
        print("  Computing expected air time by route...")
        if self.air_time_col in df_prep.columns:
            route_air = (
                df_prep
                .filter(col(self.air_time_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .filter(col(self.dest_col).isNotNull())
                .groupBy(self.origin_col, self.dest_col)
                .agg(
                    avg(self.air_time_col).alias("expected_air_time_route_minutes"),
                    count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            route_air_pd = route_air.toPandas()
            if len(route_air_pd) > 0:
                expected_air_time_route = route_air_pd
                print(f"    ✓ Generated expected air time for {len(route_air_pd)} routes")
        
        # 2. Temporal expected air time by route (Prophet)
        if self.use_prophet and self.air_time_col in df_prep.columns:
            print("  Computing temporal expected air time by route (Prophet)...")
            route_temporal_spark = (
                df_prep
                .filter(col(self.air_time_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .filter(col(self.dest_col).isNotNull())
                .groupBy(self.origin_col, self.dest_col, "date")
                .agg(avg(self.air_time_col).alias("avg_air_time"))
                .orderBy(self.origin_col, self.dest_col, "date")
            )
            route_temporal = route_temporal_spark.toPandas()
            
            if len(route_temporal) > 0:
                route_temporal['ds'] = pd.to_datetime(route_temporal['date'])
                route_temporal_list = []
                
                for (origin, dest), group in route_temporal.groupby([self.origin_col, self.dest_col]):
                    if len(group) >= self.min_days_required:
                        forecast = self._fit_prophet_model(group, 'avg_air_time')
                        if forecast is not None:
                            forecast[self.origin_col] = origin
                            forecast[self.dest_col] = dest
                            forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')
                            route_temporal_list.append(forecast[['origin', 'dest', 'date_str', 'yhat']])
                
                if route_temporal_list:
                    expected_air_time_route_temporal = pd.concat(route_temporal_list, ignore_index=True)
                    expected_air_time_route_temporal = expected_air_time_route_temporal.rename(
                        columns={'yhat': 'expected_air_time_route_temporal_minutes'}
                    )
                    print(f"    ✓ Generated temporal expected air time for {expected_air_time_route_temporal[[self.origin_col, self.dest_col]].drop_duplicates().shape[0]} routes")
        
        # 3. Non-temporal expected turnover time by carrier and airport
        print("  Computing expected turnover time by carrier and airport...")
        # Note: Turnover time needs to be computed from lineage features
        # For now, we'll compute it if lineage features are available
        if 'lineage_turnover_time_minutes' in df_prep.columns:
            turnover_spark = (
                df_prep
                .filter(col('lineage_turnover_time_minutes').isNotNull())
                .filter(col(self.carrier_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.carrier_col, self.origin_col)
                .agg(
                    avg('lineage_turnover_time_minutes').alias("expected_turnover_time_carrier_airport_minutes"),
                    count("*").alias("count")
                )
                .filter(col("count") >= self.min_observations)
                .drop("count")
            )
            turnover_pd = turnover_spark.toPandas()
            if len(turnover_pd) > 0:
                expected_turnover_time_carrier_airport = turnover_pd
                print(f"    ✓ Generated expected turnover time for {len(turnover_pd)} carrier-airport pairs")
        
        # 4. Temporal expected turnover time (Prophet)
        if self.use_prophet and 'lineage_turnover_time_minutes' in df_prep.columns:
            print("  Computing temporal expected turnover time (Prophet)...")
            turnover_temporal_spark = (
                df_prep
                .filter(col('lineage_turnover_time_minutes').isNotNull())
                .filter(col(self.carrier_col).isNotNull())
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.carrier_col, self.origin_col, "date")
                .agg(avg('lineage_turnover_time_minutes').alias("avg_turnover_time"))
                .orderBy(self.carrier_col, self.origin_col, "date")
            )
            turnover_temporal = turnover_temporal_spark.toPandas()
            
            if len(turnover_temporal) > 0:
                turnover_temporal['ds'] = pd.to_datetime(turnover_temporal['date'])
                turnover_temporal_list = []
                
                for (carrier, airport), group in turnover_temporal.groupby([self.carrier_col, self.origin_col]):
                    if len(group) >= self.min_days_required:
                        forecast = self._fit_prophet_model(group, 'avg_turnover_time')
                        if forecast is not None:
                            forecast[self.carrier_col] = carrier
                            forecast[self.origin_col] = airport
                            forecast['date_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')
                            turnover_temporal_list.append(forecast[[self.carrier_col, self.origin_col, 'date_str', 'yhat']])
                
                if turnover_temporal_list:
                    expected_turnover_time_temporal = pd.concat(turnover_temporal_list, ignore_index=True)
                    expected_turnover_time_temporal = expected_turnover_time_temporal.rename(
                        columns={'yhat': 'expected_turnover_time_temporal_minutes'}
                    )
                    print(f"    ✓ Generated temporal expected turnover time for {expected_turnover_time_temporal[[self.carrier_col, self.origin_col]].drop_duplicates().shape[0]} carrier-airport pairs")
        
        # 5. Aircraft-based expected values (optional, if tail_num available)
        if 'tail_num' in df_prep.columns:
            print("  Computing aircraft-based expected values...")
            if self.air_time_col in df_prep.columns:
                aircraft_air = (
                    df_prep
                    .filter(col(self.air_time_col).isNotNull())
                    .filter(col('tail_num').isNotNull())
                    .groupBy('tail_num')
                    .agg(
                        avg(self.air_time_col).alias("expected_air_time_aircraft_minutes"),
                        count("*").alias("count")
                    )
                    .filter(col("count") >= self.min_observations)
                    .drop("count")
                )
                aircraft_air_pd = aircraft_air.toPandas()
                if len(aircraft_air_pd) > 0:
                    expected_air_time_aircraft = aircraft_air_pd
                    print(f"    ✓ Generated expected air time for {len(aircraft_air_pd)} aircraft")
            
            if 'lineage_turnover_time_minutes' in df_prep.columns:
                aircraft_turnover = (
                    df_prep
                    .filter(col('lineage_turnover_time_minutes').isNotNull())
                    .filter(col('tail_num').isNotNull())
                    .groupBy('tail_num')
                    .agg(
                        avg('lineage_turnover_time_minutes').alias("expected_turnover_time_aircraft_minutes"),
                        count("*").alias("count")
                    )
                    .filter(col("count") >= self.min_observations)
                    .drop("count")
                )
                aircraft_turnover_pd = aircraft_turnover.toPandas()
                if len(aircraft_turnover_pd) > 0:
                    expected_turnover_time_aircraft = aircraft_turnover_pd
                    print(f"    ✓ Generated expected turnover time for {len(aircraft_turnover_pd)} aircraft")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Conditional expected value generation complete! (took {duration})")
        
        return ConditionalExpectedValuesModel(
            expected_air_time_route=expected_air_time_route,
            expected_turnover_time_carrier_airport=expected_turnover_time_carrier_airport,
            expected_air_time_route_temporal=expected_air_time_route_temporal,
            expected_turnover_time_temporal=expected_turnover_time_temporal,
            expected_air_time_aircraft=expected_air_time_aircraft,
            expected_turnover_time_aircraft=expected_turnover_time_aircraft,
            date_col=self.date_col,
            carrier_col=self.carrier_col,
            origin_col=self.origin_col,
            dest_col=self.dest_col,
            version=None,  # Not using lookup tables
            fold_index=None,
            lookup_base_path=None
        )

