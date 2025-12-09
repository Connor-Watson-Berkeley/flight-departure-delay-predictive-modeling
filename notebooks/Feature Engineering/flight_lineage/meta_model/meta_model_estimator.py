"""
meta_model_estimator.py - Meta-Models for Predicting Previous Flight Components

Spark ML Estimator that trains Random Forest models to predict:
- prev_flight_air_time: Actual air time of previous flight
- prev_flight_taxi_time: Actual taxi time (taxi_in + taxi_out) of previous flight
- prev_flight_total_duration: Actual total elapsed time of previous flight

These predictions are then used as features in the final departure delay model.

CV-Safe: Trained on training fold only, applied to validation/test folds.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, coalesce, lit, when
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from datetime import datetime


class MetaModelModel(Model):
    """Model returned by MetaModelEstimator after fitting"""
    
    def __init__(self, 
                 air_time_pipeline,
                 taxi_time_pipeline,
                 total_duration_pipeline):
        super(MetaModelModel, self).__init__()
        self._air_time_pipeline = air_time_pipeline
        self._taxi_time_pipeline = taxi_time_pipeline
        self._total_duration_pipeline = total_duration_pipeline
        self._spark = SparkSession.builder.getOrCreate()
    
    def _transform(self, df):
        """Apply meta-models to predict previous flight components."""
        df_with_predictions = df
        
        # Predict air time
        if self._air_time_pipeline:
            df_with_predictions = self._air_time_pipeline.transform(df_with_predictions)
            # Rename prediction column (may have multiple predictions, so be careful)
            if "prediction" in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.withColumnRenamed(
                    "prediction", "predicted_prev_flight_air_time"
                )
            # Drop intermediate "features" column to avoid conflicts with next meta-model
            if "features" in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.drop("features")
        
        # Compute taxi time if needed (for taxi_time meta-model or as fallback)
        if "prev_flight_taxi_time" not in df_with_predictions.columns:
            df_with_predictions = df_with_predictions.withColumn(
                "prev_flight_taxi_time",
                coalesce(
                    col("prev_flight_taxi_in"), lit(0.0)
                ) + coalesce(
                    col("prev_flight_taxi_out"), lit(0.0)
                )
            )
        
        # Predict taxi time
        if self._taxi_time_pipeline:
            df_with_predictions = self._taxi_time_pipeline.transform(df_with_predictions)
            # Rename prediction column
            if "prediction" in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.withColumnRenamed(
                    "prediction", "predicted_prev_flight_taxi_time"
                )
            # Drop intermediate "features" column to avoid conflicts with next meta-model
            if "features" in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.drop("features")
        
        # Predict total duration
        if self._total_duration_pipeline:
            # Add target column if needed
            if "prev_flight_total_duration" not in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.withColumn(
                    "prev_flight_total_duration",
                    col("prev_flight_actual_elapsed_time")
                )
            df_with_predictions = self._total_duration_pipeline.transform(df_with_predictions)
            # Rename prediction column
            if "prediction" in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.withColumnRenamed(
                    "prediction", "predicted_prev_flight_total_duration"
                )
            # Drop intermediate "features" column (no more meta-models after this)
            if "features" in df_with_predictions.columns:
                df_with_predictions = df_with_predictions.drop("features")
        
        # Impute NULL predictions (for first flights or jumps)
        df_with_predictions = df_with_predictions.withColumn(
            "predicted_prev_flight_air_time",
            coalesce(
                col("predicted_prev_flight_air_time"),
                col("prev_flight_crs_elapsed_time"),  # Fallback to scheduled
                lit(120.0)  # Default fallback
            )
        )
        df_with_predictions = df_with_predictions.withColumn(
            "predicted_prev_flight_taxi_time",
            coalesce(
                col("predicted_prev_flight_taxi_time"),
                lit(25.0)  # Default fallback (typical taxi time)
            )
        )
        df_with_predictions = df_with_predictions.withColumn(
            "predicted_prev_flight_total_duration",
            coalesce(
                col("predicted_prev_flight_total_duration"),
                col("prev_flight_crs_elapsed_time"),  # Fallback to scheduled
                lit(120.0)  # Default fallback
            )
        )
        
        return df_with_predictions


class MetaModelEstimator(Estimator):
    """
    Spark ML Estimator that trains Random Forest meta-models to predict previous flight components.
    
    Meta-Models:
    1. prev_flight_air_time: Predicts actual air time of previous flight
    2. prev_flight_taxi_time: Predicts actual taxi time (taxi_in + taxi_out) of previous flight
    3. prev_flight_total_duration: Predicts actual total elapsed time of previous flight
    
    CV-Safe: Trained only on training fold data passed to `.fit()`.
    """
    
    def __init__(self,
                 num_trees=50,
                 max_depth=20,  # Increased for high-cardinality categoricals (3000+ routes)
                 min_instances_per_node=20,  # Increased to prevent overfitting with deeper trees
                 use_preprocessed_features=False):
        super(MetaModelEstimator, self).__init__()
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_instances_per_node = min_instances_per_node
        self.use_preprocessed_features = use_preprocessed_features
        self._spark = SparkSession.builder.getOrCreate()
    
    def _fit(self, train_df):
        """
        Train meta-models on training data.
        
        Note: CV-safe - only uses training data passed to `.fit()`.
        If use_preprocessed_features=True, assumes features are already processed.
        """
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Training meta-models for previous flight component prediction...")
        
        if self.use_preprocessed_features:
            print(f"  Using preprocessed features (features should already be imputed, indexed, and encoded)")
        
        # Prepare training data: only rows with previous flight information
        # (exclude first flights and jumps)
        train_with_prev = train_df.filter(
            (col("lineage_rank") > 1) & 
            (~col("lineage_is_jump").cast("boolean"))
        )
        
        print(f"  Training on {train_with_prev.count()} rows with previous flight data")
        
        # Debug: Check what columns are available
        available_cols = set(train_with_prev.columns)
        prev_flight_cols = [col for col in available_cols if col.startswith('prev_flight_')]
        print(f"  Debug: Found {len(prev_flight_cols)} prev_flight_* columns")
        critical_cols = [
            'prev_flight_air_time',
            'prev_flight_taxi_in',
            'prev_flight_taxi_out',
            'prev_flight_actual_elapsed_time'
        ]
        for col_name in critical_cols:
            if col_name in available_cols:
                non_null_count = train_with_prev.filter(col(col_name).isNotNull()).count()
                print(f"    ✓ {col_name}: exists ({non_null_count} non-null rows)")
            else:
                print(f"    ✗ {col_name}: MISSING")
        
        # Train meta-models
        air_time_pipeline = self._train_air_time_model(train_with_prev)
        taxi_time_pipeline = self._train_taxi_time_model(train_with_prev)
        total_duration_pipeline = self._train_total_duration_model(train_with_prev)
        
        # Debug: Count how many actually trained
        trained_count = sum([
            1 if air_time_pipeline is not None else 0,
            1 if taxi_time_pipeline is not None else 0,
            1 if total_duration_pipeline is not None else 0
        ])
        print(f"✓ Meta-model training complete! ({trained_count}/3 models trained)")
        if trained_count < 3:
            print(f"  ⚠ Warning: Only {trained_count} out of 3 meta-models trained successfully.")
            if air_time_pipeline is None:
                print(f"    - Air time meta-model: FAILED")
            if taxi_time_pipeline is None:
                print(f"    - Taxi time meta-model: FAILED")
            if total_duration_pipeline is None:
                print(f"    - Total duration meta-model: FAILED")
        else:
            print(f"  ✓ All 3 meta-models trained successfully:")
            print(f"    - Air time meta-model: ✓")
            print(f"    - Taxi time meta-model: ✓")
            print(f"    - Total duration meta-model: ✓")
        
        return MetaModelModel(
            air_time_pipeline=air_time_pipeline,
            taxi_time_pipeline=taxi_time_pipeline,
            total_duration_pipeline=total_duration_pipeline
        )
    
    def _get_meta_model_features(self, df, target_name):
        """
        Intelligently define features for meta-models based on target variable.
        
        Uses domain knowledge to select the most predictive features for each target:
        - Air Time: Distance (CRITICAL), route (origin→dest), month (jet streams), weather
        - Taxi Time: Time blocks, airport-level features, weather
        - Total Duration: Combination of air + taxi features
        
        Note: Random Forest handles high-cardinality categoricals (3000+ routes) well,
        so we include origin→dest route directly for air_time predictions.
        
        Default max_depth=20 allows trees to make route-specific splits while min_instances_per_node=20
        prevents overfitting. Uses origin + dest + distance to capture route patterns without
        the high cardinality (3000+) of explicit route combinations.
        
        Returns:
        - categorical_features: List of categorical feature names
        - numerical_features: List of numerical feature names
        """
        # Initialize with base features
        categorical_features = []
        numerical_features = []
        
        # Target-specific feature selection
        if target_name == "air_time":
            # AIR TIME: Highly dependent on distance, origin/dest, and seasonal patterns (jet streams)
            # Key insight: Distance + origin + dest can approximate route-specific patterns
            # without the 3000+ cardinality of explicit route combinations
            categorical_features = [
                # Airport features - capture route-specific patterns without high cardinality
                'prev_flight_origin',        # Origin airport (~200 categories) - affects departure routing
                'prev_flight_dest',          # Destination airport (~200 categories) - affects arrival routing
                # Random Forest will learn origin×dest interactions implicitly through tree splits
                
                # Month - CRITICAL for jet streams and seasonal wind patterns
                'prev_flight_month',
                
                # Carrier (minor effect, but some airlines have different speeds/routes)
                'prev_flight_op_carrier',
                
                # State-level (lower cardinality fallback for sparse routes)
                'prev_flight_origin_state_abr',
                'prev_flight_dest_state_abr',
            ]
            
            numerical_features = [
                # DISTANCE - MOST CRITICAL PREDICTOR (captures route length)
                'prev_flight_distance',
                
                # Scheduled elapsed time (correlates with distance)
                'prev_flight_crs_elapsed_time',
                
                # Wind patterns affect air time (jet streams, headwinds/tailwinds)
                # Origin wind at departure
                'prev_flight_origin_hourlywindspeed',
                'prev_flight_origin_hourlywinddirection',
                # Destination wind at arrival
                'prev_flight_dest_hourlywindspeed',
                'prev_flight_dest_hourlywinddirection',
                
                # Weather affecting flight conditions
                'prev_flight_origin_hourlyvisibility',
                'prev_flight_dest_hourlyvisibility',
                'prev_flight_origin_hourlyprecipitation',
                'prev_flight_dest_hourlyprecipitation',
                
                # Pressure/altitude effects
                'prev_flight_origin_hourlysealevelpressure',
                'prev_flight_dest_hourlysealevelpressure',
                
                # Airport elevation (affects takeoff/landing, slight impact on air time)
                'prev_flight_origin_elevation',
                'prev_flight_dest_elevation',
                
                # Temporal features (numerical)
                'prev_flight_day_of_month',  # Minor effect
                'prev_flight_crs_dep_time',  # Time of day (affects traffic/altitude)
            ]
            
        elif target_name == "taxi_time":
            # TAXI TIME: Dependent on airport congestion, time of day, weather
            categorical_features = [
                # Airport-level (taxi time is airport-specific, not route-specific)
                'prev_flight_origin',        # Origin airport (taxi-out)
                'prev_flight_dest',          # Destination airport (taxi-in)
                
                # Time blocks - CRITICAL for congestion patterns
                'prev_flight_dep_time_blk',  # Departure time block (affects taxi-out)
                'prev_flight_arr_time_blk',  # Arrival time block (affects taxi-in)
                
                # Day of week (weekday vs weekend patterns)
                'prev_flight_day_of_week',
                
                # Month (seasonal patterns, holidays)
                'prev_flight_month',
                
                # Carrier (some airlines have priority/preferred gates)
                'prev_flight_op_carrier',
                
                # State-level (lower cardinality)
                'prev_flight_origin_state_abr',
                'prev_flight_dest_state_abr',
            ]
            
            numerical_features = [
                # Weather at airports - affects ground operations
                'prev_flight_origin_hourlyprecipitation',  # Rain/snow slows taxi
                'prev_flight_origin_hourlyvisibility',     # Low visibility slows operations
                'prev_flight_origin_hourlywindspeed',      # High winds affect taxi
                
                'prev_flight_dest_hourlyprecipitation',
                'prev_flight_dest_hourlyvisibility',
                'prev_flight_dest_hourlywindspeed',
                
                # Airport characteristics
                'prev_flight_origin_elevation',
                'prev_flight_dest_elevation',
                
                # Temporal (numerical)
                'prev_flight_day_of_month',
                'prev_flight_crs_dep_time',
                'prev_flight_crs_arr_time',
            ]
            
        elif target_name == "total_duration":
            # TOTAL DURATION: Combination of air time + taxi time
            # Use features from both, but focus on strongest predictors
            categorical_features = [
                # Airport features (air time component) - origin + dest + distance capture route patterns
                'prev_flight_origin',        # Origin airport (~200 categories)
                'prev_flight_dest',          # Destination airport (~200 categories)
                
                # Temporal (affects both air and taxi)
                'prev_flight_month',           # Jet streams (air) + seasonal patterns (taxi)
                'prev_flight_day_of_week',     # Weekday patterns
                'prev_flight_dep_time_blk',    # Time blocks (affects both)
                'prev_flight_arr_time_blk',
                
                # Carrier and location
                'prev_flight_op_carrier',
                'prev_flight_origin_state_abr',
                'prev_flight_dest_state_abr',
            ]
            
            numerical_features = [
                # Distance - CRITICAL for total duration (air time component)
                'prev_flight_distance',
                'prev_flight_crs_elapsed_time',  # Scheduled total time
                
                # Weather (affects both air and taxi)
                'prev_flight_origin_hourlyprecipitation',
                'prev_flight_origin_hourlyvisibility',
                'prev_flight_origin_hourlywindspeed',
                'prev_flight_origin_hourlywinddirection',
                
                'prev_flight_dest_hourlyprecipitation',
                'prev_flight_dest_hourlyvisibility',
                'prev_flight_dest_hourlywindspeed',
                'prev_flight_dest_hourlywinddirection',
                
                # Airport characteristics
                'prev_flight_origin_elevation',
                'prev_flight_dest_elevation',
                
                # Temporal
                'prev_flight_day_of_month',
                'prev_flight_crs_dep_time',
                'prev_flight_crs_arr_time',
            ]
        
        # Filter to only include features that exist in DataFrame
        available_categorical = [f for f in categorical_features if f in df.columns]
        available_numerical = [f for f in numerical_features if f in df.columns]
        
        return available_categorical, available_numerical
    
    def _build_meta_model_pipeline(self, categorical_features, numerical_features, label_col, 
                                   processed_df=None):
        """
        Build a Random Forest pipeline for a meta-model.
        
        If processed_df is provided, assumes features are already processed (imputed, indexed, encoded)
        and uses them directly. Otherwise, processes features internally.
        """
        stages = []
        
        if processed_df is not None and self.use_preprocessed_features:
            # Features are already processed - check which processed columns exist
            # Assume processed features follow naming convention: {col}_IMPUTED for numerical, {col}_VEC for categorical
            available_cols = set(processed_df.columns)
            processed_numerical = [f"{col}_IMPUTED" for col in numerical_features if f"{col}_IMPUTED" in available_cols]
            processed_categorical = [f"{col}_VEC" for col in categorical_features if f"{col}_VEC" in available_cols]
            
            # Assemble processed features
            assembler = VectorAssembler(
                inputCols=processed_categorical + processed_numerical,
                outputCol="features",
                handleInvalid="skip"
            )
            stages.append(assembler)
        else:
            # Process features internally (original behavior)
            # Impute numerical features
            if numerical_features:
                imputer = Imputer(
                    inputCols=numerical_features,
                    outputCols=[f"{col}_IMPUTED" for col in numerical_features],
                    strategy="mean"
                )
                stages.append(imputer)
                imputed_numerical = [f"{col}_IMPUTED" for col in numerical_features]
            else:
                imputed_numerical = []
            
            # Index categorical features
            if categorical_features:
                indexer = StringIndexer(
                    inputCols=categorical_features,
                    outputCols=[f"{col}_INDEX" for col in categorical_features],
                    handleInvalid="keep"
                )
                stages.append(indexer)
                
                # One-hot encode
                encoder = OneHotEncoder(
                    inputCols=[f"{col}_INDEX" for col in categorical_features],
                    outputCols=[f"{col}_VEC" for col in categorical_features],
                    dropLast=False
                )
                stages.append(encoder)
                encoded_categorical = [f"{col}_VEC" for col in categorical_features]
            else:
                encoded_categorical = []
            
            # Assemble features
            assembler = VectorAssembler(
                inputCols=encoded_categorical + imputed_numerical,
                outputCol="features",
                handleInvalid="skip"
            )
            stages.append(assembler)
        
        # Random Forest Regressor
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol=label_col,
            numTrees=self.num_trees,
            maxDepth=self.max_depth,
            minInstancesPerNode=self.min_instances_per_node,
            seed=42
        )
        stages.append(rf)
        
        return Pipeline(stages=stages)
    
    def _train_air_time_model(self, train_df):
        """Train meta-model to predict prev_flight_air_time."""
        print("  Training meta-model: prev_flight_air_time")
        
        # Check if target exists
        if "prev_flight_air_time" not in train_df.columns:
            print("    ⚠ Warning: prev_flight_air_time not found. Skipping air time meta-model.")
            return None
        
        # Filter to rows with valid target
        train_air_time = train_df.filter(col("prev_flight_air_time").isNotNull())
        count = train_air_time.count()
        
        if count < 100:
            print(f"    ⚠ Warning: Only {count} rows with prev_flight_air_time. Skipping.")
            return None
        
        print(f"    Training on {count} rows")
        
        # Get features
        categorical_features, numerical_features = self._get_meta_model_features(
            train_air_time, "air_time"
        )
        
        print(f"    Using {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        
        # Build and fit pipeline
        pipeline = self._build_meta_model_pipeline(
            categorical_features, 
            numerical_features, 
            "prev_flight_air_time",
            processed_df=train_air_time if self.use_preprocessed_features else None
        )
        
        fitted_pipeline = pipeline.fit(train_air_time)
        print(f"    ✓ Air time meta-model trained")
        
        return fitted_pipeline
    
    def _train_taxi_time_model(self, train_df):
        """Train meta-model to predict prev_flight_taxi_time (taxi_in + taxi_out)."""
        print("  Training meta-model: prev_flight_taxi_time")
        
        # Compute taxi time if not already computed
        if "prev_flight_taxi_time" not in train_df.columns:
            train_df = train_df.withColumn(
                "prev_flight_taxi_time",
                coalesce(
                    col("prev_flight_taxi_in"), lit(0.0)
                ) + coalesce(
                    col("prev_flight_taxi_out"), lit(0.0)
                )
            )
        
        # Filter to rows with valid target
        train_taxi_time = train_df.filter(col("prev_flight_taxi_time").isNotNull())
        count = train_taxi_time.count()
        
        if count < 100:
            print(f"    ⚠ Warning: Only {count} rows with prev_flight_taxi_time. Skipping.")
            return None
        
        print(f"    Training on {count} rows")
        
        # Get features
        categorical_features, numerical_features = self._get_meta_model_features(
            train_taxi_time, "taxi_time"
        )
        
        print(f"    Using {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        
        # Build and fit pipeline
        pipeline = self._build_meta_model_pipeline(
            categorical_features, 
            numerical_features, 
            "prev_flight_taxi_time",
            processed_df=train_taxi_time if self.use_preprocessed_features else None
        )
        
        fitted_pipeline = pipeline.fit(train_taxi_time)
        print(f"    ✓ Taxi time meta-model trained")
        
        return fitted_pipeline
    
    def _train_total_duration_model(self, train_df):
        """Train meta-model to predict prev_flight_total_duration (actual_elapsed_time)."""
        print("  Training meta-model: prev_flight_total_duration")
        
        # Use actual_elapsed_time as target
        if "prev_flight_actual_elapsed_time" not in train_df.columns:
            print("    ⚠ Warning: prev_flight_actual_elapsed_time not found. Skipping total duration meta-model.")
            return None
        
        # Filter to rows with valid target
        train_total_duration = train_df.filter(col("prev_flight_actual_elapsed_time").isNotNull())
        count = train_total_duration.count()
        
        if count < 100:
            print(f"    ⚠ Warning: Only {count} rows with prev_flight_actual_elapsed_time. Skipping.")
            return None
        
        print(f"    Training on {count} rows")
        
        # Temporarily rename for pipeline
        train_total_duration = train_total_duration.withColumn(
            "prev_flight_total_duration",
            col("prev_flight_actual_elapsed_time")
        )
        
        # Get features
        categorical_features, numerical_features = self._get_meta_model_features(
            train_total_duration, "total_duration"
        )
        
        print(f"    Using {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        
        # Build and fit pipeline
        pipeline = self._build_meta_model_pipeline(
            categorical_features, 
            numerical_features, 
            "prev_flight_total_duration",
            processed_df=train_total_duration if self.use_preprocessed_features else None
        )
        
        fitted_pipeline = pipeline.fit(train_total_duration)
        print(f"    ✓ Total duration meta-model trained")
        
        return fitted_pipeline
