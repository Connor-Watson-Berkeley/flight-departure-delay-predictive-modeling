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
                 max_depth=10,
                 min_instances_per_node=10,
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
        
        # Train meta-models
        air_time_pipeline = self._train_air_time_model(train_with_prev)
        taxi_time_pipeline = self._train_taxi_time_model(train_with_prev)
        total_duration_pipeline = self._train_total_duration_model(train_with_prev)
        
        print(f"✓ Meta-model training complete!")
        
        return MetaModelModel(
            air_time_pipeline=air_time_pipeline,
            taxi_time_pipeline=taxi_time_pipeline,
            total_duration_pipeline=total_duration_pipeline
        )
    
    def _get_meta_model_features(self, df, target_name):
        """
        Define features for meta-models based on target variable.
        
        Uses comprehensive covariates as described in CONDITIONAL_EXPECTED_VALUES_DESIGN.md:
        - Weather variables (previous flight origin/dest)
        - Temporal features (month, day_of_week, hour_of_day, day_of_month)
        - Route characteristics (previous flight route)
        - Carrier and location (state-level)
        - Scheduled flight characteristics
        
        Returns:
        - categorical_features: List of categorical feature names
        - numerical_features: List of numerical feature names
        """
        # Categorical features
        categorical_features = [
            # Carrier and location (state-level to avoid high cardinality)
            'prev_flight_op_carrier',
            'prev_flight_origin_state_abr',
            'prev_flight_dest_state_abr',
            # Temporal features
            'prev_flight_day_of_week',
            'prev_flight_month',
        ]
        
        # Numerical features - comprehensive set
        numerical_features = [
            # Previous flight scheduled characteristics
            'prev_flight_crs_elapsed_time',
            'prev_flight_distance',
            'prev_flight_crs_dep_time',  # Hour/minute as numerical
            'prev_flight_crs_arr_time',  # Hour/minute as numerical
            
            # Weather at previous flight origin (comprehensive set)
            'prev_flight_origin_hourlyprecipitation',
            'prev_flight_origin_hourlysealevelpressure',
            'prev_flight_origin_hourlyaltimetersetting',
            'prev_flight_origin_hourlywetbulbtemperature',
            'prev_flight_origin_hourlystationpressure',
            'prev_flight_origin_hourlywinddirection',
            'prev_flight_origin_hourlyrelativehumidity',
            'prev_flight_origin_hourlywindspeed',
            'prev_flight_origin_hourlydewpointtemperature',
            'prev_flight_origin_hourlydrybulbtemperature',
            'prev_flight_origin_hourlyvisibility',
            
            # Weather at previous flight destination
            'prev_flight_dest_hourlyprecipitation',
            'prev_flight_dest_hourlysealevelpressure',
            'prev_flight_dest_hourlyaltimetersetting',
            'prev_flight_dest_hourlywetbulbtemperature',
            'prev_flight_dest_hourlystationpressure',
            'prev_flight_dest_hourlywinddirection',
            'prev_flight_dest_hourlyrelativehumidity',
            'prev_flight_dest_hourlywindspeed',
            'prev_flight_dest_hourlydewpointtemperature',
            'prev_flight_dest_hourlydrybulbtemperature',
            'prev_flight_dest_hourlyvisibility',
            
            # Airport characteristics (if available)
            'prev_flight_origin_elevation',
            'prev_flight_dest_elevation',
            
            # Temporal features (numerical)
            'prev_flight_day_of_month',
            # Note: hour_of_day can be derived from crs_dep_time or dep_time_blk
        ]
        
        # Target-specific features
        if target_name == "air_time":
            # Air time specific: route characteristics
            # Could add route-level aggregations if available
            pass
        elif target_name == "taxi_time":
            # Taxi time specific: time blocks for congestion
            categorical_features.extend([
                'prev_flight_dep_time_blk',
                'prev_flight_arr_time_blk',
            ])
            # Add airport congestion indicators if available
        elif target_name == "total_duration":
            # Total duration: combination of air and taxi
            # Could add both air_time and taxi_time components if predicting separately
            pass
        
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
