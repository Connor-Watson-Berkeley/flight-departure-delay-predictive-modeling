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


# -------------------------
# META-MODEL IDENTIFIERS
# -------------------------
# Model IDs for precomputed meta-model predictions
# Format: predicted_{target}_{MODEL_ID}
# Allows multiple models to predict the same target (e.g., RF_1, RF_2, XGB_1)
META_MODEL_IDS = {
    "RF_1": {
        "description": "Random Forest v1 (50 trees, max_depth=20, min_instances_per_node=20)",
        "num_trees": 50,
        "max_depth": 20,
        "min_instances_per_node": 20,
        "use_preprocessed_features": True
    },
    "RF_1_FAST": {
        "description": "Random Forest v1 Fast (20 trees, max_depth=10, min_instances_per_node=20) - for precomputation",
        "num_trees": 20,
        "max_depth": 10,
        "min_instances_per_node": 20,
        "use_preprocessed_features": False
    }
}


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
        """
        Apply meta-models to predict previous flight components.
        
        If pipelines are None (pass-through mode), uses existing precomputed predictions.
        Otherwise, computes predictions using trained pipelines.
        """
        # Pass-through mode: predictions already exist (precomputed in split.py)
        if (self._air_time_pipeline is None and 
            self._taxi_time_pipeline is None and 
            self._total_duration_pipeline is None):
            # Check if predictions exist (look for any model_id pattern)
            existing_pred_cols = [col for col in df.columns if col.startswith("predicted_prev_flight_air_time_")]
            if existing_pred_cols:
                # Extract model_id and rename to standard names for backwards compatibility
                model_id = existing_pred_cols[0].replace("predicted_prev_flight_air_time_", "")
                df = df.withColumnRenamed(
                    f"predicted_prev_flight_air_time_{model_id}",
                    "predicted_prev_flight_air_time"
                ).withColumnRenamed(
                    f"predicted_prev_flight_taxi_time_{model_id}",
                    "predicted_prev_flight_taxi_time"
                ).withColumnRenamed(
                    f"predicted_prev_flight_total_duration_{model_id}",
                    "predicted_prev_flight_total_duration"
                )
                return df
            else:
                raise ValueError("Meta-model predictions not found. "
                               "Either precompute predictions or fit the model first.")
        
        # Normal mode: compute predictions using trained pipelines
        # Create special case flags if they don't exist (for prediction on new data)
        if "is_first_flight" not in df.columns:
            df = df.withColumn(
                "is_first_flight",
                (col("lineage_rank") == 1).cast("int")  # Convert boolean to int for StringIndexer
            )
        if "is_jump" not in df.columns:
            df = df.withColumn(
                "is_jump",
                col("lineage_is_jump").cast("int")  # Convert boolean to int for StringIndexer
            )
        
        # Note: prev_flight graph features are now added by GraphFeaturesEstimator
        # No need to add them here - they should already be in the DataFrame
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
        
        # Impute NULL predictions (safety net for edge cases - should be rare now that we train on all flights)
        # Trees can handle NULL/missing values, but this provides a fallback for any edge cases
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
        
        # Safety check: Validate that predictions exist and are reasonable
        # Check if predictions are all NULL (should not happen with coalesce, but verify)
        total_rows = df_with_predictions.count()
        if total_rows > 0:
            air_time_null = df_with_predictions.filter(
                col("predicted_prev_flight_air_time").isNull()
            ).count()
            taxi_time_null = df_with_predictions.filter(
                col("predicted_prev_flight_taxi_time").isNull()
            ).count()
            total_duration_null = df_with_predictions.filter(
                col("predicted_prev_flight_total_duration").isNull()
            ).count()
            
            if air_time_null > 0 or taxi_time_null > 0 or total_duration_null > 0:
                raise ValueError(
                    f"Meta-model predictions contain NULL values after imputation! "
                    f"air_time: {air_time_null}/{total_rows}, "
                    f"taxi_time: {taxi_time_null}/{total_rows}, "
                    f"total_duration: {total_duration_null}/{total_rows}. "
                    f"This indicates a serious issue with meta-model inference."
                )
            
            # Check if predictions are all zeros (indicates potential failure)
            air_time_zero = df_with_predictions.filter(
                col("predicted_prev_flight_air_time") == 0.0
            ).count()
            if air_time_zero == total_rows:
                raise ValueError(
                    f"All meta-model predictions for air_time are 0.0! "
                    f"This indicates meta-model inference failed. "
                    f"Check that meta-models were trained successfully."
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
        
        If predictions already exist (precomputed in split.py), skips training and returns pass-through model.
        """
        # Check if predictions already exist (precomputed)
        # Look for any model_id predictions
        existing_pred_cols = [col for col in train_df.columns if col.startswith("predicted_prev_flight_air_time_")]
        if existing_pred_cols:
            # Extract model_id from first existing column
            existing_model_id = existing_pred_cols[0].replace("predicted_prev_flight_air_time_", "")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Meta-model predictions already present ({existing_model_id}), skipping training")
            # Return pass-through model that uses existing predictions
            return MetaModelModel(
                air_time_pipeline=None,  # None indicates pass-through mode
                taxi_time_pipeline=None,
                total_duration_pipeline=None
            )
        
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Training meta-models for previous flight component prediction...")
        
        if self.use_preprocessed_features:
            print(f"  Using preprocessed features (features should already be imputed, indexed, and encoded)")
        
        # Create flags for special cases: first flights and jumps
        # These allow the model to learn special handling for these cases
        # When is_first_flight=True, model learns to predict imputed values (scheduled times)
        # When is_jump=True, model learns to handle aircraft repositioning events
        # This prevents these cases from contaminating predictions for normal flights
        train_df = train_df.withColumn(
            "is_first_flight",
            (col("lineage_rank") == 1).cast("int")  # Convert boolean to int for StringIndexer
        )
        train_df = train_df.withColumn(
            "is_jump",
            col("lineage_is_jump").cast("int")  # Convert boolean to int for StringIndexer
        )
        
        # Prepare training data: include ALL flights (first flights, jumps, and normal rotations)
        # We now include jumps because:
        # 1. We need to inference on jumps anyway (they occur in production)
        # 2. Trees can handle NULL/missing values in predictions
        # 3. There are "known jumps" when we have data gaps due to weather station data bugs
        # 4. The is_jump flag allows the model to learn special handling for jumps
        # First flights and jumps will have imputed prev_flight_* features from flight_lineage_features.py
        # The model will learn to handle these cases via the is_first_flight and is_jump flags
        train_with_prev = train_df  # No filtering - include all flights
        
        # Note: prev_flight graph features are now added by GraphFeaturesEstimator
        # No need to add them here - they should already be in the DataFrame
        
        # Count breakdown for logging
        total_count = train_with_prev.count()
        first_flight_count = train_with_prev.filter(col("is_first_flight") == 1).count()
        jump_count = train_with_prev.filter(col("is_jump") == 1).count()
        normal_count = total_count - first_flight_count - jump_count
        print(f"  Training on {total_count:,} rows ({normal_count:,} rotations, {first_flight_count:,} first flights, {jump_count:,} jumps)")
        
        # OPTIMIZATION: If using raw features, preprocess once and cache to avoid repeating expensive operations
        # (StringIndexer, OneHotEncoder) for each meta-model
        # NOTE: StringIndexer.fit() is inherently slow on high-cardinality categoricals (200+ airports)
        # This is a one-time cost per fold - subsequent models reuse the cached preprocessed DataFrame
        preprocessed_df = None
        if not self.use_preprocessed_features:
            # Collect all features used by all meta-models
            all_categorical = set()
            all_numerical = set()
            for target in ["air_time", "taxi_time", "total_duration"]:
                cat, num = self._get_meta_model_features(train_with_prev, target)
                all_categorical.update(cat)
                all_numerical.update(num)
            
            # CV-SAFE: Preprocess once on training data only (fit on train, will be applied to val later)
            # Imputer and StringIndexer are fit on train_with_prev (training data only)
            # This ensures no data leakage - imputation means and index mappings come from training data only
            stages = []
            if all_numerical:
                imputer = Imputer(
                    inputCols=list(all_numerical),
                    outputCols=[f"{col}_IMPUTED" for col in all_numerical],
                    strategy="mean"
                )
                stages.append(imputer)
            
            if all_categorical:
                # StringIndexer: fit on training data only (CV-safe)
                # Index mappings (e.g., airport codes -> 0, 1, 2...) are learned from training data
                indexer = StringIndexer(
                    inputCols=list(all_categorical),
                    outputCols=[f"{col}_INDEX" for col in all_categorical],
                    handleInvalid="keep"  # Unknown categories in val/test get index 0
                )
                stages.append(indexer)
                encoder = OneHotEncoder(
                    inputCols=[f"{col}_INDEX" for col in all_categorical],
                    outputCols=[f"{col}_VEC" for col in all_categorical],
                    dropLast=False
                )
                stages.append(encoder)
            
            if stages:
                from pyspark.ml import Pipeline as MLPipeline
                # OPTIMIZATION: Cache input DataFrame before fit/transform to speed up StringIndexer
                # StringIndexer needs to scan data to build index - caching helps
                train_with_prev_cached = train_with_prev.cache()
                _ = train_with_prev_cached.count()  # Trigger cache
                
                preprocess_pipeline = MLPipeline(stages=stages)
                # Fit on training data only (CV-safe)
                fitted_preprocessor = preprocess_pipeline.fit(train_with_prev_cached)
                # Transform training data
                preprocessed_df = fitted_preprocessor.transform(train_with_prev_cached).cache()
                # Trigger cache materialization
                _ = preprocessed_df.count()
                print(f"  ✓ Preprocessed {len(all_categorical)} categorical and {len(all_numerical)} numerical features (cached)")
                
                # Unpersist input cache to free memory
                train_with_prev_cached.unpersist()
        
        # Train meta-models (pass preprocessed_df if available)
        air_time_pipeline = self._train_air_time_model(train_with_prev, preprocessed_df=preprocessed_df)
        taxi_time_pipeline = self._train_taxi_time_model(train_with_prev, preprocessed_df=preprocessed_df)
        total_duration_pipeline = self._train_total_duration_model(train_with_prev, preprocessed_df=preprocessed_df)
        
        # Debug: Count how many actually trained
        trained_count = sum([
            1 if air_time_pipeline is not None else 0,
            1 if taxi_time_pipeline is not None else 0,
            1 if total_duration_pipeline is not None else 0
        ])
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        if trained_count < 3:
            print(f"[{timestamp}] ⚠ Meta-model training complete ({trained_count}/3 models, took {duration})")
            failed = []
            if air_time_pipeline is None:
                failed.append("air_time")
            if taxi_time_pipeline is None:
                failed.append("taxi_time")
            if total_duration_pipeline is None:
                failed.append("total_duration")
            print(f"  ⚠ Failed: {', '.join(failed)}")
        else:
            print(f"[{timestamp}] ✓ Meta-model training complete! (3/3 models, took {duration})")
        
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
            # AIR TIME: Time from prev_flight_origin (departure) → prev_flight_dest (arrival)
            # Highly dependent on distance, origin/dest, and seasonal patterns (jet streams)
            # Key insight: Distance + origin + dest can approximate route-specific patterns
            # without the 3000+ cardinality of explicit route combinations
            categorical_features = [
                # Special case flags - allow model to learn special handling
                # When is_first_flight=True, model learns to predict imputed values (scheduled times)
                # When is_jump=True, model learns to handle aircraft repositioning events
                # This prevents these cases from contaminating predictions for normal flights
                'is_first_flight',  # Boolean flag: lineage_rank == 1
                'is_jump',          # Boolean flag: lineage_is_jump (aircraft repositioning or data gaps)
                
                # Airport features - capture route-specific patterns
                # Air time is from prev_flight_origin → prev_flight_dest (not origin!)
                'prev_flight_origin',        # Previous flight's origin airport (~200 categories) - where it departed
                'prev_flight_dest',          # Previous flight's destination airport (~200 categories) - where it arrived
                # Note: For jumps, prev_flight_dest != origin, so we need prev_flight_dest explicitly
                # Note: For first flights, these are imputed to origin (from flight_lineage_features.py)
                # Note: For jumps, these represent the actual previous flight route (repositioning)
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
                
                # Graph features for previous flight's route (prev_flight_origin → prev_flight_dest)
                # PageRank captures airport importance/connectivity
                # Note: For jumps, prev_flight_dest != origin, so we need prev_flight_dest_pagerank_* explicitly
                'prev_flight_origin_pagerank_weighted',
                'prev_flight_origin_pagerank_unweighted',
                'prev_flight_dest_pagerank_weighted',  # Needed for jumps (prev_flight_dest != origin)
                'prev_flight_dest_pagerank_unweighted',
            ]
            
        elif target_name == "taxi_time":
            # TAXI TIME: Dependent on airport congestion, time of day, weather
            # Taxi time is airport-specific (not route-specific), so we only need current flight's origin
            # (which is where the previous flight landed and where taxi-out happens)
            categorical_features = [
                # Special case flags - allow model to learn special handling
                'is_first_flight',  # Boolean flag: lineage_rank == 1
                'is_jump',          # Boolean flag: lineage_is_jump (aircraft repositioning or data gaps)
                
                # Airport-level (taxi time is airport-specific, not route-specific)
                'origin',                    # Current flight's origin (where taxi-out happens, = prev_flight_dest)
                # Note: For first flights, prev_flight_dest is imputed to origin
                # Note: For jumps, prev_flight_dest != origin (repositioning)
                
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
                'origin_state_abr',          # Current flight's origin state (= prev_flight_dest_state)
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
                
                # Graph features for airport where plane is currently (current origin)
                # Taxi time is predicted for the plane's current location (current flight's origin)
                # The plane is taxiing at the current origin airport when we make the prediction
                # PageRank captures airport importance/connectivity, which affects taxi times
                # (busy hubs have longer taxi times than smaller airports)
                # Note: For normal rotations, prev_flight_dest = origin, so origin_pagerank_* is correct
                # Note: For jumps, prev_flight_dest != origin, but we still use origin_pagerank_* because
                #       the plane is at the current origin when we're predicting (current departure time is a feature)
                'origin_pagerank_weighted',  # Current flight's origin (where plane is now, where taxi happens)
                'origin_pagerank_unweighted',
                # Also include prev_flight_origin for context (where previous flight departed from)
                'prev_flight_origin_pagerank_weighted',  # Previous flight's origin (context)
                'prev_flight_origin_pagerank_unweighted',
            ]
            
        elif target_name == "total_duration":
            # TOTAL DURATION: Combination of air time + taxi time
            # Air time is from prev_flight_origin → prev_flight_dest
            # Use features from both, but focus on strongest predictors
            categorical_features = [
                # Special case flags - allow model to learn special handling
                'is_first_flight',  # Boolean flag: lineage_rank == 1
                'is_jump',          # Boolean flag: lineage_is_jump (aircraft repositioning or data gaps)
                
                # Airport features (air time component) - origin + dest + distance capture route patterns
                # Total duration includes air time (prev_flight_origin → prev_flight_dest)
                'prev_flight_origin',        # Previous flight's origin airport (~200 categories) - where it departed
                'prev_flight_dest',          # Previous flight's destination airport (~200 categories) - where it arrived
                # Note: For jumps, prev_flight_dest != origin, so we need prev_flight_dest explicitly
                # Note: For first flights, these are imputed to origin (from flight_lineage_features.py)
                # Note: For jumps, these represent the actual previous flight route (repositioning)
                
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
                
                # Graph features for previous flight's route (prev_flight_origin → prev_flight_dest)
                # PageRank captures airport importance/connectivity
                # May correlate with both air time (route patterns) and taxi time (airport size)
                # Note: For jumps, prev_flight_dest != origin, so we need prev_flight_dest_pagerank_* explicitly
                'prev_flight_origin_pagerank_weighted',
                'prev_flight_origin_pagerank_unweighted',
                'prev_flight_dest_pagerank_weighted',  # Needed for jumps (prev_flight_dest != origin)
                'prev_flight_dest_pagerank_unweighted',
            ]
        
        # Filter to only include features that exist in DataFrame
        # If use_preprocessed_features=True, check for processed feature names (_VEC, _IMPUTED)
        # Otherwise, check for original feature names
        if self.use_preprocessed_features:
            # Check for processed feature names
            available_cols = set(df.columns)
            available_categorical = [f for f in categorical_features if f"{f}_VEC" in available_cols]
            available_numerical = [f for f in numerical_features if f"{f}_IMPUTED" in available_cols]
        else:
            # Check for original feature names
            available_categorical = [f for f in categorical_features if f in df.columns]
            available_numerical = [f for f in numerical_features if f in df.columns]
        
        # Validate graph features are available (target-specific)
        if target_name == "air_time":
            graph_feature_names = [
                'prev_flight_origin_pagerank_weighted',
                'prev_flight_origin_pagerank_unweighted',
                'prev_flight_dest_pagerank_weighted',  # Needed for jumps (prev_flight_dest != origin)
                'prev_flight_dest_pagerank_unweighted'
            ]
        elif target_name == "taxi_time":
            graph_feature_names = [
                'origin_pagerank_weighted',  # Current origin (where plane is now, where taxi happens)
                'origin_pagerank_unweighted',
                'prev_flight_origin_pagerank_weighted',  # Context (where previous flight departed)
                'prev_flight_origin_pagerank_unweighted'
            ]
        elif target_name == "total_duration":
            graph_feature_names = [
                'prev_flight_origin_pagerank_weighted',
                'prev_flight_origin_pagerank_unweighted',
                'prev_flight_dest_pagerank_weighted',  # Needed for jumps (prev_flight_dest != origin)
                'prev_flight_dest_pagerank_unweighted'
            ]
        else:
            graph_feature_names = []
        
        available_graph_features = [f for f in graph_feature_names if f in available_numerical]
        if not available_graph_features:
            print(f"    ⚠ Warning: No graph features found for {target_name} meta-model")
        
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
            
            # Validate graph features in processed features
            # Note: For rotations, prev_flight_dest = origin, but for jumps they differ
            # We use prev_flight_dest_pagerank_* explicitly to handle jumps correctly
            graph_feature_names = [
                'prev_flight_origin_pagerank_weighted',
                'prev_flight_origin_pagerank_unweighted',
                'prev_flight_dest_pagerank_weighted',  # Needed for jumps (prev_flight_dest != origin)
                'prev_flight_dest_pagerank_unweighted',
                'origin_pagerank_weighted',
                'origin_pagerank_unweighted',
                'dest_pagerank_weighted',
                'dest_pagerank_unweighted'
            ]
            processed_graph_features = [f"{col}_IMPUTED" for col in graph_feature_names if f"{col}_IMPUTED" in processed_numerical]
            if not processed_graph_features:
                print(f"    ⚠ Warning: No processed graph features found")
            
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
    
    def _train_air_time_model(self, train_df, preprocessed_df=None):
        """Train meta-model to predict prev_flight_air_time."""
        print("  Training meta-model: prev_flight_air_time", end="")
        
        # Check if target exists
        if "prev_flight_air_time" not in train_df.columns:
            print(" - ⚠ SKIPPED (target not found)")
            return None
        
        # Filter to rows with valid target
        train_air_time = train_df.filter(col("prev_flight_air_time").isNotNull())
        count = train_air_time.count()
        
        if count < 100:
            print(f" - ⚠ SKIPPED (only {count} rows)")
            return None
        
        # Get features
        categorical_features, numerical_features = self._get_meta_model_features(
            train_air_time, "air_time"
        )
        
        print(f" ({count:,} rows, {len(categorical_features)} cat, {len(numerical_features)} num)", end="")
        
        # Use preprocessed_df if available (optimization: avoid re-preprocessing)
        processed_df = None
        if preprocessed_df is not None:
            # Filter preprocessed_df to match train_air_time rows
            processed_df = preprocessed_df.filter(col("prev_flight_air_time").isNotNull())
        elif self.use_preprocessed_features:
            processed_df = train_air_time
        
        # Build and fit pipeline
        pipeline = self._build_meta_model_pipeline(
            categorical_features, 
            numerical_features, 
            "prev_flight_air_time",
            processed_df=processed_df
        )
        
        fitted_pipeline = pipeline.fit(train_air_time)
        print(" ✓")
        
        return fitted_pipeline
    
    def _train_taxi_time_model(self, train_df, preprocessed_df=None):
        """Train meta-model to predict prev_flight_taxi_time (taxi_in + taxi_out)."""
        print("  Training meta-model: prev_flight_taxi_time", end="")
        
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
            # Also add to preprocessed_df if available
            if preprocessed_df is not None:
                preprocessed_df = preprocessed_df.withColumn(
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
            print(f" - ⚠ SKIPPED (only {count} rows)")
            return None
        
        # Get features
        categorical_features, numerical_features = self._get_meta_model_features(
            train_taxi_time, "taxi_time"
        )
        
        print(f" ({count:,} rows, {len(categorical_features)} cat, {len(numerical_features)} num)", end="")
        
        # Use preprocessed_df if available (optimization: avoid re-preprocessing)
        processed_df = None
        if preprocessed_df is not None:
            processed_df = preprocessed_df.filter(col("prev_flight_taxi_time").isNotNull())
        elif self.use_preprocessed_features:
            processed_df = train_taxi_time
        
        # Build and fit pipeline
        pipeline = self._build_meta_model_pipeline(
            categorical_features, 
            numerical_features, 
            "prev_flight_taxi_time",
            processed_df=processed_df
        )
        
        fitted_pipeline = pipeline.fit(train_taxi_time)
        print(" ✓")
        
        return fitted_pipeline
    
    def _train_total_duration_model(self, train_df, preprocessed_df=None):
        """Train meta-model to predict prev_flight_total_duration (actual_elapsed_time)."""
        print("  Training meta-model: prev_flight_total_duration", end="")
        
        # Use actual_elapsed_time as target
        if "prev_flight_actual_elapsed_time" not in train_df.columns:
            print(" - ⚠ SKIPPED (target not found)")
            return None
        
        # Filter to rows with valid target
        train_total_duration = train_df.filter(col("prev_flight_actual_elapsed_time").isNotNull())
        count = train_total_duration.count()
        
        if count < 100:
            print(f" - ⚠ SKIPPED (only {count} rows)")
            return None
        
        # Temporarily rename for pipeline
        train_total_duration = train_total_duration.withColumn(
            "prev_flight_total_duration",
            col("prev_flight_actual_elapsed_time")
        )
        
        # Also add to preprocessed_df if available
        processed_total_duration = None
        if preprocessed_df is not None:
            processed_total_duration = preprocessed_df.filter(
                col("prev_flight_actual_elapsed_time").isNotNull()
            ).withColumn(
                "prev_flight_total_duration",
                col("prev_flight_actual_elapsed_time")
            )
        
        # Get features
        categorical_features, numerical_features = self._get_meta_model_features(
            train_total_duration, "total_duration"
        )
        
        print(f" ({count:,} rows, {len(categorical_features)} cat, {len(numerical_features)} num)", end="")
        
        # Use preprocessed_df if available (optimization: avoid re-preprocessing)
        processed_df = processed_total_duration if processed_total_duration is not None else (train_total_duration if self.use_preprocessed_features else None)
        
        # Build and fit pipeline
        pipeline = self._build_meta_model_pipeline(
            categorical_features, 
            numerical_features, 
            "prev_flight_total_duration",
            processed_df=processed_df
        )
        
        fitted_pipeline = pipeline.fit(train_total_duration)
        print(" ✓")
        
        return fitted_pipeline


# -------------------------
# REUSABLE META-MODEL COMPUTATION FUNCTION
# -------------------------
def compute_meta_model_predictions(train_df, val_df, model_id="RF_1", verbose=True, use_preprocessed_features=None):
    """
    Train meta-models on training data and apply predictions to both train and val DataFrames.
    
    This function is designed for precomputing meta-model predictions during fold creation.
    It trains on training data only (CV-safe) and applies to both train and val.
    
    Args:
        train_df: Training DataFrame (used to train meta-models)
        val_df: Validation/test DataFrame (gets predictions joined)
        model_id: Model identifier (e.g., "RF_1") - must exist in META_MODEL_IDS
        verbose: Whether to print progress messages
        use_preprocessed_features: Override the model config's use_preprocessed_features setting.
                                   If None, uses the value from META_MODEL_IDS.
                                   For precomputation (raw features), set to False.
                                   For pipeline use (preprocessed features), set to True.
    
    Returns:
        tuple: (train_df_with_predictions, val_df_with_predictions) both with predicted columns:
            - predicted_prev_flight_air_time_{model_id}
            - predicted_prev_flight_taxi_time_{model_id}
            - predicted_prev_flight_total_duration_{model_id}
    """
    if model_id not in META_MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id}. Available: {list(META_MODEL_IDS.keys())}")
    
    model_config = META_MODEL_IDS[model_id]
    
    # Use override if provided, otherwise use config default
    if use_preprocessed_features is None:
        use_preprocessed_features = model_config.get("use_preprocessed_features", False)
    
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing meta-model predictions ({model_id})...")
        print(f"  Config: {model_config['description']}")
        print(f"  Using {'preprocessed' if use_preprocessed_features else 'raw'} features")
    
    # Create estimator with model config
    estimator = MetaModelEstimator(
        num_trees=model_config["num_trees"],
        max_depth=model_config["max_depth"],
        min_instances_per_node=model_config["min_instances_per_node"],
        use_preprocessed_features=use_preprocessed_features
    )
    
    # Train on training data
    model = estimator.fit(train_df)
    
    # Apply to both train and val
    try:
        train_with_predictions = model.transform(train_df)
        val_with_predictions = model.transform(val_df)
    except Exception as e:
        raise RuntimeError(
            f"Meta-model inference failed! Error: {str(e)}\n"
            f"This may indicate missing features or model training issues. "
            f"Check that all required features are present in the DataFrames."
        ) from e
    
    # Rename prediction columns to include model_id
    prediction_cols = [
        "predicted_prev_flight_air_time",
        "predicted_prev_flight_taxi_time",
        "predicted_prev_flight_total_duration"
    ]
    
    for col_name in prediction_cols:
        if col_name in train_with_predictions.columns:
            new_name = f"{col_name}_{model_id}"
            train_with_predictions = train_with_predictions.withColumnRenamed(col_name, new_name)
            val_with_predictions = val_with_predictions.withColumnRenamed(col_name, new_name)
        else:
            if verbose:
                print(f"  ⚠ Warning: Prediction column '{col_name}' not found in output. Meta-model may have failed.")
    
    # Safety check: Validate that predictions were created
    missing_cols = [col for col in prediction_cols if f"{col}_{model_id}" not in val_with_predictions.columns]
    if missing_cols:
        raise ValueError(
            f"Meta-model predictions missing for: {missing_cols}. "
            f"Expected columns: {[f'{col}_{model_id}' for col in prediction_cols]}. "
            f"Available columns: {sorted(val_with_predictions.columns)}. "
            f"This indicates meta-model inference failed."
        )
    
    if verbose:
        # Validate predictions are reasonable (not all zeros)
        val_total = val_with_predictions.count()
        if val_total > 0:
            air_time_col = f"predicted_prev_flight_air_time_{model_id}"
            air_time_zero = val_with_predictions.filter(col(air_time_col) == 0.0).count()
            if air_time_zero == val_total:
                print(f"  ⚠ WARNING: All validation predictions for air_time are 0.0! This may indicate inference failure.")
            elif air_time_zero > val_total * 0.5:
                print(f"  ⚠ Warning: {air_time_zero:,}/{val_total:,} validation predictions for air_time are 0.0 (>{air_time_zero/val_total*100:.1f}%)")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Meta-model predictions computed! (took {duration})")
    
    return train_with_predictions, val_with_predictions

