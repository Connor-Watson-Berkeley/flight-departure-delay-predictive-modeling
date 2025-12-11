"""
cv.py (CUSTOM or OTPW, with fold strategy support)

Assumptions:
- Folds were created from split.py with N_FOLDS = 3 and CREATE_TEST_FOLD = True
- Therefore total fold indices written = 4:
    FOLD_1_VAL, FOLD_2_VAL, FOLD_3_VAL, FOLD_4_TEST
- Files live in:
    dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed
- File naming:
    OTPW_{SOURCE}_{VERSION}_FOLD_{i}_{TRAIN|VAL|TEST}.parquet
    where SOURCE is "CUSTOM" or "OTPW"

Usage:
    # Load CUSTOM data (default)
    loader = FlightDelayDataLoader()
    
    # Load OTPW data
    loader = FlightDelayDataLoader(source="OTPW")
    
    # Load OTPW data with graph features
    loader = FlightDelayDataLoader(source="OTPW", suffix="_with_graph")

Fold Strategies:
- "non_overlapping" (default): Uses pre-saved non-overlapping rolling window folds
    - 3 CV folds: 1-year training, 1-year validation (non-overlapping)
    - 1 test fold: All previous data training, 4-year test
- "overlapping": Creates overlapping 2-year training / 1-year validation folds
    - 3 CV folds: 2-year training, 1-year validation (training windows overlap by 1 year)
    - 1 test fold: 2-year training, 1-year test
- "expanding": (Future) Expanding window strategy
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import time


# -----------------------------
# HARD-CODED GLOBALS
# -----------------------------
FOLDER_PATH = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "CUSTOM"
VERSIONS = ["3M", "12M", "60M"]

# 3 CV folds + 1 test fold = 4 total fold indices
TOTAL_FOLDS = 4


# -----------------------------
# DATA LOADER (CUSTOM OR OTPW)
# -----------------------------
class FlightDelayDataLoader:
    def __init__(self, suffix: str = "", source: str = None):
        """
        Initialize the data loader.
        
        Args:
            suffix: Optional suffix to append to version (e.g., "_with_graph", "_with_graph_and_metamodels")
                    Default: "" (base folds with only flight lineage features)
            source: Source dataset to load ("CUSTOM" or "OTPW"). 
                    Default: None (uses global SOURCE variable for backwards compatibility)
        """
        self.folds = {}
        self.suffix = suffix
        self.source = source if source is not None else SOURCE  # Use provided source or fall back to global
        self.numerical_features = [
            'hourlyprecipitation',
            'hourlysealevelpressure',
            'hourlyaltimetersetting',
            'hourlywetbulbtemperature',
            'hourlystationpressure',
            'hourlywinddirection',
            'hourlyrelativehumidity',
            'hourlywindspeed',
            'hourlydewpointtemperature',
            'hourlydrybulbtemperature',
            'hourlyvisibility',
            'crs_elapsed_time',
            'distance',
            'elevation',
            
            # Flight lineage features
            'lineage_rank',
            'prev_flight_dep_delay',
            'prev_flight_arr_delay',
            'prev_flight_air_time',
            'prev_flight_crs_elapsed_time',
            'prev_flight_taxi_in',
            'prev_flight_taxi_out',
            'prev_flight_actual_elapsed_time',
            'prev_flight_distance',
            # Scheduled time features (data leakage-free)
            'scheduled_lineage_rotation_time_minutes',
            'scheduled_lineage_turnover_time_minutes',
            # Actual time features (may have data leakage)
            'lineage_rotation_time_minutes',
            'lineage_actual_turnover_time_minutes',
            'lineage_actual_taxi_time_minutes',
            'lineage_actual_turn_time_minutes',
            # Safe features (intelligent data leakage handling)
            'safe_lineage_rotation_time_minutes',
            'safe_prev_departure_delay',
            'safe_prev_arrival_delay',
            'safe_time_since_prev_arrival',
            # Other lineage features
            'lineage_expected_flight_time_minutes',
            'lineage_cumulative_delay',
            'lineage_num_previous_flights',
            'lineage_avg_delay_previous_flights',
            'lineage_max_delay_previous_flights',
            # Required time features
            'required_time_prev_flight_minutes',
            'safe_required_time_prev_flight_minutes',
            'safe_impossible_on_time_flag',
            # Rolling average delay features (data leakage-free)
            'tail_num_rolling_avg_delay_24h',
            'tail_num_rolling_avg_delay_7d',
            'tail_num_rolling_avg_delay_30d',
            'origin_rolling_avg_delay_24h',
            'origin_rolling_avg_delay_7d',
            'origin_rolling_avg_delay_30d',
        ]
        
        # Dynamically add graph features if suffix contains "_with_graph"
        if "_with_graph" in suffix:
            graph_features = [
                'origin_pagerank_weighted',
                'origin_pagerank_unweighted',
                'dest_pagerank_weighted',
                'dest_pagerank_unweighted',
                # Previous flight graph features (may or may not exist depending on lineage)
                'prev_flight_origin_pagerank_weighted',
                'prev_flight_origin_pagerank_unweighted',
                'prev_flight_dest_pagerank_weighted',
                'prev_flight_dest_pagerank_unweighted',
            ]
            self.numerical_features.extend(graph_features)
        
        # Dynamically add meta-model prediction features if suffix contains "_with_graph_and_metamodels"
        if "_with_graph_and_metamodels" in suffix:
            meta_model_features = [
                'predicted_prev_flight_air_time_XGB_1',
                'predicted_prev_flight_turnover_time_XGB_1',
                'predicted_prev_flight_total_duration_XGB_1',
            ]
            self.numerical_features.extend(meta_model_features)

    def _cast_numerics(self, df):
        """Safely cast all configured numeric columns to doubles."""
        NULL_PAT = r'^(NA|N/A|NULL|null|None|none|\\N|\\s*|\\.|M|T)$'
        
        for colname in self.numerical_features:
            if colname in df.columns:
                df = df.withColumn(
                    colname,
                    F.regexp_replace(F.col(colname).cast("string"), NULL_PAT, "")
                    .cast("double")
                )
        
        # Explicitly cast labels to expected numeric types
        if "DEP_DELAY" in df.columns:
            df = df.withColumn("DEP_DELAY", col("DEP_DELAY").cast("double"))
        if "DEP_DEL15" in df.columns:
            df = df.withColumn("DEP_DEL15", col("DEP_DEL15").cast("int"))
        if "SEVERE_DEL60" in df.columns:
            df = df.withColumn("SEVERE_DEL60", col("SEVERE_DEL60").cast("int"))
        
        return df

    def _load_parquet(self, name):
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.parquet(f"{FOLDER_PATH}/{name}.parquet")
        df = self._cast_numerics(df)
        return df

    def _load_version(self, version):
        folds = []
        for fold_idx in range(1, TOTAL_FOLDS + 1):
            # Include suffix in filename if provided
            # Use self.source (which may be provided or fall back to global SOURCE)
            train_name = f"OTPW_{self.source}_{version}{self.suffix}_FOLD_{fold_idx}_TRAIN"
            train_df = self._load_parquet(train_name)

            if fold_idx < TOTAL_FOLDS:
                val_name = f"OTPW_{self.source}_{version}{self.suffix}_FOLD_{fold_idx}_VAL"
                val_df = self._load_parquet(val_name)
                folds.append((train_df, val_df))
            else:
                test_name = f"OTPW_{self.source}_{version}{self.suffix}_FOLD_{fold_idx}_TEST"
                test_df = self._load_parquet(test_name)
                folds.append((train_df, test_df))

        return folds

    def load(self):
        """Load all available versions, skipping any that don't exist."""
        for v in VERSIONS:
            try:
                self.folds[v] = self._load_version(v)
            except Exception as e:
                # Skip versions that don't exist or can't be loaded
                error_msg = str(e)
                if "PATH_NOT_FOUND" in error_msg or "does not exist" in error_msg:
                    print(f"⚠ Skipping version '{v}' - files not found (suffix: '{self.suffix}')")
                else:
                    # Re-raise if it's a different error (permissions, etc.)
                    raise

    def get_version(self, version):
        return self.folds[version]
    
    def get_valid_numerical_features(self, df, sample_size=10000):
        """
        Get numerical features that exist in the DataFrame and have at least some non-null values.
        This is useful for filtering out all-NULL columns before passing to Imputer.
        
        Args:
            df: Spark DataFrame to check
            sample_size: Number of rows to sample for checking (default: 10000)
                        Use None to check all rows (slower but more accurate)
        
        Returns:
            list: List of numerical feature names that are valid (exist and have non-null values)
        """
        valid_features = []
        
        # Sample DataFrame if specified (for performance)
        check_df = df if sample_size is None else df.limit(sample_size)
        
        # Check each numerical feature
        for col_name in self.numerical_features:
            if col_name not in df.columns:
                # Column doesn't exist - skip it
                continue
            
            # Check if column has any non-null values
            non_null_count = check_df.filter(F.col(col_name).isNotNull()).count()
            
            if non_null_count > 0:
                valid_features.append(col_name)
            else:
                # All values are NULL - skip it
                print(f"⚠ Skipping '{col_name}' - all values are NULL")
        
        return valid_features


# -----------------------------
# EVALUATOR (NULL-SAFE RMSE)
# -----------------------------
class FlightDelayEvaluator:
    def __init__(
        self,
        prediction_col="prediction",
        numeric_label_col="DEP_DELAY",
        binary_label_col="DEP_DEL15",
        severe_label_col="SEVERE_DEL60",
    ):
        self.prediction_col = prediction_col
        self.numeric_label_col = numeric_label_col
        self.binary_label_col = binary_label_col
        self.severe_label_col = severe_label_col

        self.rmse_evaluator = RegressionEvaluator(
            predictionCol=prediction_col,
            labelCol=numeric_label_col,
            metricName="rmse"
        )

    def calculate_rmse(self, predictions_df):
        # Drop any residual nulls before RegressionEvaluator sees them
        clean = predictions_df.dropna(
            subset=[self.numeric_label_col, self.prediction_col]
        )
        return self.rmse_evaluator.evaluate(clean)

    def _calculate_classification_metrics(self, predictions_df, threshold, label_col):
        # Null-safe for classification too
        df = predictions_df.dropna(subset=[self.prediction_col, label_col])

        pred_binary_col = f"pred_binary_{threshold}"
        df = df.withColumn(
            pred_binary_col,
            F.when(F.col(self.prediction_col) >= threshold, 1).otherwise(0)
        )

        tp = df.filter((F.col(pred_binary_col) == 1) & (F.col(label_col) == 1)).count()
        fp = df.filter((F.col(pred_binary_col) == 1) & (F.col(label_col) == 0)).count()
        tn = df.filter((F.col(pred_binary_col) == 0) & (F.col(label_col) == 0)).count()
        fn = df.filter((F.col(pred_binary_col) == 0) & (F.col(label_col) == 1)).count()

        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / total if total else 0.0

        return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                    precision=precision, recall=recall, f1=f1, accuracy=accuracy)

    def calculate_otpa_metrics(self, predictions_df):
        """Calculate On-Time Prediction Accuracy (OTPA) metrics.
        
        Returns:
            dict: Contains 'accuracy', 'precision', 'recall', 'f1' for 15-minute threshold
        """
        return self._calculate_classification_metrics(
            predictions_df, threshold=15, label_col=self.binary_label_col
        )

    def calculate_sddr_metrics(self, predictions_df):
        """Calculate Severe Delay Detection Rate (SDDR) metrics.
        
        Returns:
            dict: Contains 'accuracy', 'precision', 'recall', 'f1' for 60-minute threshold
        """
        return self._calculate_classification_metrics(
            predictions_df, threshold=60, label_col=self.severe_label_col
        )

    def evaluate(self, predictions_df):
        """Evaluate predictions and return all metrics.
        
        Returns:
            dict: Contains:
                - "rmse": Root Mean Squared Error
                - "otpa": On-Time Prediction Accuracy metrics (dict with accuracy, precision, recall, f1)
                - "sddr": Severe Delay Detection Rate metrics (dict with accuracy, precision, recall, f1)
        """
        otpa_metrics = self.calculate_otpa_metrics(predictions_df)
        sddr_metrics = self.calculate_sddr_metrics(predictions_df)
        
        return {
            "rmse": self.calculate_rmse(predictions_df),
            "otpa": otpa_metrics["accuracy"],  # Backwards compatibility
            "otpa_precision": otpa_metrics["precision"],
            "otpa_recall": otpa_metrics["recall"],
            "otpa_f1": otpa_metrics["f1"],
            "sddr": sddr_metrics["recall"],  # Backwards compatibility (SDDR is recall)
            "sddr_precision": sddr_metrics["precision"],
            "sddr_recall": sddr_metrics["recall"],
            "sddr_f1": sddr_metrics["f1"],
        }


# -----------------------------
# CROSS-VALIDATOR (WITH FOLD STRATEGY SUPPORT)
# -----------------------------
class FlightDelayCV:
    def __init__(self, estimator, version, dataloader=None):
        """
        Initialize cross-validator with optional fold strategy.
        
        Args:
            estimator: Spark ML estimator (Pipeline, etc.)
            version: Data version ("3M", "12M", or "60M")
            dataloader: Optional FlightDelayDataLoader instance (default: None)
                - If provided, uses the provided data loader instead of creating a new one
                - If None, creates and loads a new FlightDelayDataLoader instance
        """
        self.estimator = estimator
        self.version = version
        if dataloader:
            self.data_loader = dataloader
        else:
            self.data_loader = FlightDelayDataLoader()
            self.data_loader.load()

        self.evaluator = FlightDelayEvaluator()
        
        self.folds = self.data_loader.get_version(version)

        self.metrics = []
        self.models = []
        self.test_metric = None
        self.test_model = None
    
    def fit(self):
        # CV folds only (exclude last test fold)
        results = []
        for i, (train_df, val_df) in enumerate(self.folds[:-1]):
            fold_index = i  # Use 0-based indexing (0, 1, 2)
            
            # Set version and fold_index on estimator if it supports it
            # This allows estimators like ConditionalExpectedValuesEstimator to know which fold
            # lookup tables to use during fit/transform
            if hasattr(self.estimator, 'setVersion'):
                # If estimator has setVersion method, set version from CV
                self.estimator.setVersion(self.version)
            if hasattr(self.estimator, 'setFoldIndex'):
                # If estimator has setFoldIndex method, use it
                self.estimator.setFoldIndex(fold_index)
            
            # If it's a Pipeline, try to set version and fold_index on stages that support it
            if hasattr(self.estimator, 'stages'):
                for stage in self.estimator.getStages():
                    if hasattr(stage, 'setVersion'):
                        stage.setVersion(self.version)
                    if hasattr(stage, 'setFoldIndex'):
                        stage.setFoldIndex(fold_index)

            model = self.estimator.fit(train_df)
            
            # Evaluate on training set
            train_preds = model.transform(train_df)
            train_metric = self.evaluator.evaluate(train_preds)
            
            # Evaluate on validation set
            val_preds = model.transform(val_df)
            val_metric = self.evaluator.evaluate(val_preds)

            self.metrics.append(val_metric)
            self.models.append(model)
            
            # Add both train and val results
            results.append({**{f"{k}_train": v for k, v in train_metric.items()}, 
                          **{f"{k}_val": v for k, v in val_metric.items()},
                          'fold': f'Fold {i}'})
            
            # ADDED_BY_SID_START
            # CRITICAL: Clean up after EVERY fold
            print(f"Cleaning up fold {i}...")
            # Get SparkSession for cache clearing
            spark = SparkSession.builder.getOrCreate()
            spark.catalog.clearCache()
            
            try:
                train_df.unpersist()
                val_df.unpersist()
                if 'val_preds' in locals():
                    val_preds.unpersist()
            except:
                pass
            
            import gc
            gc.collect()
            print(f"✓ Fold {i} cleaned up\n")
            # ADDED_BY_SID_END
            


        # Create DataFrame with alternating Train/Val rows
        formatted_results = []
        for result in results:
            fold = result['fold']
            # Train row
            formatted_results.append({
                'Fold': f"{fold} Train",
                'rmse': result['rmse_train'],
                'otpa': result['otpa_train'],
                'otpa_precision': result.get('otpa_precision_train', None),
                'otpa_recall': result.get('otpa_recall_train', None),
                'otpa_f1': result.get('otpa_f1_train', None),
                'sddr': result['sddr_train'],
                'sddr_precision': result.get('sddr_precision_train', None),
                'sddr_recall': result.get('sddr_recall_train', None),
                'sddr_f1': result.get('sddr_f1_train', None)
            })
            # Val row
            formatted_results.append({
                'Fold': f"{fold} Val",
                'rmse': result['rmse_val'],
                'otpa': result['otpa_val'],
                'otpa_precision': result.get('otpa_precision_val', None),
                'otpa_recall': result.get('otpa_recall_val', None),
                'otpa_f1': result.get('otpa_f1_val', None),
                'sddr': result['sddr_val'],
                'sddr_precision': result.get('sddr_precision_val', None),
                'sddr_recall': result.get('sddr_recall_val', None),
                'sddr_f1': result.get('sddr_f1_val', None)
            })
        
        m = pd.DataFrame(formatted_results)
        
        # Calculate mean and std for train and val separately
        train_rows = m[m['Fold'].str.contains('Train')]
        val_rows = m[m['Fold'].str.contains('Val')]
        
        mean_train = pd.DataFrame([{
            'Fold': 'Mean Train',
            'rmse': train_rows['rmse'].mean(),
            'otpa': train_rows['otpa'].mean(),
            'otpa_precision': train_rows['otpa_precision'].mean() if 'otpa_precision' in train_rows.columns else None,
            'otpa_recall': train_rows['otpa_recall'].mean() if 'otpa_recall' in train_rows.columns else None,
            'otpa_f1': train_rows['otpa_f1'].mean() if 'otpa_f1' in train_rows.columns else None,
            'sddr': train_rows['sddr'].mean(),
            'sddr_precision': train_rows['sddr_precision'].mean() if 'sddr_precision' in train_rows.columns else None,
            'sddr_recall': train_rows['sddr_recall'].mean() if 'sddr_recall' in train_rows.columns else None,
            'sddr_f1': train_rows['sddr_f1'].mean() if 'sddr_f1' in train_rows.columns else None
        }])
        
        std_train = pd.DataFrame([{
            'Fold': 'Std Train',
            'rmse': train_rows['rmse'].std(),
            'otpa': train_rows['otpa'].std(),
            'otpa_precision': train_rows['otpa_precision'].std() if 'otpa_precision' in train_rows.columns else None,
            'otpa_recall': train_rows['otpa_recall'].std() if 'otpa_recall' in train_rows.columns else None,
            'otpa_f1': train_rows['otpa_f1'].std() if 'otpa_f1' in train_rows.columns else None,
            'sddr': train_rows['sddr'].std(),
            'sddr_precision': train_rows['sddr_precision'].std() if 'sddr_precision' in train_rows.columns else None,
            'sddr_recall': train_rows['sddr_recall'].std() if 'sddr_recall' in train_rows.columns else None,
            'sddr_f1': train_rows['sddr_f1'].std() if 'sddr_f1' in train_rows.columns else None
        }])
        
        mean_val = pd.DataFrame([{
            'Fold': 'Mean Val',
            'rmse': val_rows['rmse'].mean(),
            'otpa': val_rows['otpa'].mean(),
            'otpa_precision': val_rows['otpa_precision'].mean() if 'otpa_precision' in val_rows.columns else None,
            'otpa_recall': val_rows['otpa_recall'].mean() if 'otpa_recall' in val_rows.columns else None,
            'otpa_f1': val_rows['otpa_f1'].mean() if 'otpa_f1' in val_rows.columns else None,
            'sddr': val_rows['sddr'].mean(),
            'sddr_precision': val_rows['sddr_precision'].mean() if 'sddr_precision' in val_rows.columns else None,
            'sddr_recall': val_rows['sddr_recall'].mean() if 'sddr_recall' in val_rows.columns else None,
            'sddr_f1': val_rows['sddr_f1'].mean() if 'sddr_f1' in val_rows.columns else None
        }])
        
        std_val = pd.DataFrame([{
            'Fold': 'Std Val',
            'rmse': val_rows['rmse'].std(),
            'otpa': val_rows['otpa'].std(),
            'otpa_precision': val_rows['otpa_precision'].std() if 'otpa_precision' in val_rows.columns else None,
            'otpa_recall': val_rows['otpa_recall'].std() if 'otpa_recall' in val_rows.columns else None,
            'otpa_f1': val_rows['otpa_f1'].std() if 'otpa_f1' in val_rows.columns else None,
            'sddr': val_rows['sddr'].std(),
            'sddr_precision': val_rows['sddr_precision'].std() if 'sddr_precision' in val_rows.columns else None,
            'sddr_recall': val_rows['sddr_recall'].std() if 'sddr_recall' in val_rows.columns else None,
            'sddr_f1': val_rows['sddr_f1'].std() if 'sddr_f1' in val_rows.columns else None
        }])
        
        m = pd.concat([m, mean_train, std_train, mean_val, std_val], ignore_index=True)
        return m

    def evaluate(self, use_fold_3_val_train=False):
        """
        Evaluate model on test set.
        
        Args:
            use_fold_3_val_train (bool): If True, train on fold 3's validation data (2018) instead of all previous data.
                                         Fold 3 is at index 2 (fold[-2]), and we use its validation portion (2018 data).
                                         This helps avoid temporal drift when predicting 2019 test set.
                                         Default: False (backwards compatible - uses all previous data)
        
        Returns:
            pd.DataFrame with Train and Test metrics
        """
        if use_fold_3_val_train:
            # Use fold 3 (index 2, fold[-2]) validation data (2018) to train, and test fold's test data (2019)
            # Fold 3 is at index 2: self.folds[2] = (train_df, val_df) where val_df is 2018 data
            # Test fold is at index 3: self.folds[3] = (train_all_previous, test_2019)
            _, fold_3_val_df = self.folds[-2]  # Use fold 3's validation data (2018) for training
            _, test_df = self.folds[-1]  # Use test fold's test data (2019)
            train_df = fold_3_val_df
            fold_index = 2  # Use fold 3's index for metadata
            print("=" * 80)
            print("EVALUATION MODE: Training on Fold 3 Validation Data (2018), Testing on Test Set (2019)")
            print("=" * 80)
            print("This helps avoid temporal drift from training on older data (2015-2017)")
            print("when predicting 2019 test set.")
        else:
            # Default behavior: use all previous data for training (backwards compatible)
            train_df, test_df = self.folds[-1]
            fold_index = 3  # Test fold is always at index 3 (0-based: folds are 0, 1, 2, 3)
        
        # Set version and fold_index on estimator if it supports it
        if hasattr(self.estimator, 'setVersion'):
            self.estimator.setVersion(self.version)
        if hasattr(self.estimator, 'setFoldIndex'):
            self.estimator.setFoldIndex(fold_index)
        
        # If it's a Pipeline, try to set version and fold_index on stages that support it
        if hasattr(self.estimator, 'stages'):
            for stage in self.estimator.getStages():
                if hasattr(stage, 'setVersion'):
                    stage.setVersion(self.version)
                if hasattr(stage, 'setFoldIndex'):
                    stage.setFoldIndex(fold_index)

        self.test_model = self.estimator.fit(train_df)
        
        # Evaluate on training set
        train_preds = self.test_model.transform(train_df)
        train_metric = self.evaluator.evaluate(train_preds)
        
        # Evaluate on test set
        test_preds = self.test_model.transform(test_df)
        test_metric = self.evaluator.evaluate(test_preds)
        
        self.test_metric = test_metric
        
        # Create DataFrame with Train and Test rows
        results = pd.DataFrame([
            {
                'Split': 'Train',
                'rmse': train_metric['rmse'],
                'otpa': train_metric['otpa'],
                'otpa_precision': train_metric.get('otpa_precision', None),
                'otpa_recall': train_metric.get('otpa_recall', None),
                'otpa_f1': train_metric.get('otpa_f1', None),
                'sddr': train_metric['sddr'],
                'sddr_precision': train_metric.get('sddr_precision', None),
                'sddr_recall': train_metric.get('sddr_recall', None),
                'sddr_f1': train_metric.get('sddr_f1', None)
            },
            {
                'Split': 'Test',
                'rmse': test_metric['rmse'],
                'otpa': test_metric['otpa'],
                'otpa_precision': test_metric.get('otpa_precision', None),
                'otpa_recall': test_metric.get('otpa_recall', None),
                'otpa_f1': test_metric.get('otpa_f1', None),
                'sddr': test_metric['sddr'],
                'sddr_precision': test_metric.get('sddr_precision', None),
                'sddr_recall': test_metric.get('sddr_recall', None),
                'sddr_f1': test_metric.get('sddr_f1', None)
            }
        ])
        
        return results
    
    def benchmark_inference(self, dataset="cv_folds", model=None, return_predictions=False):
        """
        Benchmark inference time on different datasets without re-training.
        Useful for measuring model runtime on various dataset sizes.
        
        Args:
            dataset (str): Which dataset to inference on. Options:
                - "cv_folds": Inference on all 3 CV validation sets (one per fold)
                - "test_train": Inference on test fold's training data (fold 4, all previous data)
                - "fold_3_val": Inference on fold 3's validation data (2018 data, same as use_fold_3_val_train)
            model: Optional model to use. If None, uses stored models from fit().
                   For "cv_folds", should be a list of 3 models (one per fold) or None to use self.models.
                   For other datasets, should be a single model or None to use self.test_model.
            return_predictions (bool): If True, returns predictions DataFrame(s) in addition to timing.
                                      If False, only returns timing information.
        
        Returns:
            dict: Contains timing information and optionally predictions:
                - "dataset": The dataset used
                - "num_rows": Number of rows inferred on
                - "total_time_seconds": Total inference time
                - "time_per_row_seconds": Average time per row
                - "predictions": (optional) Predictions DataFrame(s) if return_predictions=True
        """
        if dataset == "cv_folds":
            if model is None:
                if len(self.models) == 0:
                    raise ValueError("No models found. Run fit() first or provide models.")
                models = self.models
            else:
                if not isinstance(model, list) or len(model) != 3:
                    raise ValueError("For 'cv_folds', model must be a list of 3 models or None to use stored models.")
                models = model
            
            total_rows = 0
            total_time = 0
            predictions_list = []
            
            print("Benchmarking inference on CV validation sets...")
            for i, (train_df, val_df) in enumerate(self.folds[:-1]):
                if i >= len(models):
                    break
                
                model = models[i]
                num_rows = val_df.count()
                total_rows += num_rows
                
                print(f"  Fold {i+1} validation set: {num_rows:,} rows...", end=" ")
                start_time = time.time()
                preds = model.transform(val_df)
                # Force evaluation by counting (or materializing predictions)
                _ = preds.count()  # Materialize the DataFrame
                elapsed = time.time() - start_time
                total_time += elapsed
                print(f"{elapsed:.2f}s ({elapsed/num_rows*1000:.4f}ms per row)")
                
                if return_predictions:
                    predictions_list.append(preds)
            
            result = {
                "dataset": "cv_folds",
                "num_rows": total_rows,
                "total_time_seconds": total_time,
                "time_per_row_seconds": total_time / total_rows if total_rows > 0 else 0,
                "num_folds": len(models)
            }
            if return_predictions:
                result["predictions"] = predictions_list
            
        elif dataset == "test_train":
            if model is None:
                if self.test_model is None:
                    raise ValueError("No test model found. Run evaluate() first or provide a model.")
                model = self.test_model
            else:
                model = model
            
            train_df, _ = self.folds[-1]
            num_rows = train_df.count()
            
            print(f"Benchmarking inference on test fold's training data: {num_rows:,} rows...", end=" ")
            start_time = time.time()
            preds = model.transform(train_df)
            _ = preds.count()  # Materialize
            elapsed = time.time() - start_time
            print(f"{elapsed:.2f}s ({elapsed/num_rows*1000:.4f}ms per row)")
            
            result = {
                "dataset": "test_train",
                "num_rows": num_rows,
                "total_time_seconds": elapsed,
                "time_per_row_seconds": elapsed / num_rows if num_rows > 0 else 0
            }
            if return_predictions:
                result["predictions"] = preds
        
        elif dataset == "fold_3_val":
            if model is None:
                if self.test_model is None:
                    raise ValueError("No test model found. Run evaluate() first or provide a model.")
                model = self.test_model
            else:
                model = model
            
            _, fold_3_val_df = self.folds[-2]  # Fold 3's validation data (2018)
            num_rows = fold_3_val_df.count()
            
            print(f"Benchmarking inference on fold 3 validation data (2018): {num_rows:,} rows...", end=" ")
            start_time = time.time()
            preds = model.transform(fold_3_val_df)
            _ = preds.count()  # Materialize
            elapsed = time.time() - start_time
            print(f"{elapsed:.2f}s ({elapsed/num_rows*1000:.4f}ms per row)")
            
            result = {
                "dataset": "fold_3_val",
                "num_rows": num_rows,
                "total_time_seconds": elapsed,
                "time_per_row_seconds": elapsed / num_rows if num_rows > 0 else 0
            }
            if return_predictions:
                result["predictions"] = preds
        
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Must be one of: 'cv_folds', 'test_train', 'fold_3_val'")
        
        return result