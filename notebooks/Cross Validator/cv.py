"""
cv.py (CUSTOM-only, with fold strategy support)

Assumptions:
- Folds were created from split.py with N_FOLDS = 3 and CREATE_TEST_FOLD = True
- Therefore total fold indices written = 4:
    FOLD_1_VAL, FOLD_2_VAL, FOLD_3_VAL, FOLD_4_TEST
- Files live in:
    dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed
- File naming:
    OTPW_CUSTOM_{VERSION}_FOLD_{i}_{TRAIN|VAL|TEST}.parquet

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


# -----------------------------
# HARD-CODED GLOBALS
# -----------------------------
FOLDER_PATH = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "CUSTOM"
VERSIONS = ["3M", "12M", "60M"]

# 3 CV folds + 1 test fold = 4 total fold indices
TOTAL_FOLDS = 4


# -----------------------------
# DATA LOADER (CUSTOM ONLY)
# -----------------------------
class FlightDelayDataLoader:
    def __init__(self):
        self.folds = {}
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
            # Other lineage features
            'lineage_expected_flight_time_minutes',
            'lineage_cumulative_delay',
            'lineage_num_previous_flights',
            'lineage_avg_delay_previous_flights',
            'lineage_max_delay_previous_flights',
            # Required time features
            'required_time_prev_flight_minutes',
            'safe_required_time_prev_flight_minutes',
            'impossible_on_time_flag',
            'safe_impossible_on_time_flag',
        ]

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
            train_name = f"OTPW_{SOURCE}_{version}_FOLD_{fold_idx}_TRAIN"
            train_df = self._load_parquet(train_name)

            if fold_idx < TOTAL_FOLDS:
                val_name = f"OTPW_{SOURCE}_{version}_FOLD_{fold_idx}_VAL"
                val_df = self._load_parquet(val_name)
                folds.append((train_df, val_df))
            else:
                test_name = f"OTPW_{SOURCE}_{version}_FOLD_{fold_idx}_TEST"
                test_df = self._load_parquet(test_name)
                folds.append((train_df, test_df))

        return folds

    def load(self):
        for v in VERSIONS:
            self.folds[v] = self._load_version(v)

    def get_version(self, version):
        return self.folds[version]


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
        return self._calculate_classification_metrics(
            predictions_df, threshold=15, label_col=self.binary_label_col
        )["accuracy"]

    def calculate_sddr_metrics(self, predictions_df):
        return self._calculate_classification_metrics(
            predictions_df, threshold=60, label_col=self.severe_label_col
        )["recall"]

    def evaluate(self, predictions_df):
        return {
            "rmse": self.calculate_rmse(predictions_df),
            "otpa": self.calculate_otpa_metrics(predictions_df),
            "sddr": self.calculate_sddr_metrics(predictions_df),
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

        # Create DataFrame with alternating Train/Val rows
        formatted_results = []
        for result in results:
            fold = result['fold']
            # Train row
            formatted_results.append({
                'Fold': f"{fold} Train",
                'rmse': result['rmse_train'],
                'otpa': result['otpa_train'],
                'sddr': result['sddr_train']
            })
            # Val row
            formatted_results.append({
                'Fold': f"{fold} Val",
                'rmse': result['rmse_val'],
                'otpa': result['otpa_val'],
                'sddr': result['sddr_val']
            })
        
        m = pd.DataFrame(formatted_results)
        
        # Calculate mean and std for train and val separately
        train_rows = m[m['Fold'].str.contains('Train')]
        val_rows = m[m['Fold'].str.contains('Val')]
        
        mean_train = pd.DataFrame([{
            'Fold': 'Mean Train',
            'rmse': train_rows['rmse'].mean(),
            'otpa': train_rows['otpa'].mean(),
            'sddr': train_rows['sddr'].mean()
        }])
        
        std_train = pd.DataFrame([{
            'Fold': 'Std Train',
            'rmse': train_rows['rmse'].std(),
            'otpa': train_rows['otpa'].std(),
            'sddr': train_rows['sddr'].std()
        }])
        
        mean_val = pd.DataFrame([{
            'Fold': 'Mean Val',
            'rmse': val_rows['rmse'].mean(),
            'otpa': val_rows['otpa'].mean(),
            'sddr': val_rows['sddr'].mean()
        }])
        
        std_val = pd.DataFrame([{
            'Fold': 'Std Val',
            'rmse': val_rows['rmse'].std(),
            'otpa': val_rows['otpa'].std(),
            'sddr': val_rows['sddr'].std()
        }])
        
        m = pd.concat([m, mean_train, std_train, mean_val, std_val], ignore_index=True)
        return m

    def evaluate(self):
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
                'sddr': train_metric['sddr']
            },
            {
                'Split': 'Test',
                'rmse': test_metric['rmse'],
                'otpa': test_metric['otpa'],
                'sddr': test_metric['sddr']
            }
        ])
        
        return results