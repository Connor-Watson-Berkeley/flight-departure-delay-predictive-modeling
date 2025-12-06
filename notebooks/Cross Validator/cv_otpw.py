"""
cv_otpw.py (simplified, OTPW-only, no parametrization)

Indri's hacky modified version of cv.py for OTPW data

Assumptions:
- Folds were created from split.py with N_FOLDS = 3 and CREATE_TEST_FOLD = True
- Therefore total fold indices written = 4:
    FOLD_1_VAL, FOLD_2_VAL, FOLD_3_VAL, FOLD_4_TEST
- Files live in:
    dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed
- File naming:
    OTPW_OTPW_{VERSION}_FOLD_{i}_{TRAIN|VAL|TEST}.parquet
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd


# -----------------------------
# HARD-CODED GLOBALS
# -----------------------------
FOLDER_PATH = "dbfs:/mnt/mids-w261/student-groups/Group_4_2/processed"
SOURCE = "OTPW"
VERSIONS = ["3M"]

# 3 CV folds + 1 test fold = 4 total fold indices
TOTAL_FOLDS = 4


# -----------------------------
# DATA LOADER (CUSTOM ONLY)
# -----------------------------
class FlightDelayDataLoader:
    def __init__(self):
        self.folds = {}
        self.numerical_features = [
            'HourlyPrecipitation',
            'HourlySeaLevelPressure',
            'HourlyAltimeterSetting',
            'HourlyWetBulbTemperature',
            'HourlyStationPressure',
            'HourlyWindDirection',
            'HourlyRelativeHumidity',
            'HourlyWindSpeed',
            'HourlyDewPointTemperature',
            'HourlyDryBulbTemperature',
            'HourlyVisibility',
            'CRS_ELAPSED_TIME',
            'DISTANCE',
            'ELEVATION',
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
# CROSS-VALIDATOR (NO PARAMS)
# -----------------------------
class FlightDelayCV:
    def __init__(self, estimator, version, dataloader = None):
        self.estimator = estimator
        self.version = version

        if dataloader:
            self.data_loader = dataloader
        else:
            self.data_loader = FlightDelayDataLoader()
            self.data_loader.load()

        self.evaluator = FlightDelayEvaluator()
        self.folds = self.data_loader.get_version(version)

        self.train_metrics = []
        self.val_metrics = []
        self.models = []
        self.test_train_metric = None
        self.test_metric = None
        self.test_model = None

    def fit(self):
        # CV folds only (exclude last test fold)
        for fold_idx, (train_df, val_df) in enumerate(self.folds[:-1]):
            model = self.estimator.fit(train_df)
            
            # Evaluate on training data
            train_preds = model.transform(train_df)
            train_metric = self.evaluator.evaluate(train_preds)
            self.train_metrics.append(train_metric)
            
            # Evaluate on validation data
            val_preds = model.transform(val_df)
            val_metric = self.evaluator.evaluate(val_preds)
            self.val_metrics.append(val_metric)
            
            self.models.append(model)

        # Create combined DataFrame with train and val metrics
        train_df = pd.DataFrame(self.train_metrics)
        val_df = pd.DataFrame(self.val_metrics)
        
        # Rename indices to include train/val labels
        train_df.index = [f"{i}-train" for i in range(len(train_df))]
        val_df.index = [f"{i}-val" for i in range(len(val_df))]
        
        # Interleave train and val rows
        combined_rows = []
        for i in range(len(self.train_metrics)):
            combined_rows.append(train_df.iloc[i])
            combined_rows.append(val_df.iloc[i])
        
        m = pd.DataFrame(combined_rows)
        
        # Add mean and std for train and val separately
        train_mean = train_df.mean()
        train_mean.name = "mean-train"
        train_std = train_df.std()
        train_std.name = "std-train"
        
        val_mean = val_df.mean()
        val_mean.name = "mean-val"
        val_std = val_df.std()
        val_std.name = "std-val"
        
        m = pd.concat([m, train_mean.to_frame().T, train_std.to_frame().T,
                       val_mean.to_frame().T, val_std.to_frame().T])
        
        return m

    def evaluate(self):
        # deprecated in favor of `test`
        # kept for backwards compatibility
        print("`evaluate()` is deprecated! Use `test()` instead")
        return self.test()
    
    def test(self):
        # does the same thing as evaluate
        train_df, test_df = self.folds[-1]
        self.test_model = self.estimator.fit(train_df)
        
        # Evaluate on training data
        train_preds = self.test_model.transform(train_df)
        self.test_train_metric = self.evaluator.evaluate(train_preds)
        
        # Evaluate on test data
        test_preds = self.test_model.transform(test_df)
        self.test_metric = self.evaluator.evaluate(test_preds)
        
        # Return both as a DataFrame
        results = pd.DataFrame([self.test_train_metric, self.test_metric],
                              index=["train", "test"])
        return results
