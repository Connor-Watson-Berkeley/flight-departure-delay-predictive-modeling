"""
Cross-validation utilities for flight delay prediction models.

This module provides classes for loading flight delay data in k-fold format,
evaluating predictions, and running cross-validation on estimators.
"""

from pyspark.sql.functions import col, when
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd


class FlightDelayDataLoader:
    """
    Data loader for flight delay prediction with k-fold cross-validation.

    Loads pre-split training and validation folds from parquet files.
    Supports multiple data versions (3M, 12M) with configurable number of folds.
    """

    def __init__(
        self,
        folder_path="dbfs:/student-groups/Group_4_2",
        n_folds=5,
        local_mode=False
    ):
        """
        Initialize the data loader.

        Args:
            folder_path: Path to folder containing parquet files
            n_folds: Number of cross-validation folds
            local_mode: If True, read from local filesystem (default: False for DBFS)
        """
        self.folder_path = folder_path
        self.n_folds = n_folds
        self.local_mode = local_mode
        self.folds = {}
        self.versions = ["3M", "12M"]
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
            'QUARTER',
            'DAY_OF_MONTH',
            'FLIGHTS',
            'DISTANCE',
            'YEAR',
            'MONTH',
            'origin_station_lat',
            'origin_station_lon',
            'origin_airport_lat',
            'origin_airport_lon',
            'origin_station_dis',
            'dest_station_lat',
            'dest_station_lon',
            'dest_airport_lat',
            'dest_airport_lon',
            'dest_station_dis',
            'LATITUDE',
            'LONGITUDE',
            'ELEVATION',
        ]

    def load_team_data(self, dataset_name):
        """
        Load a single dataset and cast numerical columns to double.

        Args:
            dataset_name: Name of the dataset (without .parquet extension)

        Returns:
            Spark DataFrame with numerical columns cast to double
        """
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        df = spark.read.parquet(f"{self.folder_path}/{dataset_name}.parquet")

        # Cast all numerical columns to double
        for num_col in self.numerical_features:
            if num_col in df.columns:
                df = df.withColumn(num_col, col(num_col).cast("double"))

        return df

    def load_version(self, version):
        """
        Load all folds for a specific data version.

        Args:
            version: Data version (e.g., "3M" or "12M")

        Returns:
            List of (train_df, val_df) tuples for each fold
        """
        folds = []
        for fold_idx in range(1, self.n_folds + 1):
            train_df = self.load_team_data(f"OTPW_{version}_FOLD_{fold_idx}_TRAIN")
            if fold_idx != self.n_folds:
                val_df = self.load_team_data(f"OTPW_{version}_FOLD_{fold_idx}_VAL")
                folds.append((train_df, val_df))
            else:
                test_df = self.load_team_data(f"OTPW_{version}_FOLD_{fold_idx}_TEST")
                folds.append((train_df, test_df))
        return folds

    def load(self):
        """Load all versions of the data."""
        for version in self.versions:
            version_data = self.load_version(version)
            self.folds[version] = version_data

    def get_version(self, version):
        """
        Get folds for a specific version.

        Args:
            version: Data version (e.g., "3M" or "12M")

        Returns:
            List of (train_df, val_df) tuples for the specified version
        """
        return self.folds[version]


class FlightDelayEvaluator:
    """
    Evaluator for flight delay prediction models.

    Calculates multiple metrics:
    - RMSE (Root Mean Square Error) for continuous delay prediction
    - OTPA (On-Time Prediction Accuracy) for binary classification (delay >= 15 min)
    - SDDR (Severe Delay Detection Rate / Recall) for severe delays (delay >= 60 min)
    - Precision, Recall, F1 for both OTPA and SDDR
    """

    def __init__(
        self,
        prediction_col="prediction",
        numeric_label_col="DEP_DELAY",
        binary_label_col="DEP_DEL15",
        severe_label_col="SEVERE_DEL60"
    ):
        """
        Initialize the evaluator.

        Args:
            prediction_col: Name of the prediction column
            numeric_label_col: Name of the continuous label column (delay in minutes)
            binary_label_col: Name of the binary label column (0 if delay < 15 min, 1 if >= 15 min)
            severe_label_col: Name of the severe delay label column (0 if delay < 60 min, 1 if >= 60 min)
        """
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
        """
        Calculate Root Mean Square Error.

        Args:
            predictions_df: DataFrame with predictions and labels

        Returns:
            RMSE value
        """
        return self.rmse_evaluator.evaluate(predictions_df)

    def _calculate_classification_metrics(self, predictions_df, threshold, label_col):
        """
        Calculate precision, recall, F1 for a binary classification task.

        Args:
            predictions_df: DataFrame with predictions and labels
            threshold: Threshold for converting predictions to binary (e.g., 15 or 60)
            label_col: Name of the binary label column

        Returns:
            Dictionary with TP, FP, TN, FN, precision, recall, F1, accuracy
        """
        # Create predicted binary column
        pred_binary_col = f"pred_binary_{threshold}"
        df = predictions_df.withColumn(
            pred_binary_col,
            F.when(F.col(self.prediction_col) >= threshold, 1).otherwise(0)
        )

        # Calculate confusion matrix components
        tp = df.filter((F.col(pred_binary_col) == 1) & (F.col(label_col) == 1)).count()
        fp = df.filter((F.col(pred_binary_col) == 1) & (F.col(label_col) == 0)).count()
        tn = df.filter((F.col(pred_binary_col) == 0) & (F.col(label_col) == 0)).count()
        fn = df.filter((F.col(pred_binary_col) == 0) & (F.col(label_col) == 1)).count()

        total = tp + fp + tn + fn

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

    def calculate_otpa_metrics(self, predictions_df):
        """
        Calculate OTPA (On-Time Prediction Accuracy).

        Evaluates binary classification for delays >= 15 minutes.

        Args:
            predictions_df: DataFrame with predictions and labels

        Returns:
            OTPA (accuracy) value
        """
        metrics = self._calculate_classification_metrics(
            predictions_df,
            threshold=15,
            label_col=self.binary_label_col
        )

        return metrics["accuracy"]

    def calculate_sddr_metrics(self, predictions_df):
        """
        Calculate SDDR (Severe Delay Detection Rate).

        SDDR is the recall for severe delays (>= 60 minutes), measuring the
        proportion of severely delayed flights correctly identified.

        Args:
            predictions_df: DataFrame with predictions and labels

        Returns:
            SDDR (recall) value
        """
        metrics = self._calculate_classification_metrics(
            predictions_df,
            threshold=60,
            label_col=self.severe_label_col
        )

        return metrics["recall"]

    def evaluate(self, predictions_df):
        """
        Evaluate predictions using all available metrics.

        Args:
            predictions_df: DataFrame with predictions and labels

        Returns:
            Dictionary with RMSE, OTPA, and SDDR metrics
        """
        # Calculate RMSE
        rmse = self.calculate_rmse(predictions_df)

        # Calculate OTPA metrics
        otpa = self.calculate_otpa_metrics(predictions_df)

        # Calculate SDDR metrics
        sddr = self.calculate_sddr_metrics(predictions_df)

        # Combine all metrics
        return {
            "rmse": rmse,
            "otpa": otpa,
            "sddr": sddr
        }


class FlightDelayCV:
    """
    Cross-validator for flight delay prediction models.

    Runs k-fold cross-validation on any estimator that implements
    fit() and transform() methods compatible with Spark ML.
    """

    def __init__(
        self,
        estimator,
        version,
        data_loader=None,
        evaluator=None
    ):
        """
        Initialize the cross-validator.

        Args:
            estimator: Model to evaluate (must have fit() and transform() methods)
            version: Data version to use (e.g., "3M" or "12M")
            data_loader: FlightDelayDataLoader instance (creates new one if None)
            evaluator: FlightDelayEvaluator instance (creates new one if None)
        """
        if data_loader is None:
            data_loader = FlightDelayDataLoader()
            data_loader.load()
        else:
            if not data_loader.folds:
                data_loader.load()
        self.data_loader = data_loader

        if evaluator is None:
            evaluator = FlightDelayEvaluator()
        self.evaluator = evaluator

        self.version = version
        self.estimator = estimator

        self.folds = self.data_loader.get_version(version)
        self.n_folds = len(self.folds)

        self.metrics = []
        self.models = []

        self.test_metric = None
        self.test_model = None

    def fit(self):
        """
        Fit the estimator on all training/validation folds.

        Trains a model on each fold and evaluates on the corresponding
        validation set. The last fold (test set) is reserved for final evaluation.

        Returns:
            Pandas DataFrame with metrics for each fold plus mean and std
        """
        # Only fit on training/validation folds, not the last test fold
        for fold_idx, (train_df, val_df) in enumerate(self.folds[:-1]):
            model = self.estimator.fit(train_df)
            predictions = model.transform(val_df)
            metric = self.evaluator.evaluate(predictions)

            self.metrics.append(metric)
            self.models.append(model)

        self.metrics_pd = pd.DataFrame(self.metrics)
        self.metrics_pd.loc['mean'] = self.metrics_pd.mean()
        self.metrics_pd.loc['std'] = self.metrics_pd.std()

        return self.metrics_pd

    def evaluate(self):
        """
        Evaluate the estimator on the final test fold.

        Trains a model on the test fold's training data and evaluates
        on the test set. Stores results in test_metric and test_model attributes.
        """
        # Evaluate on the test fold
        train_df, test_df = self.folds[-1]
        test_model = self.estimator.fit(train_df)
        predictions = test_model.transform(test_df)
        test_metric = self.evaluator.evaluate(predictions)

        self.test_metric = test_metric
        self.test_model = test_model
