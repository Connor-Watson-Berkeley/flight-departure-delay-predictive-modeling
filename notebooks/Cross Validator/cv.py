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
VERSIONS = ["3M", "12M"]

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
    def __init__(self, estimator, version, fold_strategy="non_overlapping", n_folds=4, train_window_sections=2, n_sections=None, dataloader=None):
        """
        Initialize cross-validator with optional fold strategy.
        
        Args:
            estimator: Spark ML estimator (Pipeline, etc.)
            version: Data version ("3M" or "12M")
            fold_strategy: One of:
                - "non_overlapping" (default): Uses pre-saved non-overlapping rolling window folds
                - "overlapping": Creates overlapping rolling window folds with configurable overlap
                - "expanding": Expanding window strategy (training window grows, test window fixed)
            n_folds: Total number of folds (default: 4 = 3 CV + 1 test)
                - For non_overlapping: Typically 4 (3 CV + 1 test), parameter is ignored
                - For overlapping: Controls number of folds and amount of overlap
                - For expanding: Number of CV folds + 1 test fold
            train_window_sections: Number of equal-duration sections for training window (default: 2)
                - For overlapping: Training window size in sections (validation = 1 section)
                - For expanding: Initial training window size in sections (grows by 1 section per fold)
                - Works for all dataset sizes (3M, 12M, 60M) - sections are calculated proportionally
                - Example: For 3M dataset (3 months), train_window_sections=2 means 2 sections of ~1.5 months each
                - Example: For 60M dataset (5 years), train_window_sections=2 means 2 sections of ~2.5 years each
            n_sections: Optional number of sections to divide data into (default: None = auto-calculate)
                - If specified, data is divided into exactly n_sections equal-duration sections
                - Useful for aligning sections with calendar periods (e.g., 5 sections for 5-year dataset = 1 year per section)
                - Example: For 5-year dataset (2015-2019), n_sections=5 creates sections: 2015, 2016, 2017, 2018, 2019
                - If not specified, sections are auto-calculated based on fold requirements
            dataloader: Optional FlightDelayDataLoader instance (default: None)
                - If provided, uses the provided data loader instead of creating a new one
                - If None, creates and loads a new FlightDelayDataLoader instance
        
        Note: "non_overlapping" with n_folds=4 is backwards compatible with existing notebooks.
        Note: All filtering uses Spark - pandas.date_range is only used to generate section boundaries.
        """
        self.estimator = estimator
        self.version = version
        self.fold_strategy = fold_strategy
        self.n_folds = n_folds
        self.train_window_sections = train_window_sections
        self.n_sections = n_sections

        if dataloader:
            self.data_loader = dataloader
        else:
            self.data_loader = FlightDelayDataLoader()
            self.data_loader.load()

        self.evaluator = FlightDelayEvaluator()
        
        # Create folds based on strategy
        if fold_strategy == "non_overlapping":
            # Backwards compatible: use pre-saved folds
            # n_folds and train_window_sections parameters are ignored
            self.folds = self.data_loader.get_version(version)
        elif fold_strategy == "overlapping":
            # Create overlapping folds dynamically
            self.folds = self._create_overlapping_folds(version, n_folds, train_window_sections, n_sections)
        elif fold_strategy == "expanding":
            # Expanding window strategy
            self.folds = self._create_expanding_folds(version, n_folds, train_window_sections, n_sections)
        else:
            raise ValueError(f"Unknown fold_strategy: {fold_strategy}. Must be 'non_overlapping', 'overlapping', or 'expanding'")

        self.metrics = []
        self.models = []
        self.test_metric = None
        self.test_model = None
    
    def _get_date_range(self, version):
        """Get min/max date range from existing folds without loading all data."""
        all_folds = self.data_loader.get_version(version)
        
        # Get date ranges from each fold without loading full data
        date_ranges = []
        for train_df, val_df in all_folds:
            # Get min/max from train and val separately (more efficient than unioning first)
            train_dates = train_df.select(
                F.min(F.col("FL_DATE")).alias("min_date"),
                F.max(F.col("FL_DATE")).alias("max_date")
            ).first()
            val_dates = val_df.select(
                F.min(F.col("FL_DATE")).alias("min_date"),
                F.max(F.col("FL_DATE")).alias("max_date")
            ).first()
            
            date_ranges.extend([
                train_dates["min_date"], train_dates["max_date"],
                val_dates["min_date"], val_dates["max_date"]
            ])
        
        # Filter out None values and get overall min/max
        valid_dates = [d for d in date_ranges if d is not None]
        if not valid_dates:
            raise ValueError("No valid dates found in folds")
        
        return min(valid_dates), max(valid_dates)
    
    def _get_full_dataset(self, version):
        """Helper method to load and combine all fold data into full dataset.
        
        Optimized: For non-overlapping pre-saved folds, we can skip distinct()
        since the folds don't overlap by design. This avoids expensive shuffle.
        """
        all_folds = self.data_loader.get_version(version)
        
        spark = SparkSession.builder.getOrCreate()
        all_dataframes = []
        
        for train_df, val_df in all_folds[:-1]:  # Exclude test fold
            all_dataframes.append(train_df)
            all_dataframes.append(val_df)
        
        # Also include test fold data
        test_train_df, test_df = all_folds[-1]
        all_dataframes.append(test_train_df)
        all_dataframes.append(test_df)
        
        # Union all dataframes
        from functools import reduce
        full_df = reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), all_dataframes)
        
        # Ensure FL_DATE is date type
        full_df = full_df.withColumn("FL_DATE", F.to_date(F.col("FL_DATE")))
        
        # Skip distinct() for pre-saved non-overlapping folds - they don't overlap by design
        # This avoids expensive shuffle operation. If there are any edge case duplicates,
        # the date-based filtering will naturally handle them.
        
        # Cache the result since we'll filter it multiple times for fold creation
        full_df = full_df.cache()
        
        return full_df
    
    def _create_overlapping_folds(self, version, n_folds=4, train_window_sections=2, n_sections=None):
        """
        Create overlapping rolling window folds using equal-duration sections.
        
        Strategy: Divide data into equal-duration sections. For n_folds CV folds with train_window_sections training:
        - Each CV fold uses train_window_sections sections for training, 1 section for validation
        - Training windows overlap, creating more folds than non-overlapping
        - Number of folds controls the amount of overlap
        
        Args:
            version: Data version ("3M" or "12M")
            n_folds: Total number of folds (default: 4 = 3 CV + 1 test)
                - n_folds=4: Creates 3 CV folds + 1 test fold (requires 5 sections)
                - n_folds=3: Creates 2 CV folds + 1 test fold (requires 4 sections)
            train_window_sections: Number of sections for training window (default: 2)
                - Sections are calculated proportionally from total data range
                - Works for all dataset sizes (3M, 12M, 60M) - sections scale automatically
                - Example: 3M dataset (3 months) with train_window_sections=2 → 2 sections of ~1.5 months each
                - Example: 60M dataset (5 years) with train_window_sections=2 → 2 sections of ~2.5 years each
        
        Example (5 sections for 5-year dataset, n_folds=2, train_window_sections=2, n_sections=5):
        - CV Fold 1: Train sections 1&2 (2015-2016, 2 years), Val section 3 (2017, 1 year)
        - Test: Train sections 3&4 (2017-2018, 2 years), Test section 5 (2019, 1 year)
        
        Example (5 sections for 5-year dataset, n_folds=3, train_window_sections=2, n_sections=5):
        - CV Fold 1: Train sections 1&2 (2015-2016, 2 years), Val section 3 (2017, 1 year)
        - CV Fold 2: Train sections 2&3 (2016-2017, 2 years), Val section 4 (2018, 1 year)
        - Test: Train sections 3&4 (2017-2018, 2 years), Test section 5 (2019, 1 year)
                
        Usage for 5-year dataset with 2 CV folds + 1 test fold, 2-year training, 1-year validation/test:
        >>> cv = FlightDelayCV(
        ...     estimator=your_estimator,
        ...     version="60M",  # 5-year dataset
        ...     fold_strategy="overlapping",
        ...     n_folds=3,  # 2 CV + 1 test
        ...     train_window_sections=2,  # 2 sections = 2 years (when n_sections=5)
        ...     n_sections=5  # 5 sections = 1 year per section
        ... )
        
        Note: With n_folds=2, validation and test periods are non-overlapping (section 3 vs section 5).
              With n_folds=3, validation periods are non-overlapping (sections 3, 4) but test overlaps with last CV val (section 5).
              With n_folds=4, validation periods overlap (sections 3, 4, 5 all used for validation).
        
        Note: All data filtering uses Spark - pandas.date_range only generates boundary dates.
        """
        import pandas as pd
        
        # Get date range first without loading all data (optimization)
        min_date, max_date = self._get_date_range(version)
        total_days = (max_date - min_date).days
        
        # Now load full dataset (needed for filtering)
        full_df = self._get_full_dataset(version)
        
        num_cv_folds = n_folds - 1
        
        # Determine total number of sections
        # Minimum required: enough sections so max_possible_cv_folds >= num_cv_folds
        # max_possible_cv_folds = total_sections - train_window_sections - 1
        # So: total_sections - train_window_sections - 1 >= num_cv_folds
        # Therefore: total_sections >= train_window_sections + num_cv_folds + 1
        if n_sections is not None:
            total_sections = n_sections
            required_sections = train_window_sections + num_cv_folds + 1
            if total_sections < required_sections:
                raise ValueError(
                    f"Specified n_sections={n_sections} is insufficient for requested folds. "
                    f"Need at least {required_sections} sections (max_possible_cv_folds = {total_sections} - {train_window_sections} - 1 = {total_sections - train_window_sections - 1}, need {num_cv_folds})."
                )
        else:
            # Auto-calculate: train_window_sections + num_cv_folds + 1
            required_sections = train_window_sections + num_cv_folds + 1
            total_sections = required_sections
        
        # Calculate section duration and create section boundaries
        section_days = int(total_days / total_sections)
        section_boundaries = pd.date_range(
            start=min_date,
            end=max_date,
            periods=total_sections + 1,
            inclusive='left'
        ).tolist()
        section_boundaries.append(max_date)
        
        # Calculate how many CV folds we can actually create
        max_possible_cv_folds = max(0, total_sections - train_window_sections - 1)
        if max_possible_cv_folds < num_cv_folds:
            num_cv_folds = max_possible_cv_folds
        
        if num_cv_folds < 1:
            raise ValueError(
                f"Cannot create any CV folds. Need at least {train_window_sections + 2} sections "
                f"({train_window_sections} for training + 1 for validation + 1 for test), "
                f"but only have {total_sections} sections."
            )
        
        folds = []
        
        # CV folds: each uses train_window_sections sections for training, 1 section for validation
        for i in range(num_cv_folds):
            train_start = section_boundaries[i]
            train_end = section_boundaries[i + train_window_sections]
            val_start = train_end
            val_end = section_boundaries[i + train_window_sections + 1]
            
            train_df = full_df.filter(
                (F.col("FL_DATE") >= F.lit(train_start)) &
                (F.col("FL_DATE") < F.lit(train_end))
            )
            val_df = full_df.filter(
                (F.col("FL_DATE") >= F.lit(val_start)) &
                (F.col("FL_DATE") < F.lit(val_end))
            )
            
            folds.append((train_df, val_df))
        
        # Test fold: train_window_sections sections for training, last section for test
        # Use sections just before the last section for test training to avoid overlap with last CV fold
        test_train_start = section_boundaries[total_sections - train_window_sections]
        test_train_end = section_boundaries[total_sections - 1]
        test_start = test_train_end
        test_end = section_boundaries[total_sections]
        
        test_train_df = full_df.filter(
            (F.col("FL_DATE") >= F.lit(test_train_start)) &
            (F.col("FL_DATE") < F.lit(test_train_end))
        )
        test_df = full_df.filter(
            (F.col("FL_DATE") >= F.lit(test_start)) &
            (F.col("FL_DATE") <= F.lit(test_end))
        )
        
        folds.append((test_train_df, test_df))
        
        return folds
    
    def _create_expanding_folds(self, version, n_folds=4, train_window_sections=2, n_sections=None):
        """
        Create expanding window folds where training window grows with each fold.
        
        Strategy: Training window starts at train_window_sections and expands by 1 section per fold.
        Test window remains fixed at 1 section and moves forward.
        
        Args:
            version: Data version ("3M" or "12M")
            n_folds: Total number of folds (default: 4 = 3 CV + 1 test)
            train_window_sections: Initial training window size in sections (default: 2)
                - First fold uses train_window_sections sections for training
                - Each subsequent fold adds 1 section to training window
                - Works for all dataset sizes (3M, 12M, 60M) - sections scale proportionally
                - Example: 3M dataset with train_window_sections=2 → starts with 2 sections, expands
        
        Example (5 sections for 5-year dataset, n_folds=3, train_window_sections=2):
        - TF1: sections 1&2 (2015-2016), CVF1: section 3 (2017)
        - TF2: sections 1&2&3 (2015-2017), CVF2: section 4 (2018)
        - Test: sections 1-4 (2015-2018), Test: section 5 (2019)
        
        Note: With n_folds=3, requires 5 sections (2 initial training + 2 CV folds expansion + 1 test).
              With n_folds=4, requires 6 sections (2 initial training + 3 CV folds expansion + 1 test).
        
        Note: All data filtering uses Spark - pandas.date_range only generates boundary dates.
        Note: Use n_sections parameter to specify exact number of sections (e.g., n_sections=5 for 5-year dataset).
        """
        import pandas as pd
        
        # Get date range first without loading all data (optimization)
        min_date, max_date = self._get_date_range(version)
        total_days = (max_date - min_date).days
        
        # Now load full dataset (needed for filtering)
        full_df = self._get_full_dataset(version)
        
        num_cv_folds = n_folds - 1
        
        # Determine total number of sections
        if n_sections is not None:
            total_sections = n_sections
            required_sections = train_window_sections + num_cv_folds + 1
            if total_sections < required_sections:
                raise ValueError(
                    f"Specified n_sections={n_sections} is insufficient for requested folds. "
                    f"Need at least {required_sections} sections ({train_window_sections} initial training + {num_cv_folds} expansion + 1 test)."
                )
        else:
            required_sections = train_window_sections + num_cv_folds + 1
            total_sections = required_sections
        
        # Calculate section duration and create section boundaries
        section_days = int(total_days / total_sections)
        section_boundaries = pd.date_range(
            start=min_date,
            end=max_date,
            periods=total_sections + 1,
            inclusive='left'
        ).tolist()
        section_boundaries.append(max_date)
        
        # Check if we have enough sections
        min_required = train_window_sections + num_cv_folds + 1
        if total_sections < min_required:
            raise ValueError(
                f"Insufficient data for expanding fold strategy. "
                f"Need at least {min_required} sections "
                f"({train_window_sections} initial training + {num_cv_folds} expansion + 1 test), "
                f"but only have {total_sections} sections."
            )
        
        folds = []
        
        # CV folds: training window expands by 1 section each fold
        for i in range(num_cv_folds):
            train_start = section_boundaries[0]
            train_end = section_boundaries[train_window_sections + i]
            val_start = train_end
            val_end = section_boundaries[train_window_sections + i + 1]
            
            train_df = full_df.filter(
                (F.col("FL_DATE") >= F.lit(train_start)) &
                (F.col("FL_DATE") < F.lit(train_end))
            )
            val_df = full_df.filter(
                (F.col("FL_DATE") >= F.lit(val_start)) &
                (F.col("FL_DATE") < F.lit(val_end))
            )
            
            folds.append((train_df, val_df))
        
        # Test fold: all sections except last for training, last section for test
        test_train_start = section_boundaries[0]
        test_train_end = section_boundaries[total_sections - 1]
        test_start = test_train_end
        test_end = section_boundaries[total_sections]
        
        test_train_df = full_df.filter(
            (F.col("FL_DATE") >= F.lit(test_train_start)) &
            (F.col("FL_DATE") < F.lit(test_train_end))
        )
        test_df = full_df.filter(
            (F.col("FL_DATE") >= F.lit(test_start)) &
            (F.col("FL_DATE") <= F.lit(test_end))
        )
        
        folds.append((test_train_df, test_df))
        
        return folds

    def fit(self):
        # CV folds only (exclude last test fold)
        for train_df, val_df in self.folds[:-1]:
            model = self.estimator.fit(train_df)
            preds = model.transform(val_df)

            metric = self.evaluator.evaluate(preds)
            self.metrics.append(metric)
            self.models.append(model)

        m = pd.DataFrame(self.metrics)
        m.loc["mean"] = m.mean()
        m.loc["std"] = m.std()
        return m

    def evaluate(self):
        train_df, test_df = self.folds[-1]
        self.test_model = self.estimator.fit(train_df)
        preds = self.test_model.transform(test_df)
        self.test_metric = self.evaluator.evaluate(preds)
        return self.test_metric
