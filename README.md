# flight-departure-delay-predictive-modeling
A Machine Learning at Scale project predicting flight departure delays using Databricks / PySpark for distributed training of ML predictive models.

# User Guide

## Pull Joined Dataset

TODO(Sid): Provide code on how to pull the joined dataset into a notebook

## Temporal Cross Validation

### Setup

```python
from cv import FlightDelayDataLoader, FlightDelayCV, FlightDelayEvaluator
```

### Load Cross-Validation Data

```python
# Initialize the data loader
loader = FlightDelayDataLoader(
    folder_path="dbfs:/student-groups/Group_4_2",
    n_folds=5
)
loader.load()

# Get folds for a specific version (e.g., "3M" or "12M")
folds = loader.get_version("3M")

# Access individual fold
for fold_idx, (train_df, val_df) in enumerate(folds):
    print(f"Fold {fold_idx + 1}: Train={train_df.count()}, Val={val_df.count()}")
```

### Run Cross-Validation

```python
# Initialize your estimator (e.g., linear regression pipeline)
from pyspark.ml import Pipeline

your_pipeline = [...]

cv = FlightDelayCV(
    estimator=your_pipeline,
    version="3M",
    data_loader=loader
)

# Fit on all folds
metrics_df = cv.fit()
print(metrics_df)

# Evaluate on test fold
cv.evaluate()
print(cv.test_metric)
```

### Access Metrics

```python
# Metrics from cross-validation folds
rmse_scores = cv.metrics_pd['rmse']
otpa_scores = cv.metrics_pd['otpa']
sddr_scores = cv.metrics_pd['sddr']

# Test set metrics
test_rmse = cv.test_metric['rmse']
test_otpa = cv.test_metric['otpa']
test_sddr = cv.test_metric['sddr']
```

# Dev Guide

## Save Tabular Data
```python
SECTION = "4"
NUMBER = "2"
GROUP_FOLDER_PATH = f"dbfs:/student-groups/Group_{SECTION}_{NUMBER}"
file_name = "YOUR_FILE_NAME_HERE"

df.write.parquet(f"{GROUP_FOLDER_PATH}/{file_name}.parquet")
```

# Appendix
