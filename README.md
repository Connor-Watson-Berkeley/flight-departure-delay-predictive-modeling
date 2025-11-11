# flight-departure-delay-predictive-modeling
A Machine Learning at Scale project predicting flight departure delays using Databricks / PySpark for distributed training of ML predictive models.

# User Guide

## Pull Joined Dataset

TODO(Sid): Provide code on how to pull the joined dataset into a notebook

## Temporal Cross Validation

 Draft a user guide on how to pull and usethe cross validation data for use in model development. Nothing too verbose, just code snippits is fine

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
