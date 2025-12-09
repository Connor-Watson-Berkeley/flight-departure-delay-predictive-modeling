"""
precompute_features.py - Precompute graph and meta-model features for folds

Simple one-line integration for split.py to add graph features and meta-model predictions.
Similar to flight_lineage_features.add_flight_lineage_features().
"""

import importlib.util
from pyspark.sql import DataFrame

# Import dependencies from Databricks path
_base_path = "/Workspace/Shared/Team 4_2/flight-departure-delay-predictive-modeling/notebooks/Feature Engineering"

# Import graph_features
graph_features_path = f"{_base_path}/graph_features.py"
spec = importlib.util.spec_from_file_location("graph_features", graph_features_path)
graph_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_features)

# Import meta_model_estimator
meta_model_path = f"{_base_path}/meta_model_estimator.py"
spec = importlib.util.spec_from_file_location("meta_model_estimator", meta_model_path)
meta_model_estimator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta_model_estimator)


def add_precomputed_features(train_df: DataFrame, val_df: DataFrame,
                             model_ids: list[str] = None,
                             compute_graph: bool = True,
                             compute_meta_models: bool = True,
                             verbose: bool = True,
                             warm_start_scores: dict = None) -> tuple[DataFrame, DataFrame, dict]:
    """
    Add precomputed graph features and meta-model predictions to train and val DataFrames.
    
    This function computes:
    1. Graph features (PageRank) on training data, applied to both train and val
    2. Meta-model predictions (air_time, taxi_time, total_duration) on training data, applied to both
    
    CV-Safe: All computation uses training data only, then applied to validation/test.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation/test DataFrame
        model_ids: List of meta-model identifiers (e.g., ["RF_1", "RF_2", "XGB_1"])
                   Defaults to ["RF_1"] if None
        compute_graph: Whether to compute graph features (default True)
        compute_meta_models: Whether to compute meta-model predictions (default True)
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (train_df_with_features, val_df_with_features, pagerank_scores_dict)
               pagerank_scores_dict is {airport: {weighted: score, unweighted: score}} for warm start
               Returns None for pagerank_scores_dict if compute_graph=False
    
    Example:
        # Single model
        train_df, val_df, scores = add_precomputed_features(train_df, val_df, model_ids=["RF_1"])
        
        # Multiple models with warm start
        train_df, val_df, scores = add_precomputed_features(
            train_df, val_df, 
            model_ids=["RF_1", "RF_2"],
            warm_start_scores=previous_fold_scores
        )
    """
    if model_ids is None:
        model_ids = ["RF_1"]
    
    train_result = train_df
    val_result = val_df
    pagerank_scores_dict = None
    
    # Compute graph features
    if compute_graph:
        train_result, val_result, pagerank_scores_dict = graph_features.compute_pagerank_features(
            train_result, val_result, verbose=verbose, warm_start_scores=warm_start_scores
        )
    
    # Compute meta-model predictions
    if compute_meta_models:
        # Compute predictions for each model
        # Use raw features (use_preprocessed_features=False) since we're precomputing before pipeline preprocessing
        for model_id in model_ids:
            try:
                train_result, val_result = meta_model_estimator.compute_meta_model_predictions(
                    train_result, val_result, model_id=model_id, verbose=verbose, use_preprocessed_features=False
                )
            except Exception as e:
                if verbose:
                    print(f"  ⚠ ERROR: Meta-model {model_id} failed: {str(e)}")
                    print(f"  This may be due to resource constraints or missing features.")
                raise RuntimeError(f"Meta-model precomputation failed for {model_id}. Error: {str(e)}") from e
    
    return train_result, val_result, pagerank_scores_dict

