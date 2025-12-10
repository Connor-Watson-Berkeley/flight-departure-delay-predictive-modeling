"""
graph_features.py - Graph-based feature engineering for flight delay prediction

Provides GraphFeaturesEstimator for use in Spark ML pipelines.
Computes PageRank features (weighted and unweighted) from flight network graph.

RESTORED VERSION (Dec 2024):
This file was restored to commit 3021702 (safe version before RDD implementation).
This version uses GraphFrames for both weighted and unweighted PageRank.

The broken preprocessing version is saved as graph_features_preprocessing_snapshot.py.
The RDD implementation version (9814117) is available in git history.

This version works with existing model pipelines and apply_engineered_features.py.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, broadcast
from pyspark.ml.base import Estimator, Model
from graphframes import GraphFrame
from datetime import datetime


# -------------------------
# REUSABLE GRAPH COMPUTATION FUNCTION
# -------------------------
def compute_pagerank_features(train_df, val_df, 
                              origin_col="origin", dest_col="dest",
                              reset_probability=0.15, max_iter=50,
                              checkpoint_dir="dbfs:/tmp/graphframes_checkpoint",
                              verbose=True):
    """
    Compute PageRank features on training data and join to both train and val DataFrames.
    
    This function is designed for precomputing graph features during fold creation.
    It computes PageRank on training data only (CV-safe) and applies to both train and val.
    
    Args:
        train_df: Training DataFrame (used to build graph)
        val_df: Validation/test DataFrame (gets graph features joined)
        origin_col: Column name for origin airport
        dest_col: Column name for destination airport
        reset_probability: PageRank reset probability (default 0.15)
        max_iter: Maximum PageRank iterations (default 50, higher since we only compute once)
        checkpoint_dir: Spark checkpoint directory for GraphFrames
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (train_df_with_features, val_df_with_features) both with graph features joined
    """
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing graph features (train: {train_df.count():,} rows)...")
    
    # Set checkpoint directory
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(checkpoint_dir)
    
    # Build graph from training data only (CV-safe)
    edges = (
        train_df
        .select(origin_col, dest_col)
        .filter(
            col(origin_col).isNotNull() & 
            col(dest_col).isNotNull()
        )
        .groupBy(origin_col, dest_col)
        .count()
        .withColumnRenamed(origin_col, "src")
        .withColumnRenamed(dest_col, "dst")
        .withColumnRenamed("count", "weight")
    )
    
    src_airports = edges.select(col("src").alias("id")).distinct()
    dst_airports = edges.select(col("dst").alias("id")).distinct()
    vertices = src_airports.union(dst_airports).distinct()
    
    g = GraphFrame(vertices, edges)
    
    # Compute unweighted PageRank
    edges_unweighted = edges.select("src", "dst")
    g_unweighted = GraphFrame(g.vertices, edges_unweighted)
    pagerank_unweighted = g_unweighted.pageRank(
        resetProbability=reset_probability,
        maxIter=max_iter
    )
    
    # Compute weighted PageRank (using duplication workaround)
    edges_weighted = (
        edges
        .withColumn("seq", F.sequence(F.lit(0), col("weight").cast("int") - 1))
        .select("src", "dst", F.explode("seq").alias("_"))
        .select("src", "dst")
    )
    g_weighted = GraphFrame(g.vertices, edges_weighted)
    pagerank_weighted = g_weighted.pageRank(
        resetProbability=reset_probability,
        maxIter=max_iter
    )
    
    # Combine PageRank scores
    pr_unw = pagerank_unweighted.vertices.select(
        col("id").alias("airport"),
        col("pagerank").alias("pagerank_unweighted")
    )
    pr_w = pagerank_weighted.vertices.select(
        col("id").alias("airport"),
        col("pagerank").alias("pagerank_weighted")
    )
    pagerank_scores = pr_unw.join(pr_w, "airport", "outer")
    
    # Broadcast for efficient joins
    broadcast_scores = broadcast(pagerank_scores)
    
    # Join to train_df
    train_with_features = (
        train_df
        .join(broadcast_scores, col(origin_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
        .drop("airport")
        .join(broadcast_scores, col(dest_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
        .drop("airport")
    )
    
    # Join to val_df
    val_with_features = (
        val_df
        .join(broadcast_scores, col(origin_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
        .drop("airport")
        .join(broadcast_scores, col(dest_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
        .drop("airport")
    )
    
    # Add prev_flight graph features if prev_flight columns exist
    if "prev_flight_origin" in train_with_features.columns:
        train_with_features = (
            train_with_features
            .join(broadcast_scores, col("prev_flight_origin") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
            .drop("airport")
        )
        val_with_features = (
            val_with_features
            .join(broadcast_scores, col("prev_flight_origin") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
            .drop("airport")
        )
    
    if "prev_flight_dest" in train_with_features.columns:
        train_with_features = (
            train_with_features
            .join(broadcast_scores, col("prev_flight_dest") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_dest_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_dest_pagerank_unweighted")
            .drop("airport")
        )
        val_with_features = (
            val_with_features
            .join(broadcast_scores, col("prev_flight_dest") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_dest_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_dest_pagerank_unweighted")
            .drop("airport")
        )
    
    # Fill NULL PageRank values with 0
    pagerank_cols = [
        "origin_pagerank_weighted", "origin_pagerank_unweighted",
        "dest_pagerank_weighted", "dest_pagerank_unweighted"
    ]
    if "prev_flight_origin_pagerank_weighted" in train_with_features.columns:
        pagerank_cols.extend([
            "prev_flight_origin_pagerank_weighted", "prev_flight_origin_pagerank_unweighted"
        ])
    if "prev_flight_dest_pagerank_weighted" in train_with_features.columns:
        pagerank_cols.extend([
            "prev_flight_dest_pagerank_weighted", "prev_flight_dest_pagerank_unweighted"
        ])
    
    fill_dict = {col_name: 0.0 for col_name in pagerank_cols}
    train_with_features = train_with_features.fillna(fill_dict)
    val_with_features = val_with_features.fillna(fill_dict)
    
    # Safety check: Validate that graph features were successfully joined
    if verbose:
        # Check for missing airports (filled with 0.0)
        train_missing = train_with_features.filter(
            col("origin_pagerank_weighted") == 0.0
        ).count()
        val_missing = val_with_features.filter(
            col("origin_pagerank_weighted") == 0.0
        ).count()
        train_total = train_with_features.count()
        val_total = val_with_features.count()
        
        if train_missing > 0 or val_missing > 0:
            print(f"  ⚠ Warning: {train_missing:,}/{train_total:,} train rows and {val_missing:,}/{val_total:,} val rows have missing airports (filled with 0.0)")
            if val_missing > val_total * 0.1:  # More than 10% missing
                print(f"  ⚠ WARNING: High percentage of missing airports in validation data! This may indicate data issues.")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Graph features computed! (took {duration})")
    
    return train_with_features, val_with_features


class GraphFeaturesModel(Model):
    """Model returned by GraphFeaturesEstimator after fitting"""
    
    def __init__(self, pagerank_scores=None, origin_col="origin", dest_col="dest"):
        super(GraphFeaturesModel, self).__init__()
        self.pagerank_scores = pagerank_scores
        self.origin_col = origin_col
        self.dest_col = dest_col
    
    def _transform(self, df):
        """Join PageRank scores to input DataFrame
        
        Optimizations:
        - Broadcasts PageRank scores (small lookup table ~200 airports) for faster joins
        - PageRank scores are already cached from fit() for reuse across multiple transforms
        - If pagerank_scores is None, assumes features are already present (pass-through mode)
        
        Pass-through Mode (pagerank_scores=None):
        - If features exist: Returns DataFrame unchanged (maximum efficiency, no-op)
        - If features missing: Raises error (safety check)
        
        Normal Mode (pagerank_scores not None):
        - Joins PageRank features normally (backwards compatible behavior)
        """
        # Pass-through mode: features already exist (precomputed in split.py)
        if self.pagerank_scores is None:
            # Verify required features exist
            required = ["origin_pagerank_weighted", "origin_pagerank_unweighted",
                       "dest_pagerank_weighted", "dest_pagerank_unweighted"]
            if all(f in df.columns for f in required):
                return df  # Features already present, no-op (maximum efficiency)
            else:
                raise ValueError("Graph features not found and model has no pagerank_scores. "
                               "Either precompute features or fit the model first.")
        
        # Broadcast PageRank scores for efficient joins (small lookup table ~200 airports)
        # This avoids shuffling the large DataFrame and speeds up joins significantly
        broadcast_scores = broadcast(self.pagerank_scores)
        
        # Join PageRank scores for origin and destination airports
        df_with_features = (
            df
            .join(
                broadcast_scores,
                col(self.origin_col) == col("airport"),
                "left"
            )
            .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
            .drop("airport")
            .join(
                broadcast_scores,
                col(self.dest_col) == col("airport"),
                "left"
            )
            .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
            .drop("airport")
        )
        
        # Add prev_flight graph features (if prev_flight columns exist)
        # Note: For rotations, prev_flight_dest = origin, but for jumps they differ
        # We add both prev_flight_origin and prev_flight_dest PageRank features for meta models
        # (main model only uses origin/dest, not prev_flight features)
        if "prev_flight_origin" in df_with_features.columns:
            # Add prev_flight_origin PageRank
            df_with_features = (
                df_with_features
                .join(
                    broadcast_scores,
                    col("prev_flight_origin") == col("airport"),
                    "left"
                )
                .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
                .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
                .drop("airport")
            )
        
        if "prev_flight_dest" in df_with_features.columns:
            # Add prev_flight_dest PageRank (needed for jumps where prev_flight_dest != origin)
            # For rotations, this will equal origin_pagerank_*, but for jumps it's different
            df_with_features = (
                df_with_features
                .join(
                    broadcast_scores,
                    col("prev_flight_dest") == col("airport"),
                    "left"
                )
                .withColumnRenamed("pagerank_weighted", "prev_flight_dest_pagerank_weighted")
                .withColumnRenamed("pagerank_unweighted", "prev_flight_dest_pagerank_unweighted")
                .drop("airport")
            )
        
        # Fill NULL PageRank values with 0 (for airports not in training graph)
        # TODO: Consider better imputation strategy. Isolated nodes in PageRank still receive
        #       PageRank from teleportation (reset probability). A new airport not in the training
        #       graph would theoretically have some PageRank if it were added as an isolated node.
        #       Current approach (0.0) assumes no connectivity, but we may want to impute with
        #       the theoretical minimum PageRank value (e.g., reset_probability / num_nodes) or
        #       the minimum observed PageRank from training data.
        pagerank_cols = [
            "origin_pagerank_weighted",
            "origin_pagerank_unweighted",
            "dest_pagerank_weighted",
            "dest_pagerank_unweighted"
        ]
        # Add prev_flight graph feature columns if they exist (for meta models)
        if "prev_flight_origin_pagerank_weighted" in df_with_features.columns:
            pagerank_cols.extend([
                "prev_flight_origin_pagerank_weighted",
                "prev_flight_origin_pagerank_unweighted"
            ])
        if "prev_flight_dest_pagerank_weighted" in df_with_features.columns:
            pagerank_cols.extend([
                "prev_flight_dest_pagerank_weighted",
                "prev_flight_dest_pagerank_unweighted"
            ])
        
        # Fill all NULL PageRank values in a single operation (more efficient than loop)
        fill_dict = {col_name: 0.0 for col_name in pagerank_cols}
        df_with_features = df_with_features.fillna(fill_dict)
        
        return df_with_features


class GraphFeaturesEstimator(Estimator):
    """
    Spark ML Estimator that adds graph-based features (PageRank) to flight data.
    
    In fit(): Builds graph from training data and computes PageRank scores
    Returns a GraphFeaturesModel that can transform DataFrames
    
    Features added:
    - origin_pagerank_weighted: Weighted PageRank of origin airport
    - origin_pagerank_unweighted: Unweighted PageRank of origin airport
    - dest_pagerank_weighted: Weighted PageRank of destination airport
    - dest_pagerank_unweighted: Unweighted PageRank of destination airport
    - prev_flight_origin_pagerank_weighted: Weighted PageRank of previous flight's origin (if prev_flight_origin exists, for meta models)
    - prev_flight_origin_pagerank_unweighted: Unweighted PageRank of previous flight's origin (if prev_flight_origin exists, for meta models)
    - prev_flight_dest_pagerank_weighted: Weighted PageRank of previous flight's destination (if prev_flight_dest exists, for meta models)
    - prev_flight_dest_pagerank_unweighted: Unweighted PageRank of previous flight's destination (if prev_flight_dest exists, for meta models)
    
    Note: For rotations, prev_flight_dest = origin, so prev_flight_dest_pagerank_* = origin_pagerank_*.
    However, for jumps (aircraft repositioning), prev_flight_dest != origin, so we need separate features.
    These prev_flight features are used by meta models but not by the main model.
    """
    
    def __init__(self, 
                 origin_col="origin",
                 dest_col="dest",
                 reset_probability=0.15,
                 max_iter=50,
                 checkpoint_dir="dbfs:/tmp/graphframes_checkpoint"):
        super(GraphFeaturesEstimator, self).__init__()
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.reset_probability = reset_probability
        self.max_iter = max_iter
        self.checkpoint_dir = checkpoint_dir
        self._spark = SparkSession.builder.getOrCreate()
        
    def _build_graph(self, df):
        """Build graph from flight data: nodes=airports, edges=flights"""
        # Create edges: (origin, dest) with count as weight
        edges = (
            df
            .select(self.origin_col, self.dest_col)
            .filter(
                col(self.origin_col).isNotNull() & 
                col(self.dest_col).isNotNull()
            )
            .groupBy(self.origin_col, self.dest_col)
            .count()
            .withColumnRenamed(self.origin_col, "src")
            .withColumnRenamed(self.dest_col, "dst")
            .withColumnRenamed("count", "weight")
        )
        
        # Create vertices: all unique airports
        src_airports = edges.select(col("src").alias("id")).distinct()
        dst_airports = edges.select(col("dst").alias("id")).distinct()
        vertices = src_airports.union(dst_airports).distinct()
        
        return GraphFrame(vertices, edges), edges
    
    def _create_weighted_edges(self, edges):
        """Create weighted graph using duplication workaround"""
        # Duplicate edges based on weight using sequence and explode
        edges_weighted = (
            edges
            .withColumn("seq", F.sequence(F.lit(0), col("weight").cast("int") - 1))
            .select("src", "dst", F.explode("seq").alias("_"))
            .select("src", "dst")
        )
        return edges_weighted
    
    def _fit(self, df):
        """
        Build graph from training data and compute PageRank scores.
        
        If graph features already exist in the DataFrame, skips computation and returns a pass-through model.
        This allows precomputed graph features (e.g., from split.py) to be used without recomputation.
        
        Backwards Compatible: If features don't exist, computes them normally (existing behavior).
        Maximum Efficiency: If features exist, returns pass-through model (no computation, no-op in transform).
        """
        # Check if graph features already exist
        required_features = [
            "origin_pagerank_weighted", "origin_pagerank_unweighted",
            "dest_pagerank_weighted", "dest_pagerank_unweighted"
        ]
        existing_features = [f for f in required_features if f in df.columns]
        
        if len(existing_features) == len(required_features):
            # All required features exist - return pass-through model (maximum efficiency)
            # This model will do no-op in _transform() if features are present
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Graph features already present, skipping computation")
            return GraphFeaturesModel(
                pagerank_scores=None,  # None indicates pass-through mode (no computation)
                origin_col=self.origin_col,
                dest_col=self.dest_col
            )
        
        # Graph features don't exist - compute them
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Generating graph features...")
        
        # Set checkpoint directory
        sc = self._spark.sparkContext
        sc.setCheckpointDir(self.checkpoint_dir)
        
        # Build graph
        g, edges = self._build_graph(df)
        
        # Compute unweighted PageRank (edges without weight column)
        edges_unweighted = edges.select("src", "dst")
        g_unweighted = GraphFrame(g.vertices, edges_unweighted)
        pagerank_unweighted = g_unweighted.pageRank(
            resetProbability=self.reset_probability,
            maxIter=self.max_iter
        )
        
        # Compute weighted PageRank (using duplication workaround)
        edges_weighted = self._create_weighted_edges(edges)
        g_weighted = GraphFrame(g.vertices, edges_weighted)
        pagerank_weighted = g_weighted.pageRank(
            resetProbability=self.reset_probability,
            maxIter=self.max_iter
        )
        
        # Combine PageRank scores into single DataFrame
        pr_unw = pagerank_unweighted.vertices.select(
            col("id").alias("airport"),
            col("pagerank").alias("pagerank_unweighted")
        )
        
        pr_w = pagerank_weighted.vertices.select(
            col("id").alias("airport"),
            col("pagerank").alias("pagerank_weighted")
        )
        
        pagerank_scores = pr_unw.join(pr_w, "airport", "outer")

        # Cache the PageRank scores since they'll be used multiple times in transform (origin + dest joins)
        pagerank_scores = pagerank_scores.cache()

        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Graph feature generation complete! (took {duration})")
        
        # Return a Model instance
        return GraphFeaturesModel(
            pagerank_scores=pagerank_scores,
            origin_col=self.origin_col,
            dest_col=self.dest_col
        )