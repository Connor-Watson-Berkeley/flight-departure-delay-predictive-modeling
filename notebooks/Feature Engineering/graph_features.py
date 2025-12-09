"""
graph_features.py - Graph-based feature engineering for flight delay prediction

Provides GraphFeaturesEstimator for use in Spark ML pipelines.
Computes PageRank features (weighted and unweighted) from flight network graph.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, broadcast
from pyspark.ml.base import Estimator, Model
from pyspark.sql.types import FloatType
from datetime import datetime
from collections import defaultdict
from graphframes import GraphFrame


# -------------------------
# RDD-BASED PAGERANK IMPLEMENTATION (FAST)
# -------------------------
def _init_graph_rdd(edges_df, sc):
    """
    Initialize graph RDD from edges DataFrame.
    Returns: RDD of (node_id, (score, edges_dict)) where edges_dict maps neighbor -> weight
    
    Follows the homework 5 pattern but adapted for DataFrame input.
    """
    # Convert edges to RDD: (src, dst, weight)
    # Handle both cases: weight column exists or defaults to 1
    if 'weight' in edges_df.columns:
        edges_rdd = edges_df.rdd.map(lambda row: (row.src, row.dst, int(row.weight)))
    else:
        edges_rdd = edges_df.rdd.map(lambda row: (row.src, row.dst, 1))
    
    # Build adjacency list: aggregate edges by source node
    # For weighted PageRank: sum weights for duplicate edges
    # Result: (node, {neighbor: total_weight})
    adjacency = edges_rdd.map(lambda x: (x[0], {x[1]: x[2]})).reduceByKey(
        lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a.keys()) | set(b.keys())}
    )
    
    # Get all nodes (sources and destinations) to count N
    all_nodes = edges_rdd.flatMap(lambda x: [x[0], x[1]]).distinct()
    N = all_nodes.count()
    initial_score = 1.0 / N if N > 0 else 0.0
    initial_score_bc = sc.broadcast(initial_score)
    
    # Create graph RDD for nodes with edges: (node, (score, edges_dict))
    graph_with_edges = adjacency.map(lambda x: (x[0], (initial_score_bc.value, x[1])))
    
    # Find dangling nodes (destinations that are not sources)
    sources = adjacency.map(lambda x: x[0]).distinct()
    destinations = edges_rdd.map(lambda x: x[1]).distinct()
    dangling_nodes = destinations.subtract(sources).map(
        lambda x: (x, (initial_score_bc.value, {}))
    )
    
    # Combine all nodes
    graph_rdd = graph_with_edges.union(dangling_nodes)
    
    return graph_rdd, N


def _run_pagerank_rdd(graph_rdd, N, alpha=0.15, max_iter=10, verbose=True, sc=None, initial_scores=None, tol=1e-4):
    """
    Run PageRank using RDD implementation (much faster than GraphFrames).
    
    Follows the homework 5 pattern but uses functional approach instead of accumulators.
    Stops early if convergence is detected (change in scores < tolerance).
    
    Args:
        graph_rdd: RDD of (node_id, (score, edges_dict))
        N: Number of nodes
        alpha: Reset probability (default 0.15)
        max_iter: Maximum iterations (default 25, reasonable balance between accuracy and speed)
        verbose: Print progress
        sc: SparkContext
        initial_scores: Optional dict of {node_id: initial_score} for warm start (faster convergence)
        tol: Convergence tolerance - stop if max change in scores < tol (default 1e-4)
               Convergence is checked every 5 iterations (after iteration 5) to avoid expensive joins
    
    Returns:
        RDD of (node_id, pagerank_score)
    """
    if sc is None:
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
    
    # Set checkpoint directory for lineage breaking (required for checkpoint())
    # Use a temporary directory if not already set
    if sc.getCheckpointDir() is None:
        sc.setCheckpointDir("dbfs:/tmp/pagerank_checkpoint")
    
    # Broadcast parameters
    a = sc.broadcast(alpha)
    N_bc = sc.broadcast(N)
    
    # Warm start: if initial_scores provided, use them instead of uniform initialization
    if initial_scores is not None:
        initial_scores_bc = sc.broadcast(initial_scores)
        # Update graph with initial scores (keep edges_dict, update scores)
        graph_rdd = graph_rdd.map(
            lambda x: (x[0], (initial_scores_bc.value.get(x[0], 1.0 / N if N > 0 else 0.0), x[1][1]))
        )
    
    # Helper functions (matching homework 5 logic)
    def distribute_contributions(node):
        """Distribute node's score to neighbors weighted by edge weights"""
        node_id, (score, edges_dict) = node
        if not edges_dict or len(edges_dict) == 0:
            return []
        # Total outlinks INCLUDING weights (e.g., if node links to X twice, count = 2)
        total_links = sum(edges_dict.values())
        # Safety check: avoid division by zero (shouldn't happen with valid data)
        if total_links == 0:
            return []
        # Distribute proportionally by weight
        return [(neighbor, score * weight / total_links) for neighbor, weight in edges_dict.items()]
    
    def update_pagerank(node_and_contrib, dangling_mass):
        """Apply PageRank formula: PR = alpha/N + (1-alpha)*(contributions + dangling/N)"""
        node_id, ((score, edges_dict), new_contrib) = node_and_contrib
        received = new_contrib if new_contrib else 0.0
        # Safety check: N should never be 0 (checked in initialization), but be defensive
        if N_bc.value == 0:
            return (node_id, (0.0, edges_dict))
        new_score = a.value / N_bc.value + (1 - a.value) * (received + dangling_mass / N_bc.value)
        return (node_id, (new_score, edges_dict))
    
    # Cache the initial graph to avoid recomputation
    current_graph = graph_rdd.cache()
    
    for iteration in range(max_iter):
        iter_start = datetime.now()
        
        # Collect dangling mass (nodes with no outlinks)
        # Using functional approach instead of accumulators (more efficient in modern Spark)
        dangling_nodes = current_graph.filter(lambda x: not x[1][1] or len(x[1][1]) == 0)
        dangling_mass = dangling_nodes.map(lambda x: x[1][0]).sum()
        dangling_mass_bc = sc.broadcast(dangling_mass)
        
        # Distribute contributions to neighbors
        contribs = current_graph.flatMap(distribute_contributions)
        
        # Aggregate contributions by node
        aggregated = contribs.reduceByKey(lambda a, b: a + b)
        
        # Update PageRank scores with formula
        # CRITICAL: Create new graph first, then unpersist old one to avoid using unpersisted RDD
        new_graph = current_graph.leftOuterJoin(aggregated).map(
            lambda x: update_pagerank(x, dangling_mass_bc.value)
        ).cache()  # Cache the new graph for next iteration
        
        # Check convergence: compute max change in scores
        # CRITICAL: Check every 2 iterations after iteration 5 because each iteration doubles in cost
        # The convergence check (join) is expensive, but checking every 2 iterations balances early detection with overhead
        max_change = None
        if iteration >= 5 and (iteration + 1) % 2 == 0:  # Start checking after iteration 5, every 2 iterations
            # Compute absolute change for each node (expensive operation, but necessary to detect convergence)
            # Extract old scores before unpersisting
            old_scores = current_graph.map(lambda x: (x[0], x[1][0]))
            new_scores = new_graph.map(lambda x: (x[0], x[1][0]))
            score_changes = old_scores.join(new_scores).map(
                lambda x: abs(x[1][0] - x[1][1])  # |old_score - new_score|
            )
            max_change = score_changes.max()
        
        # Unpersist old graph after creating new one (free memory)
        if iteration > 0:
            current_graph.unpersist()
        
        current_graph = new_graph
        
        # CRITICAL: Repartition every 5 iterations to break RDD lineage and prevent exponential slowdown
        # Repartitioning forces a shuffle which breaks the lineage chain
        # This is simpler and more reliable than checkpointing
        # More frequent repartitioning (every 5 instead of 10) to combat exponential slowdown
        if (iteration + 1) % 5 == 0 and iteration > 0:
            if verbose:
                repart_start = datetime.now()
                print(f"  [{repart_start.strftime('%Y-%m-%d %H:%M:%S')}] Repartitioning at iteration {iteration + 1} to break lineage...")
            # Get current partition count and repartition (forces shuffle, breaks lineage)
            num_partitions = current_graph.getNumPartitions()
            current_graph = current_graph.repartition(num_partitions).cache()
            if verbose:
                repart_end = datetime.now()
                repart_duration = (repart_end - repart_start).total_seconds()
                print(f"  [{repart_end.strftime('%Y-%m-%d %H:%M:%S')}] ✓ Repartition complete (took {repart_duration:.2f}s)")
        
        iter_end = datetime.now()
        iter_duration = (iter_end - iter_start).total_seconds()
        timestamp = iter_end.strftime("%Y-%m-%d %H:%M:%S")  # Use end timestamp for logging
        
        # Progress logging with timestamps - print every iteration for debugging
        if verbose:
            # Print every iteration for first 15, then every 5 iterations to track progress
            should_print = (iteration + 1) <= 15 or (iteration + 1) % 5 == 0 or max_change is not None
            if should_print:
                # Only compute total_mass if we're logging (can be expensive - do less frequently)
                if (iteration + 1) <= 15 or (iteration + 1) % 10 == 0 or max_change is not None:
                    total_mass = current_graph.map(lambda x: x[1][0]).sum()
                    conv_msg = f", Max change={max_change:.2e}" if max_change is not None else ""
                    print(f"  [{timestamp}] Iteration {iteration + 1}/{max_iter}: Total mass={total_mass:.6f}, Dangling={dangling_mass:.6f}, Duration={iter_duration:.2f}s{conv_msg}")
                else:
                    # Just print iteration number and duration without total_mass
                    conv_msg = f", Max change={max_change:.2e}" if max_change is not None else ""
                    print(f"  [{timestamp}] Iteration {iteration + 1}/{max_iter}: Dangling={dangling_mass:.6f}, Duration={iter_duration:.2f}s{conv_msg}")
        
        # Check convergence - stop early if converged
        if max_change is not None and max_change < tol:
            if verbose:
                print(f"  ✓ Converged at iteration {iteration + 1} (max change={max_change:.2e} < {tol})")
            break
    
    # Extract final PageRank scores
    pagerank_rdd = current_graph.map(lambda x: (x[0], x[1][0]))
    
    return pagerank_rdd


# -------------------------
# REUSABLE GRAPH COMPUTATION FUNCTION
# -------------------------
def compute_pagerank_features(train_df, val_df, 
                              origin_col="origin", dest_col="dest",
                              reset_probability=0.15, max_iter=10,
                              checkpoint_dir="dbfs:/tmp/graphframes_checkpoint",
                              verbose=True, warm_start_scores=None):
    """
    Compute PageRank features on training data and join to both train and val DataFrames.
    
    Uses fast RDD-based PageRank implementation (much faster than GraphFrames).
    
    This function is designed for precomputing graph features during fold creation.
    It computes PageRank on training data only (CV-safe) and applies to both train and val.
    
    Args:
        train_df: Training DataFrame (used to build graph)
        val_df: Validation/test DataFrame (gets graph features joined)
        origin_col: Column name for origin airport
        dest_col: Column name for destination airport
        reset_probability: PageRank reset probability (default 0.15)
        max_iter: Maximum PageRank iterations (default 25, reasonable balance - convergence checked every 5 iterations)
        checkpoint_dir: Spark checkpoint directory (not used with RDD implementation, kept for compatibility)
        verbose: Whether to print progress messages
        warm_start_scores: Optional dict of {airport: {weighted: score, unweighted: score}} from previous fold
                          to speed up convergence (e.g., use fold 1 scores to initialize fold 2)
    
    Returns:
        tuple: (train_df_with_features, val_df_with_features) both with graph features joined
    """
    if verbose:
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Computing graph features (train: {train_df.count():,} rows)...")
    
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    
    # Build graph from training data only (CV-safe)
    # OPTIMIZATION: Use RDD reduceByKey instead of DataFrame groupBy for better performance
    # DataFrame groupBy is slower than RDD reduceByKey for this aggregation
    sc = spark.sparkContext
    edges_rdd = (
        train_df
        .select(origin_col, dest_col)
        .filter(
            col(origin_col).isNotNull() & 
            col(dest_col).isNotNull()
        )
        .rdd
        .map(lambda row: ((row[origin_col], row[dest_col]), 1))
        .reduceByKey(lambda a, b: a + b)  # Count flights per route (faster than DataFrame groupBy)
        .map(lambda x: (x[0][0], x[0][1], x[1]))  # Flatten to (src, dst, weight)
    )
    # Convert back to DataFrame
    edges = spark.createDataFrame(edges_rdd, ["src", "dst", "weight"])
    
    # Extract warm start scores if provided (only for weighted RDD implementation)
    initial_weighted = None
    if warm_start_scores is not None:
        initial_weighted = {airport: scores.get('weighted', None) 
                           for airport, scores in warm_start_scores.items() 
                           if scores.get('weighted') is not None}
        if verbose and initial_weighted:
            print(f"  Using warm start for weighted PageRank: {len(initial_weighted)} scores")
    
    # Compute unweighted PageRank using GraphFrames (more refined/optimized)
    # GraphFrames is highly optimized and doesn't need warm start
    if verbose:
        print("  Computing unweighted PageRank (GraphFrames)...")
    # Set checkpoint directory for GraphFrames
    sc.setCheckpointDir(checkpoint_dir)
    
    # Create vertices: all unique airports
    src_airports = edges.select(col("src").alias("id")).distinct()
    dst_airports = edges.select(col("dst").alias("id")).distinct()
    vertices = src_airports.union(dst_airports).distinct()
    
    # Create unweighted edges (no weight column - GraphFrames treats all edges equally)
    edges_unweighted = edges.select("src", "dst")
    
    # Create GraphFrame and run PageRank
    g_unweighted = GraphFrame(vertices, edges_unweighted)
    pagerank_unweighted_result = g_unweighted.pageRank(
        resetProbability=reset_probability,
        maxIter=max_iter
    )
    
    # Extract PageRank scores
    pagerank_unweighted_df = pagerank_unweighted_result.vertices.select(
        col("id").alias("airport"),
        col("pagerank").alias("pagerank_unweighted")
    )
    
    # Compute weighted PageRank using RDD implementation (supports native weights, no duplication needed)
    if verbose:
        print("  Computing weighted PageRank (RDD)...")
    graph_weighted, N_w = _init_graph_rdd(edges, sc)
    pagerank_weighted_rdd = _run_pagerank_rdd(
        graph_weighted, N_w, alpha=reset_probability, max_iter=max_iter, 
        verbose=verbose, sc=sc, initial_scores=initial_weighted
    )
    
    # Convert weighted RDD to DataFrame
    pagerank_weighted_df = spark.createDataFrame(
        pagerank_weighted_rdd.map(lambda x: (x[0], float(x[1]))),
        ["airport", "pagerank_weighted"]
    )
    
    # Combine PageRank scores
    pagerank_scores = pagerank_unweighted_df.join(pagerank_weighted_df, "airport", "outer")
    
    # Broadcast for efficient joins (small lookup table ~200 airports)
    broadcast_scores = broadcast(pagerank_scores)
    
    # Join origin and dest first, then cache to break lineage before prev_flight joins
    # This prevents Spark from having to recompute the origin/dest joins when adding prev_flight features
    train_with_base = (
        train_df
        .join(broadcast_scores, col(origin_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
        .drop("airport")
        .join(broadcast_scores, col(dest_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
        .drop("airport")
        .cache()  # Cache after base joins to break lineage
    )
    
    val_with_base = (
        val_df
        .join(broadcast_scores, col(origin_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
        .drop("airport")
        .join(broadcast_scores, col(dest_col) == col("airport"), "left")
        .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
        .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
        .drop("airport")
        .cache()  # Cache after base joins to break lineage
    )
    
    # Trigger cache materialization (lightweight - just count)
    _ = train_with_base.count()
    _ = val_with_base.count()
    
    # Add prev_flight graph features if prev_flight columns exist
    train_with_features = train_with_base
    val_with_features = val_with_base
    
    if "prev_flight_origin" in train_with_base.columns:
        train_with_features = (
            train_with_base
            .join(broadcast_scores, col("prev_flight_origin") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
            .drop("airport")
        )
        val_with_features = (
            val_with_base
            .join(broadcast_scores, col("prev_flight_origin") == col("airport"), "left")
            .withColumnRenamed("pagerank_weighted", "prev_flight_origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "prev_flight_origin_pagerank_unweighted")
            .drop("airport")
        )
    
    if "prev_flight_dest" in train_with_base.columns:
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
    
    # CRITICAL: Cache the final DataFrames to avoid recomputing joins on every action
    # This is especially important since these DataFrames will be used multiple times
    # (for meta-model training, pipeline fitting, etc.)
    train_with_features = train_with_features.cache()
    val_with_features = val_with_features.cache()
    
    # Safety check: Validate that graph features were successfully joined
    # Trigger caching by counting (lightweight action) - reuse counts for validation
    if verbose:
        # Check for missing airports (filled with 0.0) and trigger cache
        train_total = train_with_features.count()
        val_total = val_with_features.count()
        train_missing = train_with_features.filter(
            col("origin_pagerank_weighted") == 0.0
        ).count()
        val_missing = val_with_features.filter(
            col("origin_pagerank_weighted") == 0.0
        ).count()
        
        if train_missing > 0 or val_missing > 0:
            print(f"  ⚠ Warning: {train_missing:,}/{train_total:,} train rows and {val_missing:,}/{val_total:,} val rows have missing airports (filled with 0.0)")
            if val_missing > val_total * 0.1:  # More than 10% missing
                print(f"  ⚠ WARNING: High percentage of missing airports in validation data! This may indicate data issues.")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Graph features computed! (took {duration})")
    
    # Extract PageRank scores for warm start (convert to dict for easy reuse)
    # Note: Only weighted scores are used for warm start (RDD implementation)
    # Unweighted uses GraphFrames which doesn't support warm start
    # Convert to dict format: {airport: {weighted: score, unweighted: score}}
    pagerank_scores_dict = {}
    scores_rows = pagerank_scores.select("airport", "pagerank_weighted", "pagerank_unweighted").collect()
    for row in scores_rows:
        pagerank_scores_dict[row.airport] = {
            'weighted': row.pagerank_weighted,
            'unweighted': row.pagerank_unweighted  # Included for completeness, but not used for warm start
        }
    
    return train_with_features, val_with_features, pagerank_scores_dict


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
                 max_iter=10,
                 checkpoint_dir="dbfs:/tmp/graphframes_checkpoint"):
        super(GraphFeaturesEstimator, self).__init__()
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.reset_probability = reset_probability
        self.max_iter = max_iter
        self.checkpoint_dir = checkpoint_dir
        self._spark = SparkSession.builder.getOrCreate()
        
    def _build_edges(self, df):
        """Build edges DataFrame from flight data: nodes=airports, edges=flights"""
        # OPTIMIZATION: Use RDD reduceByKey instead of DataFrame groupBy for better performance
        # DataFrame groupBy is slower than RDD reduceByKey for this aggregation
        sc = self._spark.sparkContext
        edges_rdd = (
            df
            .select(self.origin_col, self.dest_col)
            .filter(
                col(self.origin_col).isNotNull() & 
                col(self.dest_col).isNotNull()
            )
            .rdd
            .map(lambda row: ((row[self.origin_col], row[self.dest_col]), 1))
            .reduceByKey(lambda a, b: a + b)  # Count flights per route (faster than DataFrame groupBy)
            .map(lambda x: (x[0][0], x[0][1], x[1]))  # Flatten to (src, dst, weight)
        )
        # Convert back to DataFrame
        edges = self._spark.createDataFrame(edges_rdd, ["src", "dst", "weight"])
        return edges
    
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
        
        # Graph features don't exist - compute them using fast RDD implementation
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Generating graph features...")
        
        sc = self._spark.sparkContext
        
        # Set checkpoint directory for GraphFrames
        if sc.getCheckpointDir() is None:
            sc.setCheckpointDir(self.checkpoint_dir)
        else:
            sc.setCheckpointDir(self.checkpoint_dir)
        
        # Build edges DataFrame
        edges = self._build_edges(df)
        
        # Compute unweighted PageRank using GraphFrames (matches precomputation for consistency)
        # GraphFrames is highly optimized for unweighted PageRank
        src_airports = edges.select(col("src").alias("id")).distinct()
        dst_airports = edges.select(col("dst").alias("id")).distinct()
        vertices = src_airports.union(dst_airports).distinct()
        edges_unweighted = edges.select("src", "dst")
        g_unweighted = GraphFrame(vertices, edges_unweighted)
        pagerank_unweighted_result = g_unweighted.pageRank(
            resetProbability=self.reset_probability,
            maxIter=self.max_iter
        )
        pagerank_unweighted_df = pagerank_unweighted_result.vertices.select(
            col("id").alias("airport"),
            col("pagerank").alias("pagerank_unweighted")
        )
        
        # Compute weighted PageRank using RDD implementation (supports native weights)
        graph_weighted, N_w = _init_graph_rdd(edges, sc)
        pagerank_weighted_rdd = _run_pagerank_rdd(
            graph_weighted, N_w, alpha=self.reset_probability,
            max_iter=self.max_iter, verbose=False, sc=sc
        )
        pagerank_weighted_df = self._spark.createDataFrame(
            pagerank_weighted_rdd.map(lambda x: (x[0], float(x[1]))),
            ["airport", "pagerank_weighted"]
        )
        
        # Combine PageRank scores into single DataFrame
        pagerank_scores = pagerank_unweighted_df.join(pagerank_weighted_df, "airport", "outer")

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

