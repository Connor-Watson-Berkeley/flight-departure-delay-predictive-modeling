"""
graph_features.py - Graph-based feature engineering for flight delay prediction

Provides GraphFeaturesEstimator for use in Spark ML pipelines.
Computes PageRank features (weighted and unweighted) from flight network graph.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, broadcast
from pyspark.ml.base import Estimator, Model
from graphframes import GraphFrame
from datetime import datetime


class GraphFeaturesModel(Model):
    """Model returned by GraphFeaturesEstimator after fitting"""
    
    def __init__(self, pagerank_scores=None, origin_col="origin", dest_col="dest"):
        super(GraphFeaturesModel, self).__init__()
        self.pagerank_scores = pagerank_scores
        self.origin_col = origin_col
        self.dest_col = dest_col
        # Pre-broadcast the pagerank_scores for efficient reuse across multiple transform calls
        # This avoids creating a new broadcast variable on every transform() call
        if pagerank_scores is not None:
            self.pagerank_broadcast = broadcast(pagerank_scores)
        else:
            self.pagerank_broadcast = None
    
    def _transform(self, df):
        """Join PageRank scores to input DataFrame"""
        if self.pagerank_broadcast is None:
            raise ValueError("Model must be fitted before transform()")
        
        # Prepare pagerank lookup tables with renamed columns for origin and dest
        # This allows us to do a single join instead of two sequential joins
        origin_pagerank = (
            self.pagerank_broadcast
            .select(
                col("airport").alias(f"{self.origin_col}_lookup"),
                col("pagerank_weighted").alias("origin_pagerank_weighted"),
                col("pagerank_unweighted").alias("origin_pagerank_unweighted")
            )
        )
        
        dest_pagerank = (
            self.pagerank_broadcast
            .select(
                col("airport").alias(f"{self.dest_col}_lookup"),
                col("pagerank_weighted").alias("dest_pagerank_weighted"),
                col("pagerank_unweighted").alias("dest_pagerank_unweighted")
            )
        )
        
        # Join both origin and dest PageRank scores in a single pass
        # Using left joins to preserve all rows even if airport not in pagerank_scores
        df_with_features = (
            df
            .join(
                origin_pagerank,
                col(self.origin_col) == col(f"{self.origin_col}_lookup"),
                "left"
            )
            .drop(f"{self.origin_col}_lookup")
            .join(
                dest_pagerank,
                col(self.dest_col) == col(f"{self.dest_col}_lookup"),
                "left"
            )
            .drop(f"{self.dest_col}_lookup")
        )
        
        # Fill NULL PageRank values with 0 (for airports not in training graph) in a single pass
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
        # Use a single fillna call with a dictionary instead of a loop
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
        """Build graph from training data and compute PageRank scores"""
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