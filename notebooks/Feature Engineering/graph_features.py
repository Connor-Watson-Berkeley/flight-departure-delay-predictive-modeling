"""
graph_features.py - Graph-based feature engineering for flight delay prediction

Provides GraphFeaturesTransformer for use in Spark ML pipelines.
Computes PageRank features (weighted and unweighted) from flight network graph.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col
from pyspark.ml.base import Transformer
from graphframes import GraphFrame


class GraphFeaturesTransformer(Transformer):
    """
    Spark ML Transformer that adds graph-based features (PageRank) to flight data.
    
    In fit(): Builds graph from training data and computes PageRank scores
    In transform(): Joins PageRank scores to input DataFrame
    
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
        super(GraphFeaturesTransformer, self).__init__()
        self.origin_col = origin_col
        self.dest_col = dest_col
        self.reset_probability = reset_probability
        self.max_iter = max_iter
        self.checkpoint_dir = checkpoint_dir
        
        # Will store PageRank scores after fit()
        self.pagerank_scores = None
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
        from pyspark.sql.functions import explode, sequence, lit
        
        # Duplicate edges based on weight using sequence and explode
        edges_weighted = (
            edges
            .withColumn("seq", F.sequence(lit(0), col("weight").cast("int") - 1))
            .select("src", "dst", explode("seq").alias("_"))
            .select("src", "dst")
        )
        return edges_weighted
    
    def fit(self, df):
        """Build graph from training data and compute PageRank scores"""
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
        
        self.pagerank_scores = pr_unw.join(pr_w, "airport", "outer")
        
        return self
    
    def transform(self, df):
        """Join PageRank scores to input DataFrame"""
        if self.pagerank_scores is None:
            raise ValueError("Transformer must be fitted before transform()")
        
        # Join PageRank scores for origin and destination airports
        df_with_features = (
            df
            .join(
                self.pagerank_scores,
                col(self.origin_col) == col("airport"),
                "left"
            )
            .withColumnRenamed("pagerank_weighted", "origin_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "origin_pagerank_unweighted")
            .drop("airport")
            .join(
                self.pagerank_scores,
                col(self.dest_col) == col("airport"),
                "left"
            )
            .withColumnRenamed("pagerank_weighted", "dest_pagerank_weighted")
            .withColumnRenamed("pagerank_unweighted", "dest_pagerank_unweighted")
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
        for col_name in pagerank_cols:
            df_with_features = df_with_features.fillna({col_name: 0.0})
        
        return df_with_features

