"""Network analysis utilities for processing graph data in genomic analyses.

This module provides functions for analyzing graph/network data, particularly
for computing node-level metrics in biological networks such as protein-protein
interaction networks.
"""

from typing import Optional

import networkx as nx
import pandas as pd


def get_node_metrics(
    graph: nx.Graph,
    distance_param: str,
    sim_param: str,
    central_node: Optional[str] = None,
) -> pd.DataFrame:
    """Compute important node-level metrics for an undirected graph.

    Calculates various centrality and importance measures for each node in the graph,
    including PageRank, degree centrality, harmonic centrality, betweenness centrality,
    and clustering coefficient.

    Args:
        graph: Undirected, connected NetworkX graph object
        distance_param: Edge attribute name representing dissimilarity/distance
            (used for shortest path calculations)
        sim_param: Edge attribute name representing similarity/strength
            (used for PageRank and clustering)
        central_node: Optional reference node for which to calculate distance to all
            other nodes in the graph

    Returns:
        pd.DataFrame: DataFrame where each row represents a node and columns
            represent different metrics

    Raises:
        AssertionError: If the graph is directed or not connected

    Note:
        This function only works with undirected connected graphs. For disconnected
        graphs, consider analyzing each connected component separately.
    """
    assert not nx.is_directed(graph) and nx.is_connected(
        graph
    ), "Input graph must be connected"

    metrics = pd.concat(
        [
            pd.Series(nx.pagerank(graph, weight=sim_param)).rename("page_rank"),
            pd.Series(dict(graph.degree())).rename("node_degree"),
            pd.Series(nx.degree_centrality(graph)).rename("degree_centrality"),
            pd.Series(nx.harmonic_centrality(graph, distance=distance_param)).rename(
                "harmonic_centrality"
            ),
            pd.Series(nx.betweenness_centrality(graph, weight=distance_param)).rename(
                "betweenness_centrality"
            ),
            pd.Series(nx.clustering(graph, weight=sim_param)).rename(
                "clustering_coefficient"
            ),
        ],
        axis=1,
    )

    if central_node is not None and central_node in graph.nodes:
        metrics = pd.concat(
            [
                pd.Series(nx.shortest_path_length(graph, central_node)).rename(
                    f"{central_node}_distance"
                ),
                metrics,
            ],
            axis=1,
        )

    return metrics
