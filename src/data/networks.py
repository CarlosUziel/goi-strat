from typing import Optional

import networkx as nx
import pandas as pd


def get_node_metrics(
    graph: nx.Graph,
    distance_param: str,
    sim_param: str,
    central_node: Optional[str] = None,
) -> pd.DataFrame:
    """Compute node-level graph metrics such as centrality measures.

    IMPORTANT: To be used only for undirected connected graphs.

    Args:
        graph: Graph to compute metrics for.
        central_node: Node to calculate shortest path to from all other nodes.
        distance_param: Node attribute to use for those metrics that accept a distance
            (dissimilarity) measure.
        sim_param: Node attribute to use for those metrics that accept a edge strength
            (similarity) measure.

    Returns:
        A pandas DataFrame object with metrics per node.
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
