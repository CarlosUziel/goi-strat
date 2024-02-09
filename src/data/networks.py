import logging
from collections import defaultdict
from itertools import chain, product
from typing import Callable, Optional

import networkx as nx
import pandas as pd
from stats.utils import overlap


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


def get_enrichment_network(
    degss_df: pd.DataFrame,
    sim_func: Callable = overlap,
    sim_th: Optional[float] = None,
    keep_giant: bool = False,
):
    """Given a list of differentially enriched gene sets (DEGSs), calculate the network
    based on genes overlap.

    Notes on similarity:
        https://enrichmentmap.readthedocs.io/en/latest/Parameters.html

        Jaccard vs. Overlap Coefficient

        The Overlap Coefficient is recommended when relations are expected to occur
        between large-size and small-size gene-sets, as in the case of the Gene
        Ontology.

        The Jaccard Coefficient is recommended in the opposite case. When the gene-sets
        are about the same size, Jaccard is about the half of the Overlap Coefficient
        for gene-set pairs with a small intersection, whereas it is about the same as
        the Overlap Coefficient for gene-sets with large intersections. When using the
        Overlap Coefficient and the generated map has several large gene-sets
        excessively connected to many other gene-sets, we recommend switching to the
        Jaccard Coefficient.

        Overlap Thresholds
        0.5 is moderately conservative, and is recommended for most of the analyses.
        0.3 is permissive, and might result in a messier map.

        Jaccard Thresholds
        0.5 is very conservative
        0.25 is moderately conservative

    Args:
        degss_df: DataFrame containing DEGSs information. Indeces are gene set names.
            Column "entrez_gene" is included and contains gene memberships.
        sim_func: A function that accepts two sets and returns a similarity coefficient.
        sim_th: Similarity coefficient threshold. Pairs of nodes with a similarity
            coefficient below this number will be disconnected.
        keep_giant: Whether to only keep the giant component.

    Returns:
        A networkx graph of DEGSs based on shared gene members.
    """
    # 1. Get genes for each gene set
    degss_genes = (
        degss_df["entrez_gene"].dropna().apply(lambda x: x.split("/")).to_dict()
    )

    # 2. Compute similarity for each possible pair to build adjacency matrix
    genes_sim = defaultdict(dict)
    for degss_0, degss_1 in product(degss_genes.keys(), repeat=2):
        genes_sim[degss_0][degss_1] = sim_func(
            degss_genes[degss_0], degss_genes[degss_1]
        )
    genes_sim_df = pd.DataFrame(genes_sim)

    # 2.1. Remove links between nodes that are too dissimilar
    if sim_th is not None:
        genes_sim_df[genes_sim_df < sim_th] = 0

    # 3. Generate graph
    graph = nx.from_pandas_adjacency(genes_sim_df)

    # 3.1. Remove self loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # 4. Add distance measure
    nx.set_edge_attributes(
        graph,
        {
            edge: 1 - attr
            for edge, attr in nx.get_edge_attributes(graph, "weight").items()
        },
        "distance",
    )

    # 5. Prune graph by removing isolated nodes and too-small components
    graph_components = list(nx.connected_components(graph))

    if keep_giant:
        graph = graph.subgraph(sorted(graph_components, key=len, reverse=True)[0])
    else:
        graph.remove_nodes_from(
            list(chain(*[list(comp) for comp in graph_components if len(comp) < 5]))
        )

    logging.info(
        f"Enrichment graph has {graph.number_of_nodes()} nodes, "
        f"out of all {len(degss_df)} input gene sets."
    )

    return graph
