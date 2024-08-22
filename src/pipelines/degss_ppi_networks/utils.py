import logging
from itertools import product
from multiprocessing import Lock
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
from r_wrappers.rbioapi import string_interactions_network, string_map_ids

from data.networks import get_node_metrics

logging.basicConfig()
logger = logging.getLogger(__name__)


def get_ppi_network(
    genes_symbol: Iterable[str],
    api_lock: Lock,  # type: ignore
    interaction_score: int = 500,
    network_type: str = "functional",
) -> nx.Graph:
    """Given a list of genes, get their PPI network by querying STRINGDB.add()

    Args:
        genes_symbol: Iterable of gene SYMBOL IDs.
        api_lock: Multiprocessing lock to access STRINGDB API.
        interaction_score: A minimum of interaction score for an interaction to be
            included.
        network_type: Either "functional" or "physical".

    Returns:
        A networkx graph object representing the PPI network.
    """
    # 1. Extract STRINGDB protein IDs of each gene symbol ID
    query_limit = 2000
    genes_partitions = [
        genes_symbol[i : i + query_limit]
        for i in range(0, len(genes_symbol), query_limit)
    ]

    proteins_map = (
        pd.concat(
            [
                string_map_ids(genes_partition, api_lock=api_lock)
                for genes_partition in genes_partitions
            ]
        )
        .reset_index(drop=True)
        .drop(columns=["queryIndex"])
    )

    # 2. Query STRINGDB for proteins interactions
    query_limit = 1000
    unique_proteins = proteins_map["stringId"].drop_duplicates().tolist()
    proteins_partitions = [
        unique_proteins[i : i + query_limit]
        for i in range(0, len(unique_proteins), query_limit)
    ]

    proteins_interactions = pd.concat(
        [
            string_interactions_network(
                x + y,
                required_score=interaction_score,
                network_type=network_type,
                api_lock=api_lock,
            )
            for (x, y) in product(proteins_partitions, repeat=2)
        ]
    ).reset_index(drop=True)

    # 2.1. Remove duplicates and add distance weights
    ppi_edges = (
        proteins_interactions.drop_duplicates(["stringId_A", "stringId_B"])
        .reset_index(drop=True)
        .rename(columns={"score": "weight"})
    )
    ppi_edges["distance"] = 1 - ppi_edges["weight"]

    # 3. Return PPI network
    return nx.from_pandas_edgelist(
        ppi_edges,
        target="preferredName_A",
        source="preferredName_B",
        edge_attr=True,
        create_using=nx.Graph,
    )


def process_degss_ppi_network(
    genes_symbol: Iterable[str],
    metadata_df: pd.DataFrame,
    save_path: Path,
    goi_symbol: str,
    k: int = 2,
    interaction_score: int = 500,
    network_type: str = "functional",
    api_lock: Lock = Lock(),  # type: ignore
) -> None:
    """Given multiple sets of differentially enriched gene sets (DEGSs), compute the PPI
    network of all involved genes.

    Additionally, return centrality metrics for the k-hop neighbourhood of a given gene.

    Args:
        genes_symbol: List of genes to include in the PPI network.
        metadata_df: Genes metadata, index is SYMBOL ID.
        save_path: Directory to save results to.
        goi_symbol: Gene of interest (GOI) symbol ID.
        k: Number of hops to produce the GOI neighbourhood.
        interaction_score: A minimum of interaction score for an interaction to be
            included.
        network_type: Either "functional" or "physical".
        api_lock: Multiprocessing lock to access STRINGDB API.
    """
    # 0. Setup
    logging.info(f"Number of unique genes extracted: {len(genes_symbol)}")
    genes_symbol.append(goi_symbol)

    # 2. Get PPI network
    ppi_graph = get_ppi_network(
        genes_symbol,
        api_lock,
        interaction_score=interaction_score,
        network_type=network_type,
    )

    assert not nx.is_directed(ppi_graph), "PPI graph should be undirected."

    # 2.1. Save PPI edges and all their attributes to disk
    nx.to_pandas_edgelist(ppi_graph).to_csv(save_path.joinpath("ppi_network_edges.csv"))

    # 2.2. Save PPI network to disk
    nx.write_weighted_edgelist(
        ppi_graph,
        save_path.joinpath("ppi_network.edgelist").open("wb"),
        delimiter=";",
    )

    # 3. Only keep giant component
    ppi_graph = ppi_graph.subgraph(
        sorted(nx.connected_components(ppi_graph), key=len, reverse=True)[0]
    )

    # 4. Compute node-level metrics
    node_metrics = get_node_metrics(
        ppi_graph,
        distance_param="distance",
        sim_param="weight",
        central_node=goi_symbol,
    )

    # 4.1. Add metadata (e.g. differential expression/methylation metrics)
    node_metrics = node_metrics.join(metadata_df)

    # 4.2. Save metrics to disk
    node_metrics.sort_values(by="page_rank", ascending=False).to_csv(
        save_path.joinpath("network_metrics.csv")
    )

    # 5. Get k-hop neighbourhood of gene of interest
    if goi_symbol not in ppi_graph.nodes:
        logging.warning(f"{goi_symbol} was not present in the PPI network.")
        return

    ego_graph = nx.ego_graph(ppi_graph, goi_symbol, radius=k)
    logging.info(
        f"Size of {goi_symbol} {k}-hop neighbourhood: {ego_graph.number_of_nodes()}"
    )

    # 5.1. Save GOI ego network edges attributes
    nx.to_pandas_edgelist(ego_graph).to_csv(
        save_path.joinpath(f"{goi_symbol}_{k}_ego_network_edges.csv")
    )

    # 5.2. Save GOI ego network to disk
    nx.write_weighted_edgelist(
        ego_graph,
        save_path.joinpath(f"{goi_symbol}_{k}_ego_network.edgelist").open("wb"),
        delimiter=";",
    )

    # 5.3. Get node-level metrics of nodes in the ego graph
    node_metrics_ego = node_metrics.loc[list(ego_graph.nodes)]

    # 5.4. Save metrics to disk
    node_metrics_ego.sort_values(by="page_rank", ascending=False).to_csv(
        save_path.joinpath(f"{goi_symbol}_{k}_ego_network_metrics.csv")
    )
