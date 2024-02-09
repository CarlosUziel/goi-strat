import functools
import json
import logging
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from node2vec import Node2Vec
from sklearn.cluster import DBSCAN, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tqdm import tqdm

from components.np_encoder import NpEncoder
from data.utils import parallelize_map
from pipelines.degss_ppi_networks_clustering.components import ClusterSimilarityMatrix
from utils import run_func_dict

double_brackets = ro.r("function(obj, idx){return(obj[[idx]])}")
logging.basicConfig()
logger = logging.getLogger(__name__)


def find_optimal_clusters(
    data: pd.DataFrame,
    sample_frac: float = 0.8,
    n_models: int = 128,
    n_jobs: int = 8,
    random_seed: int = 8080,
) -> Tuple[Dict[str, Any], np.array]:
    """Find best clustering for a given matrix of sample features through the use of
    an ensemble of clustering algorithm. Multiple weak learners are trained on different
    `n_clusters` values after an initial estimation of the number of clusters.

    This technique is called "ensemble clustering".

    Args:
        data: Data to cluster.
        sample_frac: Percentage of samples to sample to fit weak clustering algorithms.
        n_models: Number of weak learners to train per `n_clusters` value.
        n_jobs: Number of threads to use for clustering.

    Returns:
        The clustering statistics
        The clusters labels
    """
    # 0. Estimate initial number of clusters
    n_clusters = max(
        2, len(set(DBSCAN(min_samples=3, n_jobs=n_jobs).fit_predict(data)))
    )

    # 1. Define weak learners as base estimators
    weak_learners = [
        MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=max(32, len(data) // n_jobs),
            n_init=1,
        )
        for _ in range(n_models)
    ]

    # 2. Fit cluster similarity matrix for each weak learner
    clt_sim_matrix = ClusterSimilarityMatrix()
    for model in weak_learners:
        clt_sim_matrix.fit(model.fit(X=data.sample(frac=sample_frac)).predict(X=data))

    # 2.1. Get final normalized similarity matrix
    sim_matrix = clt_sim_matrix.similarity
    sim_matrix_norm = sim_matrix / sim_matrix.diagonal()

    # 2. Get cluster labels for k using aggregator clustering method
    cluster_labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_seed,
        n_jobs=n_jobs,
    ).fit_predict(X=sim_matrix_norm)

    # 3. Compute clustering statistics
    stats_df = pd.Series(
        cluster_labels,
        index=data.index,
    ).to_frame("cluster_label")

    # 3.1. Add per-sample silhouette score
    try:
        stats_df["silhouette_score"] = silhouette_samples(data, cluster_labels)
    except ValueError:
        stats_df["silhouette_score"] = 0

    # 3.2. Add average silhouette score per cluster
    try:
        silhouette_avg = silhouette_score(data, cluster_labels)
    except ValueError:
        silhouette_avg = 0

    stats_df["above_silhouette_avg"] = stats_df["silhouette_score"] > silhouette_avg

    # 3.3. Get min percentage of samples per cluster above average silhouette score
    min_above_avg = (
        stats_df.groupby("cluster_label")["above_silhouette_avg"]
        .apply(lambda x: np.sum(x) / len(x))
        .sort_values()
        .iloc[0]
    )

    # 3.4. Get normalized standard deviation of cluster sizes
    cluster_sizes = list(Counter(cluster_labels).values())
    cluster_sizes_std = np.std(cluster_sizes) / np.mean(cluster_sizes)

    # 4. Return statistics
    return (
        {
            "n_clusters": n_clusters,
            "silhouette_avg": silhouette_avg,
            "cluster_sizes_std": cluster_sizes_std,
            "min_above_avg": min_above_avg,
        },
        LabelEncoder().fit_transform(cluster_labels),
    )


def train_node2vec_clustering(
    ppi_graph: nx.Graph,
    hparams: Dict[str, Any],
    node2vec_threads: int = 8,
    sample_frac: float = 0.8,
    random_seed: int = 8080,
) -> Tuple[Dict[str, Union[int, str]], pd.DataFrame, Iterable[int], float]:
    """
    Train a Node2Vec model with the given hyper-parameters. Evaluate the
        hyper-parameters by clustering the samples.

    Args:
        ppi_graph: PPI network, where nodes are proteins and edges represent
            protein-protein interactions.
        hparams: Node2Vec hyper-parameter dictionary.
        node2vec_threads: Number of threads to use to train Node2Vec. This is also used
            as reference for clustering threads.
        sample_frac: Percentage of samples to sample to fit weak clustering algorithms.
        random_seed: Random state seed.

    Returns:
        A dictionary of the best hyper-parameters found.
        A dataframe with node embeddings.
        A list of the best labels found.
        The best ranking statistic achieved.
    """
    # 1. Fit Node2Vec model
    logger.info("Fitting Node2Vec model...")

    node2vec = Node2Vec(
        ppi_graph,
        workers=node2vec_threads,
        weight_key="weight",
        quiet=True,
        seed=random_seed,
        **hparams,
    )
    model = node2vec.fit(min_count=1)

    # 2. Get Node2Vec embeddings
    node_embeddings = pd.DataFrame(
        [model.wv.get_vector(str(n)) for n in ppi_graph.nodes()],
        index=ppi_graph.nodes,
    )
    node_embeddings.columns = [f"N2V-{i}" for i in range(len(node_embeddings.columns))]

    # 2.1. Standarize embeddings
    node_embeddings.iloc[:, :] = RobustScaler().fit_transform(node_embeddings.values)

    logger.info("Node2Vec embeddings successfully computed!")

    # 3. Cluster proteins
    logger.info("Clustering proteins...")

    clustering_stats, cluster_labels = find_optimal_clusters(
        node_embeddings,
        sample_frac=sample_frac,
        n_jobs=node2vec_threads,
    )

    logger.info("Clustering of DEGSs finished!")

    return {
        **hparams,
        "node_embeddings": node_embeddings,
        **clustering_stats,
        "cluster_labels": pd.Series(data=cluster_labels, index=node_embeddings.index),
    }


def tune_ppi_clustering(
    ppi_graph: nx.Graph,
    save_path: Optional[Path] = None,
    p: float = 1,
    q: float = 0.01,
    gs_threads: int = 8,
    node2vec_threads: int = 8,
    sample_frac: float = 0.8,
    random_seed: int = 8080,
) -> Dict[str, Any]:
    """
    Find the best clustering of proteins using Node2Vec embeddings as features. For each
        Node2Vec combination of hyper-parameters, consensus clustering is calculated and
        the labels returned. The best combination of hyper-parameters is chosen based on
        the silhouette score.

    Args:
        ppi_graph: PPI network, where nodes are proteins and edges represent
            protein-protein interactions.
        save_path: Path to save results to.
        p: Return hyper-parameter.
        q: In-out hyper-parameter.
        gs_threads: Number of threads to use for grid-search hyper-parameter search.
        node2vec_threads: Number of threads to use to train Node2Vec.
        sample_frac: Percentage of samples to sample to fit weak clustering algorithms.
        random_seed: Random state seed.

    Returns:
        A dictionary of the best hyper-parameters found
        A dataframe with node embeddings
        A list of the best labels found
        The best ranking statistic achieved
    """
    # 1. Define hyper-parameter grid to tune Node2Vec
    hparams_grid = {
        "p": [p],
        "q": [q],
        "dimensions": [4, 8],
        "walk_length": [128, 256],
        "num_walks": [128, 256],
    }

    # 2. Perform grid-search hyper-parameter tuning
    keys, values = zip(*hparams_grid.items())
    hparams_permutations = [dict(zip(keys, v)) for v in product(*values)]
    inputs = [
        dict(
            ppi_graph=ppi_graph,
            hparams=hparams,
            node2vec_threads=node2vec_threads,
            sample_frac=sample_frac,
            random_seed=random_seed,
        )
        for hparams in hparams_permutations
    ]
    if gs_threads > 1:
        gs_results = parallelize_map(
            functools.partial(run_func_dict, func=train_node2vec_clustering),
            inputs=inputs,
            threads=gs_threads,
        )
    else:
        gs_results = [train_node2vec_clustering(**ins) for ins in tqdm(inputs)]

    # 2.1. Check results
    gs_results = [x for x in gs_results if x is not None]
    if len(gs_results) == 0:
        logging.error("None of the grid-search iterations finished correctly.")
        return None

    # 3. Save grid search results
    if save_path is not None:
        with save_path.joinpath("grid_search_results.json").open("w") as fp:
            json.dump(
                gs_results,
                fp,
                indent=4,
                cls=NpEncoder,
            )

    gs_df = pd.DataFrame(gs_results)

    if save_path is not None:
        gs_df.drop(columns=["node_embeddings", "cluster_labels"]).sort_values(
            "silhouette_avg", ascending=False
        ).to_csv(save_path.joinpath("grid_search_results.csv"))

    # 4. Get best result
    gs_df = gs_df[gs_df["min_above_avg"] > 0]

    if gs_df.empty:
        return {}
    else:
        return (
            gs_df.sort_values(["silhouette_avg", "min_above_avg"], ascending=False)
            .iloc[0]
            .to_dict()
        )


def cluster_ppi_proteins(
    network_edges_file: Path,
    nodes_metadata_df: pd.DataFrame,
    save_path: Path,
    p: float = 1,
    q: float = 0.01,
    gs_threads: int = 8,
    node2vec_threads: int = 8,
    sample_frac: float = 0.8,
    random_seed: int = 8080,
) -> None:
    """Given a PPI network, use the Node2Vec graph embedding algorithm to generate
        node features that can be used to cluster the network proteins.

    Args:
        network_edges_file: File containing a dataframe with the list of edges of the
            PPI network and their attributes.
        nodes_metadata_df: Network genes nodes metadata, index is SYMBOL ID.
        save_path: Path to save results to.
        p: Return hyper-parameter. You can encourage more structural equivalence
            similarity by setting a higher value of q. Structural equivalence refers to
            the extent to which two nodes are connected to the same nodes, i.e., they
            share the same neighborhood while not requiring the two nodes to be directly
            connected.
        q: In-out hyper-parameter. Homophily can be achieved by setting a smaller value
            of q. homophily is defined as nodes that belong to the same network
            community (i.e., are closer to one another in the network). A smaller value
            of q approximates depth-first search.
        gs_threads: Number of threads to use for grid search.
        node2vec_threads: Number of threads to use to train Node2Vec model.
        sample_frac: Percentage of samples to sample to fit weak clustering algorithms.
        random_seed: Random state seed.
    """
    # 0. Setup
    ppi_graph = nx.from_pandas_edgelist(pd.read_csv(network_edges_file, index_col=0))

    # 1. Only keep the giant component
    ppi_graph = ppi_graph.subgraph(
        sorted(nx.connected_components(ppi_graph), key=len, reverse=True)[0]
    )

    if (n_nodes := ppi_graph.number_of_nodes()) < 50:
        logger.warning(f"PPI network is too small ({n_nodes}) for clustering.")
        return None

    # 2. Get optimal clustering of gene sets
    top_gs_result = tune_ppi_clustering(
        ppi_graph=ppi_graph,
        save_path=save_path,
        p=p,
        q=q,
        gs_threads=gs_threads,
        node2vec_threads=node2vec_threads,
        sample_frac=sample_frac,
        random_seed=random_seed,
    )

    # 3. Save results for the best hyper-parameter combination
    if len(top_gs_result) != 0:
        with save_path.joinpath("top_gs_result.json").open("w") as fp:
            json.dump(top_gs_result, fp, indent=4, cls=NpEncoder)

        # 3.1. Save clusters size
        with save_path.joinpath("clusters_size.json").open("w") as fp:
            json.dump(
                Counter(sorted(top_gs_result["cluster_labels"].tolist())),
                fp,
                indent=4,
                cls=NpEncoder,
            )

        # 3.2. Save proteins embeddings
        top_gs_result["node_embeddings"].to_csv(
            save_path.joinpath("node_embeddings.csv")
        )

        # 3.3. Save cluster IDs with nodes metadata
        top_gs_result["cluster_labels"].rename("cluster_id").to_frame().join(
            nodes_metadata_df
        ).to_csv(save_path.joinpath("genes_cluster_ids_metadata.csv"))
