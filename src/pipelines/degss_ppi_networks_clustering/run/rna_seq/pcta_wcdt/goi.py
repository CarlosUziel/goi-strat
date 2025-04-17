"""
Script to cluster proteins in PPI networks using Node2Vec embeddings and ensemble clustering.

This script analyzes protein-protein interaction (PPI) networks constructed from
differentially enriched gene sets (DEGSs) in the PCTA-WCDT prostate cancer dataset.
It generates protein clusters that represent functional modules within the network.

The script performs the following steps:
1. Loads PPI network data previously generated for high vs low FOLH1 expression
2. Applies Node2Vec algorithm to generate vector embeddings for each protein in the network
3. Uses ensemble clustering to identify protein communities based on the embeddings
4. Generates visualizations of the clustered networks
5. Saves the clustering results and embeddings to disk for downstream analysis

The clustering is performed with different parameter settings (p,q) that control
the exploration-exploitation tradeoff in the Node2Vec random walks. Lower p values
prioritize local exploration (BFS-like behavior) while lower q values prioritize
outward exploration (DFS-like behavior).

Usage:
    python goi.py [--root-dir ROOT_DIR] [--processes NUM_PROCESSES]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --processes: Number of processes for parallel processing (default: CPU count - 2)
"""

import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.degss_ppi_networks_clustering.utils import cluster_ppi_proteins
from r_wrappers.utils import map_gene_id
from utils import run_func_dict

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    type=str,
    help="Root directory",
    nargs="?",
    default="/mnt/d/phd_data",
)
parser.add_argument(
    "--processes",
    type=int,
    help="Number of processes for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
GOI_ENSEMBL: str = "ENSG00000086205"  # ENSEMBL ID for FOLH1 (PSMA)
SPECIES: str = "Homo sapiens"
org_db = OrgDB(SPECIES)
GOI_SYMBOL = map_gene_id([GOI_ENSEMBL], org_db, "ENSEMBL", "SYMBOL")[0]
DATA_ROOT: Path = STORAGE.joinpath(f"PCTA-WCDT_{GOI_SYMBOL}")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
PPI_NETWORK_PATH: Path = DATA_ROOT.joinpath("degss_ppi_networks")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
CONTRAST_COMPARISONS: Dict[
    str,
    Iterable[
        Tuple[Dict[Iterable[str], Iterable[str]], Dict[Iterable[str], Iterable[str]]]
    ],
] = {
    "met_prim": (
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"], GOI_LEVEL_PREFIX: ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"], GOI_LEVEL_PREFIX: ["low"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["met"], GOI_LEVEL_PREFIX: ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["met"], GOI_LEVEL_PREFIX: ["low"]},
        ),
    ),
}
PQ_PAIRS: Iterable[Tuple[float, float]] = (
    (1.0, 0.5),
    (1.0, 0.1),
    (0.5, 1.0),
    (0.1, 1.0),
)
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("up", "down")
LFC_THS: Iterable[float] = (0.0,)
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
INTERACTION_SCORES: Iterable[int] = (500, 700)
NETWORK_TYPES: Iterable[str] = ("functional",)
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for contrast_comparison, contrast_comparison_filters in CONTRAST_COMPARISONS.items():
    contrast_prefixes = {}
    # 1. Get contrast file prefixes
    for contrast_filters in contrast_comparison_filters:
        # 1.1. Setup
        test_filters, control_filters = contrast_filters

        # 1.2. Multi-level samples annotation
        # 1.2.1. Annotation of test samples
        contrast_level_test = "_".join(chain(*test_filters.values()))

        # 1.2.2. Annotation of control samples
        contrast_level_control = "_".join(chain(*control_filters.values()))

        # 1.3. Set experiment prefix and remove unnecesary samples
        contrasts_levels = (contrast_level_test, contrast_level_control)

        contrast_name_prefix = f"{contrast_level_test}_vs_{contrast_level_control}"

        # 2. Generate input collection for all arguments' combinations
        for (
            p_col,
            p_th,
            lfc_level,
            lfc_th,
            (p, q),
            interaction_score,
            network_type,
        ) in product(
            P_COLS,
            P_THS,
            LFC_LEVELS,
            LFC_THS,
            PQ_PAIRS,
            INTERACTION_SCORES,
            NETWORK_TYPES,
        ):
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")
            p_str = str(p).replace(".", "_")
            q_str = str(q).replace(".", "_")

            root_path = PPI_NETWORK_PATH.joinpath(
                f"{contrast_name_prefix}_degss_"
                f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                f"gsea_{'_'.join(MSIGDB_CATS)}_union"
            ).joinpath(f"{network_type}_network_at_{interaction_score}_score")

            network_edges_file = root_path.joinpath("ppi_network_edges.csv")
            network_node_metrics = pd.read_csv(
                root_path.joinpath("network_metrics.csv"), index_col=0
            )

            save_path = root_path.joinpath("clustering").joinpath(
                f"p_{p_str}_q_{q_str}"
            )
            save_path.mkdir(exist_ok=True, parents=True)

            input_collection.append(
                dict(
                    network_edges_file=network_edges_file,
                    nodes_metadata_df=network_node_metrics,
                    save_path=save_path,
                    p=p,
                    q=q,
                    gs_processes=1,
                    node2vec_processes=1,
                )
            )

# 3. Run
if __name__ == "__main__":
    freeze_support()

    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=cluster_ppi_proteins),
            input_collection,
            processes=user_args["processes"],
        )
    else:
        for ins in tqdm(input_collection):
            cluster_ppi_proteins(**ins)
