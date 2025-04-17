"""
Script to generate protein-protein interaction networks from differential gene set results.

This script constructs and analyzes protein-protein interaction (PPI) networks using
genes from differentially enriched gene sets (DEGSs) stratified by FOLH1 (PSMA) expression
levels in the PCTA-WCDT prostate cancer dataset. It integrates both physical and functional
interaction data from the STRING database.

The script performs the following steps:
1. Loads gene statistics from previously identified DEGSs
2. Filters genes based on statistical significance (p-value threshold)
3. Queries the STRING database to retrieve interaction data
4. Constructs PPI networks with varying interaction score thresholds
5. Saves the networks and their visualizations to disk

The networks are built for both primary and metastatic samples, comparing high vs low
FOLH1 expression within each sample type, and can be constructed with different
interaction types (physical or functional) and score thresholds.

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
from multiprocessing import Manager, freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.degss_ppi_networks.utils import process_degss_ppi_network
from r_wrappers.utils import map_gene_id
from utils import run_func_dict

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logger = logging.getLogger("utils")
logger.setLevel(logging.WARNING)
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
DEGSS_GENES_STATS_PATH: Path = DATA_ROOT.joinpath("degss_genes_stats")
DESEQ_PATH: Path = DATA_ROOT.joinpath("deseq2")
DEGSS_PPI_PATH: Path = DATA_ROOT.joinpath("degss_ppi_networks")
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
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (0.0,)
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
INTERACTION_SCORES: Iterable[int] = (300, 500, 700, 900)
NETWORK_TYPES: Iterable[str] = ("physical", "functional")
K_HOPS: int = 2
DEGSS_GENE_P_TH: float = 0.05
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

        # 1.3. Set experiment prefix
        contrasts_levels = (contrast_level_test, contrast_level_control)
        exp_prefix = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )
        contrast_name_prefix = f"{contrast_level_test}_vs_{contrast_level_control}"

        for (
            p_col,
            p_th,
            lfc_level,
            lfc_th,
        ) in product(P_COLS, P_THS, LFC_LEVELS, LFC_THS):
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")
            results_key = (
                f"{contrast_name_prefix}_degss_"
                f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                f"gsea_{'_'.join(MSIGDB_CATS)}_union"
            )
            save_path = DEGSS_PPI_PATH.joinpath(results_key)
            save_path.mkdir(exist_ok=True, parents=True)

            # Get DEGSs genes statistics
            degss_gene_stats = pd.read_csv(
                DEGSS_GENES_STATS_PATH.joinpath(results_key).joinpath(
                    "gene_occurrence_stats_bootstrap_z_score_summary_ann.csv"
                ),
                index_col=0,
            )

            # Extract genes involved in all DEGSs, filtered by p-value
            degss_gene_stats_filt = degss_gene_stats[
                (degss_gene_stats["p_value"] < DEGSS_GENE_P_TH)
                | (degss_gene_stats.index == "FOLH1")
            ]
            degss_gene_stats_filt.to_csv(save_path.joinpath("ppi_genes.csv"))
            genes_symbol = degss_gene_stats_filt.index.tolist()
            if len(genes_symbol) <= 5:
                logger.warning("Only five genes or less selected, skipping.")
                continue

            for interaction_score, network_type in product(
                INTERACTION_SCORES, NETWORK_TYPES
            ):
                save_path_network = save_path.joinpath(
                    f"{network_type}_network_at_{interaction_score}_score"
                )
                save_path_network.mkdir(exist_ok=True, parents=True)

                # 2. Generate input collection for all arguments' combinations
                input_collection.append(
                    dict(
                        genes_symbol=genes_symbol,
                        metadata_df=degss_gene_stats_filt,
                        save_path=save_path_network,
                        goi_symbol=GOI_SYMBOL,
                        k=K_HOPS,
                        interaction_score=interaction_score,
                        network_type=network_type,
                    )
                )

# 3. Run
if __name__ == "__main__":
    freeze_support()

    if PARALLEL and len(input_collection) > 1:
        with Manager() as manager:
            lock = manager.Lock()

            input_collection = [dict(**ins, api_lock=lock) for ins in input_collection]

            parallelize_map(
                functools.partial(run_func_dict, func=process_degss_ppi_network),
                input_collection,
                processes=user_args["processes"],
                method="fork",
            )
    else:
        for ins in tqdm(input_collection):
            process_degss_ppi_network(**ins)
