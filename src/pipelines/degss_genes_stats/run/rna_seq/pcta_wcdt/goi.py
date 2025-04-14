"""
Script to calculate statistical significance of genes in differentially enriched gene sets.

This script analyses differentially enriched gene sets (DEGSs) from the PCTA-WCDT
prostate cancer dataset to identify and rank individual genes that are frequently found
across significant gene sets. It computes statistics to measure the importance of each
gene within the enriched pathways.

The script performs the following steps:
1. Loads DEGSs identified from previous differential enrichment analyses
2. For each contrast (high vs low FOLH1 in primary and metastatic samples):
   a. Computes gene occurrence frequencies across all gene sets
   b. Performs bootstrap sampling to determine statistical significance
   c. Calculates z-scores and p-values to rank genes by importance
3. Saves the results as gene statistics tables for downstream analysis

This analysis helps identify key genes driving pathway enrichment differences between
high and low FOLH1 expression groups, providing potential targets for further investigation.

Usage:
    python goi.py [--root-dir ROOT_DIR] [--threads NUM_THREADS]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --threads: Number of threads for parallel processing (default: CPU count - 2)
"""

import argparse
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
from pipelines.degss_genes_stats.utils import compute_degss_genes_stats
from r_wrappers.utils import map_gene_id

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
    "--threads",
    type=int,
    help="Number of threads for parallel processing",
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
PRIM_METH_ROOT: Path = STORAGE.joinpath(f"TCGA-PRAD_MethArray_{GOI_SYMBOL}")
MET_METH_ROOT: Path = STORAGE.joinpath(f"WCDT-MCRPC_{GOI_SYMBOL}")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
GSVA_PATH: Path = STORAGE.joinpath("PCTA-WCDT").joinpath("data").joinpath("gsva")
INT_ANALYSIS_PATH: Path = DATA_ROOT.joinpath("integrative_analysis")
GENES_STATS_PATH: Path = DATA_ROOT.joinpath("degss_genes_stats")
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
BOOTSTRAP_ITERATIONS: int = 512

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

# Get gene sets metadata for each category
msigdb_dfs = {
    msigdb_cat: pd.read_csv(GSVA_PATH.joinpath(f"{msigdb_cat}_meta.csv"), index_col=0)
    for msigdb_cat in MSIGDB_CATS
}

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

        # 2. Generate input collection for all arguments' combinations
        for (
            p_col,
            p_th,
            lfc_level,
            lfc_th,
        ) in product(P_COLS, P_THS, LFC_LEVELS, LFC_THS):
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")

            root_dir = INT_ANALYSIS_PATH.joinpath("intersecting_degss_gsea")
            results_key = f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"

            save_dir = GENES_STATS_PATH.joinpath(
                f"{contrast_name_prefix}_degss_"
                f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                f"gsea_{'_'.join(MSIGDB_CATS)}_union"
            )
            save_dir.mkdir(exist_ok=True, parents=True)

            test_sample_type, control_sample_type = (
                contrast_level_test.split("_")[0],
                contrast_level_control.split("_")[0],
            )

            # 2.1. Get DEGSs for each category
            degss_cats_dfs = {
                msigdb_cat: pd.read_csv(degss_path, index_col=0)
                for degss_path in root_dir.glob(
                    f"**/{contrast_name_prefix}*{results_key}*union.csv"
                )
                if (msigdb_cat := degss_path.parents[1].name) in MSIGDB_CATS
            }

            input_collection.append(
                dict(
                    degss_cats_dfs=degss_cats_dfs,
                    msigdb_dfs=msigdb_dfs,
                    save_dir=save_dir,
                    bootstrap_iterations=BOOTSTRAP_ITERATIONS,
                    threads=user_args["threads"],
                )
            )

# 3. Run
if __name__ == "__main__":
    freeze_support()
    for ins in tqdm(input_collection):
        compute_degss_genes_stats(**ins)
