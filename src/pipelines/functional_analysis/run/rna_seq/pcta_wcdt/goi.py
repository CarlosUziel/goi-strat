"""
Functional enrichment analysis pipeline for RNA-seq data from PCTA-WCDT dataset.

This script performs Gene Set Enrichment Analysis (GSEA) and Over-Representation Analysis (ORA)
on differential gene expression results from PCTA-WCDT RNA-seq data. It is specifically focused
on analyzing genes associated with the gene of interest (GOI), FOLH1/PSMA,
across different sample types and expression levels.

The script:
1. Reads differential expression results from DESeq2 output
2. Performs GSEA using gene ranking by log2FoldChange
3. Performs ORA on filtered gene lists with various p-value and fold change thresholds
4. Runs analyses across multiple contrasts (e.g., high vs low GOI expression)
5. Saves results and generates visualization plots

The functional enrichment is performed across multiple gene set databases,
including GO, KEGG, MSigDB, and more, through the run_all_gsea and run_all_ora functions.

Usage:
    python goi.py [--root-dir DIRECTORY] [--processes NUM_PROCESSES]
"""

import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Union

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.functional_analysis.utils import functional_enrichment
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
    help="Root directory for data storage",
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
FUNC_PATH: Path = DATA_ROOT.joinpath("functional")
FUNC_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = FUNC_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")

# Configuration parameters
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
GOI_CLASS_PREFIX: str = f"{GOI_SYMBOL}_class"
SAMPLE_CLUSTER_CONTRAST_LEVELS: Iterable[
    Iterable[Dict[str, Iterable[Union[int, str]]]]
] = tuple(
    [
        (
            {SAMPLE_CONTRAST_FACTOR: (test[0],), GOI_LEVEL_PREFIX: (test[1],)},
            {SAMPLE_CONTRAST_FACTOR: (control[0],), GOI_LEVEL_PREFIX: (control[1],)},
        )
        for test, control in [
            (("prim", "high"), ("prim", "low")),
            (("met", "high"), ("met", "low")),
        ]
    ]
)

P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
PARALLEL: bool = True

# Main execution code
annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for sample_cluster_contrast in SAMPLE_CLUSTER_CONTRAST_LEVELS:
    # 1. Setup
    test_filters, control_filters = sample_cluster_contrast

    # 1.1. Annotation of test samples
    contrast_level_test = "_".join(chain(*test_filters.values()))

    # 1.2. Annotation of control samples
    contrast_level_control = "_".join(chain(*control_filters.values()))

    # 1.3. Set contrast levels and experiment prefix
    contrasts_levels = (contrast_level_test, contrast_level_control)
    exp_prefix = (
        f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
        f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
    )

    # 1.4. Set experiment name
    exp_name = f"{exp_prefix}_{contrast_level_test}_vs_{contrast_level_control}"

    # 1.5. Load input DEGS
    results_file = DATA_ROOT.joinpath("deseq2").joinpath(
        f"{exp_name}_deseq_results_unique.csv"
    )

    if not results_file.exists():
        continue

    # 2. Add GSEA inputs
    input_collection.append(
        dict(
            data_type="diff_expr",
            func_path=FUNC_PATH,
            plots_path=PLOTS_PATH,
            results_file=results_file,
            exp_name=exp_name,
            org_db=org_db,
            numeric_col="log2FoldChange",
            analysis_type="gsea",
        )
    )

    # 3. Add ORA inputs
    for p_col, p_th, lfc_level, lfc_th in product(P_COLS, P_THS, LFC_LEVELS, LFC_THS):
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_th).replace(".", "_")
        input_collection.append(
            dict(
                data_type="diff_expr",
                func_path=FUNC_PATH,
                plots_path=PLOTS_PATH,
                results_file=results_file,
                exp_name=f"{exp_name}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}",
                org_db=org_db,
                cspa_surfaceome_file=STORAGE.joinpath(
                    "CSPA_validated_surfaceome_proteins_human.csv"
                ),
                p_col=p_col,
                p_th=p_th,
                lfc_col="log2FoldChange",
                lfc_level=lfc_level,
                lfc_th=lfc_th,
                numeric_col="log2FoldChange",
                analysis_type="ora",
            )
        )


# 4. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=functional_enrichment),
            input_collection,
            processes=user_args["processes"] // 3,
            method="fork",
        )
    else:
        for ins in tqdm(input_collection):
            functional_enrichment(**ins)
