"""
Script to stratify PCTA-WCDT samples based on FOLH1 expression and perform GSVA analysis.

This script processes RNA-seq data from the PCTA-WCDT (Prostate Cancer Transcriptome
Atlas and West Coast Dream Team) dataset to stratify samples into high, medium, and low
expression groups based on FOLH1/PSMA (GOI) expression levels. It then performs
differential enrichment analysis between these groups using Gene Set Variation
Analysis (GSVA) scores.

The script performs the following steps:
1. Loads RNA-seq count data and sample annotations
2. Applies variance-stabilizing transformation (VST) to normalize expression values
3. Ranks and divides samples into high, medium, and low GOI expression groups
4. Tests multiple possible groupings by varying the size of each group
5. Performs differential enrichment analysis between high and low groups
6. Finds the optimal split that maximizes differences in pathway activity
7. Saves the results and updated sample annotations for downstream analysis

This stratification approach enables the identification of biological pathways and
processes associated with varying levels of FOLH1/PSMA expression in primary and
metastatic prostate cancer samples.

Usage:
    python goi_gsva_splits_annotation.py [--root-dir ROOT_DIR] [--threads NUM_THREADS]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --threads: Number of threads for parallel processing (default: CPU count - 2)
"""

import argparse
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

import pandas as pd
import rpy2.robjects as ro
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.data.utils import get_optimal_gsva_splits
from pipelines.differential_enrichment.utils import diff_enrich_gsva_limma
from r_wrappers.deseq2 import vst_transform
from r_wrappers.utils import map_gene_id, pd_df_to_rpy2_df, rpy2_df_to_pd_df
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
MAIN_ROOT: Path = STORAGE.joinpath("PCTA-WCDT")
ANNOT_PATH: Path = MAIN_ROOT.joinpath("data").joinpath("samples_annotation.csv")
RAW_COUNTS_PATH: Path = MAIN_ROOT.joinpath("data").joinpath(
    "raw_counts_wo_batch_effects.csv"
)
GSVA_PATH: Path = MAIN_ROOT.joinpath("data").joinpath("gsva")
DATA_ROOT: Path = STORAGE.joinpath(f"PCTA-WCDT_{GOI_SYMBOL}")
RESULTS_PATH: Path = DATA_ROOT.joinpath("group_splits_gsva")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
DATA_PATH.mkdir(exist_ok=True, parents=True)
ANNOT_PATH_NEW: Path = DATA_PATH.joinpath(f"{ANNOT_PATH.stem}_{GOI_SYMBOL}.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[str] = sorted(("prim", "met"))
GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all",)
LFC_THS: Iterable[float] = (0.0,)
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
MIN_PERCENTILE: float = 0.1
MID_PERCENTILE: float = 0.5
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
annot_df_contrasts = annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(CONTRASTS_LEVELS)]
counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)

common_samples = annot_df_contrasts.index.intersection(counts_df.columns)
annot_df_contrasts = annot_df_contrasts.loc[common_samples, :]
counts_df = counts_df.loc[:, common_samples]


# IMPORTANT: ranking of samples must be done on VST values. Using raw counts will yield
# a different ranking.
annot_df_contrasts[f"{GOI_SYMBOL}_CNT"] = counts_df.loc[
    GOI_ENSEMBL, annot_df_contrasts.index
]
annot_df_contrasts[f"{GOI_SYMBOL}_VST"] = rpy2_df_to_pd_df(
    vst_transform(
        ro.r("as.matrix")(pd_df_to_rpy2_df(counts_df.loc[counts_df.mean(axis=1) > 1]))
    )
).loc[GOI_ENSEMBL, annot_df_contrasts.index]

sample_types_str = "+".join(sorted(set(annot_df_contrasts[SAMPLE_CONTRAST_FACTOR])))
gsva_matrices = {
    msigdb_cat: GSVA_PATH.joinpath(
        f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}__{msigdb_cat}.csv"
    )
    for msigdb_cat in MSIGDB_CATS
}
msigdb_cats_meta = {
    msigdb_cat: GSVA_PATH.joinpath(f"{msigdb_cat}_meta.csv")
    for msigdb_cat in MSIGDB_CATS
}

# 1. Collect all function inputs
input_collection = []
group_counts = []
for contrast_level in CONTRASTS_LEVELS:
    # 1.1. Get ranked list of samples
    annot_df_sorted = annot_df_contrasts[
        annot_df_contrasts[SAMPLE_CONTRAST_FACTOR] == contrast_level
    ].sort_values(f"{GOI_SYMBOL}_VST", ascending=True)

    # 1.2. Define initial group distribution
    n = len(annot_df_sorted)
    low_min, high_min = [int(n * MIN_PERCENTILE)] * 2

    mid_n = int(n * MID_PERCENTILE)
    low_n = low_min
    high_n = n - (low_n + mid_n)

    while high_n >= high_min:
        group_counts.append((contrast_level, low_n, mid_n, high_n))

        # 1.3. Build annotation file with new groups
        annot_df_sorted_groups = deepcopy(annot_df_sorted)
        annot_df_sorted_groups[GOI_LEVEL_PREFIX] = (
            ["low"] * low_n + ["mid"] * mid_n + ["high"] * high_n
        )

        exp_prefix = f"{contrast_level}_{GOI_LEVEL_PREFIX}_high_{high_n}+low_{low_n}_"

        # 1.4. Inputs for each MSigDB category
        for msigdb_cat in MSIGDB_CATS:
            results_path = RESULTS_PATH.joinpath(msigdb_cat)
            results_path.mkdir(exist_ok=True, parents=True)

            # 1.5. Generate input collection for all arguments' combinations
            input_collection.append(
                dict(
                    gsva_matrix_path=gsva_matrices[msigdb_cat],
                    msigdb_cat_meta_path=msigdb_cats_meta[msigdb_cat],
                    annot_df_contrasts=annot_df_sorted_groups,
                    contrast_factor=GOI_LEVEL_PREFIX,
                    results_path=results_path,
                    exp_prefix=exp_prefix,
                    contrasts_levels=[("high", "low")],
                    p_cols=P_COLS,
                    p_ths=P_THS,
                    lfc_levels=LFC_LEVELS,
                    lfc_ths=LFC_THS,
                    design_factors=[GOI_LEVEL_PREFIX],
                )
            )

        # 1.6. Increase counters
        low_n += 1
        high_n -= 1


# 2. Run differential enrichment
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=diff_enrich_gsva_limma),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            diff_enrich_gsva_limma(**ins)

    # 3. Summarize results
    get_optimal_gsva_splits(
        results_path=RESULTS_PATH,
        msigdb_cats_meta_paths=msigdb_cats_meta,
        group_counts=group_counts,
        goi_level_prefix=GOI_LEVEL_PREFIX,
        msigdb_cats=MSIGDB_CATS,
        contrasts_levels=CONTRASTS_LEVELS,
        annot_df=annot_df_contrasts,
        sample_contrast_factor=SAMPLE_CONTRAST_FACTOR,
        goi_symbol=GOI_SYMBOL,
        annot_path_new=ANNOT_PATH_NEW,
        p_col="padj",
        p_th=0.05,
        lfc_level="all",
        lfc_th=0.0,
    )
