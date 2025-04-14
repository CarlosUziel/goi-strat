"""
Script to perform differential DNA methylation analysis on TCGA-PRAD dataset.

This script analyzes DNA methylation array data (Illumina 450K) from TCGA Prostate
Adenocarcinoma (TCGA-PRAD) samples to identify differentially methylated positions (DMPs)
and regions (DMRs) between sample groups stratified by FOLH1/PSMA expression levels.
It uses the minfi R package through Python wrappers.

The script performs the following steps:
1. Loads methylation array data (IDAT files) and sample annotations
2. Performs preprocessing and normalization of methylation data
3. Conducts quality control and filtering steps
4. Performs differential methylation analysis between high vs low FOLH1 expression groups
5. Identifies differentially methylated positions (DMPs) and regions (DMRs)
6. Annotates DMPs/DMRs with genomic context (promoters, gene bodies, etc.)
7. Generates visualizations including volcano plots, heatmaps, and genomic tracks
8. Saves the results to disk for downstream analysis

This analysis provides insights into epigenetic alterations associated with varying
FOLH1/PSMA expression in prostate cancer primary tumors.

Usage:
    python goi.py [--root-dir ROOT_DIR] [--threads NUM_THREADS]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --threads: Number of threads for parallel processing (default: CPU count - 2)
"""

import argparse
import functools
import json
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Union

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import filter_df, parallelize_map
from pipelines.differential_methylation.minfi_utils import (
    differential_methylation_array,
)
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
DATA_ROOT: Path = STORAGE.joinpath(f"TCGA-PRAD_MethArray_{GOI_SYMBOL}")
RESULTS_PATH: Path = DATA_ROOT.joinpath("minfi")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
IDAT_PATH: Path = (
    STORAGE.joinpath("TCGA-PRAD_MethArray").joinpath("data").joinpath("idat")
)
GENOME: str = "hg38"
ARRAY_TYPE: str = "450K"
# Can be downloaded from [here](https://zwdzwd.github.io/InfiniumAnnotation).
GENOME_ANNO_FILE: Path = (
    STORAGE.joinpath("genomes")
    .joinpath("Homo_sapiens")
    .joinpath("InfiniumAnnotation")
    .joinpath(f"HM450.{GENOME}.manifest.tsv")
)
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
        for test, control in [(("prim", "high"), ("prim", "low"))]
    ]
)
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "prim": "#4A708B",
    # same colors but for GOI levels
    "low": "#9ACD32",
    "high": "#8B3A3A",
    # dummy contrast level
    "X": "#808080",
}
GOI_LEVELS_COLORS: Dict[str, str] = {
    f"{sample_type}_{goi_level}": CONTRASTS_LEVELS_COLORS[goi_level]
    for sample_type, goi_level in product(
        CONTRASTS_LEVELS_COLORS.keys(), ("low", "high")
    )
}
CONTRASTS_LEVELS_COLORS.update(GOI_LEVELS_COLORS)
with DATA_ROOT.joinpath("CONTRASTS_LEVELS_COLORS.json").open("w") as fp:
    json.dump(CONTRASTS_LEVELS_COLORS, fp, indent=True)

ID_COL: str = "sample_id"
NORM_TYPES: Iterable[str] = ("noob_quantile",)
N_THREADS: int = 4
P_COLS: Iterable[str] = ("P.Value", "adj.P.Val")
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
LFC_THS: Iterable[float] = (1.0,)
MEAN_METH_DIFF_THS: Iterable[float] = (0.0, 0.1, 0.2)
HEATMAP_TOP_N: int = 1000
DMRS_TOP_N: int = 10
SPECIES: str = "Homo sapiens"
PARALLEL: bool = True  # must be False if len(NORM_TYPES) > 1

genome_anno = pd.read_csv(GENOME_ANNO_FILE, sep="\t").set_index("Probe_ID")
annot_df = (
    pd.read_csv(ANNOT_PATH, index_col=0, dtype=str).rename_axis(ID_COL).reset_index()
)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for sample_cluster_contrast in SAMPLE_CLUSTER_CONTRAST_LEVELS:
    # 0. Setup
    test_filters, control_filters = sample_cluster_contrast

    # 1. Multi-level samples annotation
    annot_df_contrasts = deepcopy(annot_df)

    # Keep only same sample type
    if (sample_type := test_filters[SAMPLE_CONTRAST_FACTOR][0]) == control_filters[
        SAMPLE_CONTRAST_FACTOR
    ][0]:
        annot_df_contrasts = annot_df_contrasts[
            annot_df_contrasts[SAMPLE_CONTRAST_FACTOR] == sample_type
        ]

    # 1.1. Annotation of test samples
    contrast_level_test = "_".join(chain(*test_filters.values()))
    annot_df_contrasts.loc[
        filter_df(annot_df_contrasts, test_filters).index, GOI_CLASS_PREFIX
    ] = contrast_level_test

    # 1.2. Annotation of control samples
    contrast_level_control = "_".join(chain(*control_filters.values()))
    annot_df_contrasts.loc[
        filter_df(annot_df_contrasts, control_filters).index, GOI_CLASS_PREFIX
    ] = contrast_level_control

    # 1.3. Set contrast levels and experiment prefix
    contrasts_levels = (contrast_level_test, contrast_level_control)
    annot_df_contrasts.fillna({GOI_CLASS_PREFIX: "X"}, inplace=True)
    annot_df_contrasts = pd.concat(
        [
            annot_df_contrasts[annot_df_contrasts[GOI_CLASS_PREFIX] == level]
            for level in (*contrasts_levels, "X")
        ]
    )
    exp_prefix = (
        f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
        f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
    )

    # 2. Generate input collection for all arguments' combinations
    input_collection.append(
        dict(
            exp_prefix=exp_prefix,
            genome_anno=genome_anno,
            targets=annot_df_contrasts,
            contrast_factor=GOI_CLASS_PREFIX,
            contrasts_levels=[contrasts_levels],
            idat_path=IDAT_PATH,
            results_path=RESULTS_PATH,
            plots_path=PLOTS_PATH,
            contrasts_levels_colors=CONTRASTS_LEVELS_COLORS,
            id_col=ID_COL,
            genome=GENOME,
            array_type=ARRAY_TYPE,
            norm_types=NORM_TYPES,
            n_threads=N_THREADS,
            p_cols=P_COLS,
            p_ths=P_THS,
            lfc_levels=LFC_LEVELS,
            lfc_ths=LFC_THS,
            mean_meth_diff_ths=MEAN_METH_DIFF_THS,
            heatmap_top_n=HEATMAP_TOP_N,
            dmrs_top_n=DMRS_TOP_N,
        )
    )

# 3. Run differential methylation analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=differential_methylation_array),
            input_collection,
            threads=user_args["threads"] // 4,
        )
    else:
        for ins in tqdm(input_collection):
            differential_methylation_array(**ins)
