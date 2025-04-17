"""
Script to perform differential DNA methylation analysis on WCDT dataset using methylKit.

This script analyzes reduced representation bisulfite sequencing (RRBS) data from West
Coast Dream Team (WCDT) metastatic castration-resistant prostate cancer (mCRPC) samples
to identify differentially methylated positions (DMPs) between sample groups stratified by
FOLH1/PSMA expression levels. It uses the methylKit R package through Python wrappers.

The script performs the following steps:
1. Loads methylation data from Bismark output files and sample annotations
2. Performs quality control and filtering on the methylation data
3. Identifies differentially methylated positions (DMPs) between high vs low FOLH1 expression
4. Annotates DMPs with genomic context (promoters, gene bodies, etc.)
5. Generates visualizations including methylation profiles and heatmaps
6. Saves the results to disk for downstream analysis

This analysis complements the BiSeq-based analysis by focusing on individual methylation
positions rather than regions, providing a more detailed view of epigenetic alterations
associated with FOLH1/PSMA expression in metastatic prostate cancer.

Usage:
    python goi_methylkit.py [--root-dir ROOT_DIR] [--processes NUM_PROCESSES]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --processes: Number of processes for parallel processing (default: CPU count - 2)
"""

import argparse
import json
import logging
import multiprocessing
import warnings
from copy import deepcopy
from itertools import chain, product
from pathlib import Path
from typing import Dict, Iterable, Union

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import filter_df
from pipelines.differential_methylation.methylkit_utils import (
    differential_methylation_rrbs_sites,
)
from r_wrappers.utils import map_gene_id

_ = traceback.install()
rpy2_logger.setLevel(logging.INFO)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.INFO)
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
DATA_ROOT: Path = STORAGE.joinpath(f"WCDT-MCRPC_{GOI_SYMBOL}")
RESULTS_PATH: Path = DATA_ROOT.joinpath("methylkit")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
BISMARK_PATH: Path = STORAGE.joinpath("WCDT-MCRPC").joinpath("data").joinpath("bismark")
GENOME: str = "hg38"
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
        for test, control in [(("met", "high"), ("met", "low"))]
    ]
)
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "met": "#8B3A3A",
    # same colors but for GOI levels
    "low": "#9ACD32",
    "mid": "#4A708B",
    "high": "#8B3A3A",
    # dummy contrast level
    "X": "#808080",
}
GOI_LEVELS_COLORS: Dict[str, str] = {
    f"{sample_type}_{goi_level}": CONTRASTS_LEVELS_COLORS[goi_level]
    for sample_type, goi_level in product(
        CONTRASTS_LEVELS_COLORS.keys(), ("low", "mid", "high")
    )
}
CONTRASTS_LEVELS_COLORS.update(GOI_LEVELS_COLORS)
with DATA_ROOT.joinpath("CONTRASTS_LEVELS_COLORS.json").open("w") as fp:
    json.dump(CONTRASTS_LEVELS_COLORS, fp, indent=True)

N_PROCESSES: int = 16
Q_THS: Iterable[float] = (0.05, 0.01)
MEAN_DIFF_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
MEAN_DIFF_THS: Iterable[float] = (5, 10, 20)

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
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
            annot_df=annot_df_contrasts,
            bismark_path=BISMARK_PATH,
            contrast_factor=GOI_CLASS_PREFIX,
            results_path=RESULTS_PATH,
            plots_path=PLOTS_PATH,
            contrast_levels=contrasts_levels,
            genome=GENOME,
            n_processes=N_PROCESSES,
            q_ths=Q_THS,
            mean_diff_levels=MEAN_DIFF_LEVELS,
            mean_diff_ths=MEAN_DIFF_THS,
        )
    )

# 3. Run differential methylation analysis
if __name__ == "__main__":
    for ins in tqdm(input_collection):
        differential_methylation_rrbs_sites(**ins)
