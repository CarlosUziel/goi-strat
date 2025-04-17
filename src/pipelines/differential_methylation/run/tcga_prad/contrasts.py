"""
Script to perform standard contrast differential methylation analysis on TCGA-PRAD dataset.

This script analyzes DNA methylation array data (Illumina 450K) from TCGA Prostate
Adenocarcinoma (TCGA-PRAD) samples to identify differentially methylated positions (DMPs)
and regions (DMRs) between primary tumors and normal tissue samples. Unlike the goi.py
script which stratifies samples by gene expression, this script focuses on fundamental
tissue type differences.

The script performs the following steps:
1. Loads methylation array data (IDAT files) and sample annotations
2. Performs preprocessing and normalization of methylation data
3. Conducts quality control and filtering steps
4. Performs differential methylation analysis comparing primary tumors to normal tissue
5. Identifies differentially methylated positions (DMPs) and regions (DMRs)
6. Annotates DMPs/DMRs with genomic context (promoters, gene bodies, etc.)
7. Generates visualizations including volcano plots, heatmaps, and genomic tracks
8. Saves the results to disk for downstream analysis

This analysis serves as a baseline comparison to identify prostate cancer-specific
methylation changes independent of FOLH1/PSMA expression levels.

Usage:
    python contrasts.py [--root-dir ROOT_DIR] [--processes NUM_PROCESSES]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --processes: Number of processes for parallel processing (default: CPU count - 2)
"""

import argparse
import logging
import multiprocessing
import warnings
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from components.functional_analysis.orgdb import OrgDB
from pipelines.differential_methylation.minfi_utils import (
    differential_methylation_array,
)

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
DATA_ROOT: Path = STORAGE.joinpath("TCGA-PRAD_MethArray")
RESULTS_PATH: Path = DATA_ROOT.joinpath("minfi")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
IDAT_PATH: Path = DATA_PATH.joinpath("idat")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation_common.csv")
GENOME: str = "hg38"
ARRAY_TYPE: str = "450K"
# Can be downloaded from [here](https://zwdzwd.github.io/InfiniumAnnotation).
GENOME_ANNO_FILE: Path = (
    STORAGE.joinpath("genomes")
    .joinpath("Homo_sapiens")
    .joinpath("InfiniumAnnotation")
    .joinpath(f"HM450.{GENOME}.manifest.tsv")
)
SAMPLE_CONTRAST_FACTOR: Iterable[str] = "sample_type"
CONTRASTS_LEVELS: List[Tuple[str, str]] = [
    ("prim", "norm"),
]
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "norm": "#9ACD32",
    "prim": "#4A708B",
}
NORM_TYPES: Iterable[str] = ("noob_quantile",)
N_PROCESSES: int = 4
P_COLS: Iterable[str] = ("P.Value", "adj.P.Val")
P_THS: Iterable[float] = (0.05, 0.01)
LFC_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
LFC_THS: Iterable[float] = (0.0, 1.0, 2.0)
MEAN_METH_DIFF_THS: Iterable[float] = (0.1, 0.2, 0.3)
HEATMAP_TOP_N: int = 1000
DMRS_TOP_N: int = 10
SPECIES: str = "Homo sapiens"

contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"
org_db = OrgDB(SPECIES)
genome_anno = pd.read_csv(GENOME_ANNO_FILE, sep="\t").set_index("Probe_ID")
targets = (
    pd.read_csv(ANNOT_PATH, dtype=str)
    .rename(columns={"barcode": "Basename"})
    .sort_values("probe")
    .drop_duplicates("Basename")
)
targets = targets[targets[SAMPLE_CONTRAST_FACTOR].isin(contrast_conditions)]

# 1. Generate input collection for all arguments' combinations
input_dict = dict(
    exp_prefix=exp_prefix,
    genome_anno=genome_anno,
    targets=targets,
    contrast_factor=SAMPLE_CONTRAST_FACTOR,
    contrasts_levels=CONTRASTS_LEVELS,
    idat_path=IDAT_PATH,
    results_path=RESULTS_PATH,
    plots_path=PLOTS_PATH,
    contrasts_levels_colors=CONTRASTS_LEVELS_COLORS,
    genome=GENOME,
    array_type=ARRAY_TYPE,
    norm_types=NORM_TYPES,
    n_processes=N_PROCESSES,
    p_cols=P_COLS,
    p_ths=P_THS,
    lfc_levels=LFC_LEVELS,
    lfc_ths=LFC_THS,
    mean_meth_diff_ths=MEAN_METH_DIFF_THS,
    heatmap_top_n=HEATMAP_TOP_N,
    dmrs_top_n=DMRS_TOP_N,
)


# 2. Run functional enrichment analysis
if __name__ == "__main__":
    differential_methylation_array(**input_dict)
