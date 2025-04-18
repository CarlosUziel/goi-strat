"""
Gene Set Variation Analysis (GSVA) generation for WCDT-MCRPC RNA-seq data.

This script generates GSVA matrices for the WCDT-MCRPC (West Coast Dream Team Metastatic
Castration-Resistant Prostate Cancer) dataset. It transforms RNA-seq gene expression
data into pathway-level enrichment scores using the GSVA method.

The script performs the following tasks:
1. Loads RNA-seq count data from the WCDT-MCRPC dataset
2. Processes sample annotations, focusing on metastatic samples
3. For each MSigDB collection (Hallmarks, C1-C8), calculates GSVA scores
4. Saves the resulting GSVA matrices for downstream analyses

GSVA provides a non-parametric, unsupervised method for estimating variation of gene set
enrichment across samples, enabling pathway-centric analysis of the transcriptome data.
The generated matrices serve as input for other analyses including sample stratification
and differential pathway activity analysis.

Usage:
    python generate_gsva.py [--root-dir ROOT_DIR] [--processes NUM_PROCESSES]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --processes: Number of processes for parallel processing (default: CPU count - 2)
"""

import argparse
import functools
import logging
import multiprocessing
import warnings
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.data.utils import generate_gsva_matrix
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
ROOT: Path = STORAGE.joinpath("WCDT-MCRPC")
DATA_PATH: Path = ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
RAW_COUNTS_PATH: Path = DATA_PATH.joinpath("raw_counts_srr.csv")
GSVA_PATH: Path = DATA_PATH.joinpath("gsva")
GSVA_PATH.mkdir(exist_ok=True, parents=True)
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[str] = ("met",)
SPECIES: str = "Homo sapiens"
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

org_db = OrgDB(SPECIES)

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
annot_df = annot_df[annot_df["run"].notna()].set_index("run")
counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)

common_samples = annot_df.index.intersection(counts_df.columns)
annot_df = annot_df.loc[common_samples, :]
counts_df = counts_df.loc[:, common_samples]

exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(CONTRASTS_LEVELS)}_"

# 1. Generate input collection for all arguments' combinations
input_collection = []
for msigdb_cat in MSIGDB_CATS:
    input_collection.append(
        dict(
            counts_df=counts_df,
            annot_df=annot_df,
            org_db=org_db,
            msigdb_cat=msigdb_cat,
            save_path=GSVA_PATH.joinpath(f"{exp_prefix}_{msigdb_cat}.csv"),
            gsva_processes=(user_args["processes"] // len(MSIGDB_CATS)),
        )
    )


# 2. Run differential expression analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=generate_gsva_matrix),
            input_collection,
            processes=user_args["processes"],
        )
    else:
        for ins in tqdm(input_collection):
            generate_gsva_matrix(**ins)
