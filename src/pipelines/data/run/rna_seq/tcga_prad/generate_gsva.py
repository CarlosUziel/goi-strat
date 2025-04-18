"""
Script to generate Gene Set Variation Analysis (GSVA) matrices for the TCGA-PRAD dataset.

This script preprocesses RNA-seq data from the TCGA-PRAD (Prostate Adenocarcinoma)
project and performs Gene Set Variation Analysis (GSVA) to compute enrichment scores
for collections of gene sets across samples.

The script performs the following steps:
1. Loads RNA-seq count data and sample annotations
2. Filters samples to include only those with matching annotations and data
3. For each MSigDB collection (Hallmarks, C1-C8), calculates GSVA enrichment scores
4. Saves the resulting matrices to disk for downstream analysis

The analysis focuses on comparing primary tumor (prim) and normal tissue (norm) samples,
providing a comprehensive view of pathway activity differences between these groups.

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
from copy import deepcopy
from itertools import chain
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
DATA_ROOT: Path = STORAGE.joinpath("TCGA-PRAD_RNASeq")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
RAW_COUNTS_PATH: Path = DATA_PATH.joinpath("raw_counts_wo_batch_effects.csv")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
GSVA_PATH: Path = DATA_PATH.joinpath("gsva")
GSVA_PATH.mkdir(exist_ok=True, parents=True)
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[Tuple[str, str]] = [
    ("prim", "norm"),
]
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "norm": "#9ACD32",
    "prim": "#4A708B",
}
SPECIES: str = "Homo sapiens"
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

org_db = OrgDB(SPECIES)
contrast_conditions = sorted(set(chain(*CONTRASTS_LEVELS)))
annot_df = pd.read_csv(ANNOT_PATH, index_col=0).drop_duplicates(keep=False)
annot_df_contrasts = deepcopy(
    annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(contrast_conditions)]
)

counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)
common_samples = annot_df_contrasts.index.intersection(counts_df.columns)
annot_df_contrasts = annot_df_contrasts.loc[common_samples, :]
counts_df = counts_df.loc[:, common_samples]

exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(contrast_conditions)}_"

# 1. Generate input collection for all arguments' combinations
input_collection = []
for msigdb_cat in MSIGDB_CATS:
    input_collection.append(
        dict(
            counts_df=counts_df,
            annot_df=annot_df_contrasts,
            org_db=org_db,
            msigdb_cat=msigdb_cat,
            save_path=GSVA_PATH.joinpath(f"{exp_prefix}_{msigdb_cat}.csv"),
            gsva_processes=(user_args["processes"] // len(MSIGDB_CATS)),
        )
    )


# 2. Generate GSVA matrices
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
