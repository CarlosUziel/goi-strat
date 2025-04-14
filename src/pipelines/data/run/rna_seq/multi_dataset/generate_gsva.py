"""
Gene Set Variation Analysis (GSVA) matrix generation for RNA-seq multi-dataset analysis.

This script generates GSVA matrices for multiple RNA-seq datasets across different MSigDB gene set
collections. GSVA transforms a gene expression matrix into a gene set expression matrix, allowing
for pathway-centric analysis rather than individual gene analysis.

The script performs the following operations:
1. Loads raw RNA-seq gene count data from multiple datasets
2. Filters samples based on specified contrast levels
3. For each dataset and MSigDB collection combination:
   - Computes GSVA scores for all gene sets in the collection
   - Saves results as CSV matrices for downstream analyses
   - Stores gene set metadata (descriptions and gene members)

The generated GSVA matrices serve as input for various downstream analyses including
sample stratification, differential enrichment analysis, and functional annotation.

The script supports parallel processing to efficiently handle multiple datasets and
gene set collections simultaneously.

Usage:
    python generate_gsva.py [--root-dir ROOT_DIR] [--threads NUM_THREADS]

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
from typing import Dict, Iterable

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

# Parse command line arguments
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
DATASET_NAMES: Iterable[str] = [
    "TCGA-BRCA",  # >1000 RNA-seq samples
    "TCGA-LUAD",  # 517 RNA-seq samples
    "TCGA-THCA",  # 505 RNA-seq samples
    "TCGA-UCEC",  # 557 RNA-seq samples
    "TCGA-LUSC",  # 501 RNA-seq samples
    "TCGA-KIRC",  # 533 RNA-seq samples
    "TCGA-HNSC",  # 512 RNA-seq samples
    "TCGA-LGG",  # 516 RNA-seq samples
    "PCTA-WCDT",
]
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[str] = sorted(("prim",))
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "prim": "#4A708B",
}
SPECIES: str = "Homo sapiens"
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

org_db = OrgDB(SPECIES)

# 1. Generate input collection for all arguments' combinations
input_collection = []
for project_name in DATASET_NAMES:
    DATA_ROOT: Path = STORAGE.joinpath(project_name)
    DATA_PATH: Path = DATA_ROOT.joinpath("data")
    GSVA_PATH: Path = DATA_PATH.joinpath("gsva")
    GSVA_PATH.mkdir(parents=True, exist_ok=True)
    RAW_COUNTS_PATH: Path = DATA_PATH.joinpath("raw_counts.csv")
    ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")

    annot_df = pd.read_csv(ANNOT_PATH, index_col=0)

    annot_df_contrasts = deepcopy(
        annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(CONTRASTS_LEVELS)]
    )

    counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)
    common_samples = annot_df_contrasts.index.intersection(counts_df.columns)
    annot_df_contrasts = annot_df_contrasts.loc[common_samples, :]
    counts_df = counts_df.loc[:, common_samples]

    exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(CONTRASTS_LEVELS)}_"

    for msigdb_cat in MSIGDB_CATS:
        # 1.1. Append input dictionary to input collection
        input_collection.append(
            dict(
                counts_df=counts_df,
                annot_df=annot_df_contrasts,
                org_db=org_db,
                msigdb_cat=msigdb_cat,
                save_path=GSVA_PATH.joinpath(f"{exp_prefix}_{msigdb_cat}.csv"),
                gsva_threads=1,
            )
        )

# 2. Generate GSVA matrices
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        # 2.1. Parallelize the generation of GSVA matrices
        parallelize_map(
            functools.partial(run_func_dict, func=generate_gsva_matrix),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            # 2.2. Generate GSVA matrix for each input dictionary
            generate_gsva_matrix(**ins)
