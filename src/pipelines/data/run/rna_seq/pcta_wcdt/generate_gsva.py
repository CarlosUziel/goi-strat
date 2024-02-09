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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    type=str,
    help="Root directory",
    nargs="?",
    default="/media/ssd/Perez/storage",
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
DATA_ROOT: Path = STORAGE.joinpath("PCTA_WCDT")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
GSVA_PATH: Path = DATA_PATH.joinpath("gsva")
GSVA_PATH.mkdir(parents=True, exist_ok=True)
RAW_COUNTS_PATH: Path = DATA_PATH.joinpath("raw_counts_wo_batch_effects.csv")
ANNOT_PATH: Path = DATA_PATH.joinpath("samples_annotation.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[str] = sorted(("prim", "met"))
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "prim": "#4A708B",
    "met": "#8B3A3A",
}
SPECIES: str = "Homo sapiens"
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

org_db = OrgDB(SPECIES)
annot_df = pd.read_csv(ANNOT_PATH, index_col=0)

annot_df_contrasts = deepcopy(
    annot_df[annot_df[SAMPLE_CONTRAST_FACTOR].isin(CONTRASTS_LEVELS)]
)

counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)
common_samples = annot_df_contrasts.index.intersection(counts_df.columns)
annot_df_contrasts = annot_df_contrasts.loc[common_samples, :]
counts_df = counts_df.loc[:, common_samples]

exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(CONTRASTS_LEVELS)}_"

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
            gsva_threads=(user_args["threads"] // len(MSIGDB_CATS)),
        )
    )


# 2. Run differential expression analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=generate_gsva_matrix),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            generate_gsva_matrix(**ins)
