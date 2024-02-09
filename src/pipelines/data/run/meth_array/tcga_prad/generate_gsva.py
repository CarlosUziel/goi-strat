import argparse
import functools
import logging
import multiprocessing
import warnings
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
RNA_MAIN_ROOT: Path = STORAGE.joinpath("PCTA_WCDT")
METH_MAIN_ROOT: Path = STORAGE.joinpath("TCGA_PRAD_MethArray")
METH_DATA_PATH: Path = METH_MAIN_ROOT.joinpath("data")
RNA_ANNOT_PATH: Path = RNA_MAIN_ROOT.joinpath("data").joinpath("samples_annotation.csv")
METH_ANNOT_PATH: Path = METH_DATA_PATH.joinpath("samples_annotation.csv")
RAW_COUNTS_PATH: Path = RNA_MAIN_ROOT.joinpath("data").joinpath(
    "raw_counts_wo_batch_effects.csv"
)
GSVA_PATH: Path = METH_DATA_PATH.joinpath("gsva")
GSVA_PATH.mkdir(exist_ok=True, parents=True)
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
CONTRASTS_LEVELS: Iterable[str] = ("prim",)
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "prim": "#4A708B",
}
SPECIES: str = "Homo sapiens"
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

org_db = OrgDB(SPECIES)

rna_annot_df = pd.read_csv(RNA_ANNOT_PATH, index_col=0).drop_duplicates(keep=False)
rna_annot_df = rna_annot_df[rna_annot_df[SAMPLE_CONTRAST_FACTOR].isin(CONTRASTS_LEVELS)]

meth_annot_df = pd.read_csv(METH_ANNOT_PATH, index_col=0).drop_duplicates(keep=False)
meth_annot_df = meth_annot_df[
    meth_annot_df[SAMPLE_CONTRAST_FACTOR].isin(CONTRASTS_LEVELS)
]
meth_annot_df = meth_annot_df[~meth_annot_df.index.duplicated(keep="first")]

counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)

# get common methylation and rnaseq samples
common_samples = rna_annot_df[
    rna_annot_df["sample_id"].isin(
        [sample_id[:-1] for sample_id in meth_annot_df.index]
    )
].index.intersection(counts_df.columns)

rna_annot_df = rna_annot_df.loc[common_samples, :]
counts_df = counts_df.loc[:, common_samples]

exp_prefix = f"{SAMPLE_CONTRAST_FACTOR}_{'+'.join(CONTRASTS_LEVELS)}_"

# 1. Generate input collection for all arguments' combinations
input_collection = []
for msigdb_cat in MSIGDB_CATS:
    input_collection.append(
        dict(
            counts_df=counts_df,
            annot_df=rna_annot_df,
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
