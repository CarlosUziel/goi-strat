import argparse
import datetime
import functools
import logging
import multiprocessing
import warnings
from itertools import product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import rpy2.robjects as ro
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.data.utils import goi_perc_annotation_rna_seq
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

SPECIES: str = "Homo sapiens"
org_db = OrgDB(SPECIES)

P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("all",)
LFC_THS: Iterable[float] = (0.0,)
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
MIN_PERCENTILE: float = 0.1
MID_PERCENTILE: float = 0.5
PARALLEL: bool = True
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
DATASETS_MARKERS: Dict[str, str] = {
    "TCGA-BRCA": "FOXA1",  # https://www.sciencedirect.com/science/article/abs/pii/S0960977616000242
    "TCGA-LUAD": "NKX2-1",  # https://www.nature.com/articles/nature09881
    "TCGA-THCA": "BRAF",  # https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2024.1372553/full
    "TCGA-UCEC": "MCM10",  # https://onlinelibrary.wiley.com/doi/full/10.1111/jcmm.17772
    "TCGA-LUSC": "SOX2",  # https://www.cell.com/cancer-cell/fulltext/S1535-6108(16)30436-6
    "TCGA-KIRC": "CA9",  # https://www.sciencedirect.com/science/article/abs/pii/S0959804910006982
    "TCGA-HNSC": "TP63",  # https://aacrjournals.org/mcr/article/17/6/1279/270274/Loss-of-TP63-Promotes-the-Metastasis-of-Head-and
    "TCGA-LGG": "IDH1",  # https://www.neurology.org/doi/abs/10.1212/wnl.0b013e3181f96282
    "PCTA_WCDT": "FOLH1",  # https://www.nature.com/articles/nrurol.2016.26
}
PERCENTILES: Iterable[int] = (10, 15, 20, 25, 30)

# 1. Collect all function inputs for each dataset
input_collection = []
n_projects = len(DATASETS_MARKERS)
for i, (dataset_name, goi_symbol) in enumerate(DATASETS_MARKERS.items()):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{current_time}][{i}/{n_projects-1}] Processing inputs for {dataset_name}..."
    )

    GOI_SYMBOL = goi_symbol
    GOI_ENSEMBL = map_gene_id([GOI_SYMBOL], org_db, "SYMBOL", "ENSEMBL")[0]

    MAIN_ROOT: Path = STORAGE.joinpath(dataset_name)
    ANNOT_PATH: Path = MAIN_ROOT.joinpath("data").joinpath("samples_annotation.csv")
    RAW_COUNTS_PATH: Path = MAIN_ROOT.joinpath("data").joinpath("raw_counts.csv")
    DATA_ROOT: Path = STORAGE.joinpath(f"{dataset_name}_{GOI_SYMBOL}")
    DATA_PATH: Path = DATA_ROOT.joinpath("data")
    DATA_PATH.mkdir(exist_ok=True, parents=True)
    PLOTS_PATH: Path = DATA_ROOT.joinpath("plots")
    PLOTS_PATH.mkdir(exist_ok=True, parents=True)
    ANNOT_PATH_NEW: Path = DATA_PATH.joinpath(
        f"{ANNOT_PATH.stem}_{GOI_SYMBOL}_perc.csv"
    )
    SAMPLE_CONTRAST_FACTOR: str = "sample_type"
    CONTRASTS_LEVELS: Iterable[str] = sorted(("prim",))
    GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"

    annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
    annot_df_contrasts = annot_df[
        annot_df[SAMPLE_CONTRAST_FACTOR].isin(CONTRASTS_LEVELS)
    ]
    counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)

    common_samples = annot_df_contrasts.index.intersection(counts_df.columns)
    annot_df_contrasts = annot_df_contrasts.loc[common_samples, :]
    counts_df = counts_df.loc[:, common_samples]

    sample_types_str = "+".join(sorted(set(annot_df_contrasts[SAMPLE_CONTRAST_FACTOR])))
    annot_df_contrasts[f"{GOI_SYMBOL}_CNT"] = counts_df.loc[
        GOI_ENSEMBL, annot_df_contrasts.index
    ]

    vst_file_path = MAIN_ROOT.joinpath("data").joinpath(
        f"{sample_types_str}_vst_counts.csv"
    )
    if vst_file_path.exists():
        vst_df = pd.read_csv(vst_file_path, index_col=0)
    else:
        vst_df = rpy2_df_to_pd_df(
            vst_transform(
                ro.r("as.matrix")(
                    pd_df_to_rpy2_df(counts_df.loc[counts_df.mean(axis=1) > 1])
                )
            )
        )
        vst_df.to_csv(vst_file_path)

    annot_df_contrasts[f"{GOI_SYMBOL}_VST"] = vst_df.loc[
        GOI_ENSEMBL, annot_df_contrasts.index
    ]

    # 1.1. Append function inputs to input_collection
    input_collection.append(
        dict(
            annot_df=annot_df_contrasts,
            vst_df=vst_df,
            plots_path=PLOTS_PATH,
            data_path=DATA_PATH,
            new_annot_file=ANNOT_PATH_NEW,
            goi_symbol=GOI_SYMBOL,
            sample_contrast_factor=SAMPLE_CONTRAST_FACTOR,
            contrast_levels=CONTRASTS_LEVELS,
            contrast_levels_colors=CONTRASTS_LEVELS_COLORS,
            percentiles=PERCENTILES,
        )
    )

print("Finished collecting function inputs.")

# 2. Run percentile splits annotation
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=goi_perc_annotation_rna_seq),
            input_collection,
            threads=user_args["threads"],
            method="fork",
        )
    else:
        for ins in tqdm(input_collection):
            goi_perc_annotation_rna_seq(**ins)
