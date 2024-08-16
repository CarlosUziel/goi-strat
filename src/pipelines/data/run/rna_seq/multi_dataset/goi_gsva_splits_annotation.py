import argparse
import datetime
import functools
import logging
import multiprocessing
import warnings
from copy import deepcopy
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

# 1. Collect all function inputs for each dataset
input_collection = []
input_collection_optimal = []
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
    GSVA_PATH: Path = MAIN_ROOT.joinpath("data").joinpath("gsva")
    DATA_ROOT: Path = STORAGE.joinpath(f"{dataset_name}_{GOI_SYMBOL}")
    RESULTS_PATH: Path = DATA_ROOT.joinpath("group_splits_gsva")
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    DATA_PATH: Path = DATA_ROOT.joinpath("data")
    DATA_PATH.mkdir(exist_ok=True, parents=True)
    ANNOT_PATH_NEW: Path = DATA_PATH.joinpath(
        f"{ANNOT_PATH.stem}_{GOI_SYMBOL}_gsva.csv"
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

    # 1.1. Collect all function inputs for each group split
    group_counts = []
    for contrast_level in CONTRASTS_LEVELS:
        # 1.1.1. Get ranked list of samples using VST values
        annot_df_sorted = annot_df_contrasts[
            annot_df_contrasts[SAMPLE_CONTRAST_FACTOR] == contrast_level
        ].sort_values(f"{GOI_SYMBOL}_VST", ascending=True)

        # 1.1.2. Define initial group distribution
        n = len(annot_df_sorted)
        low_min, high_min = [int(n * MIN_PERCENTILE)] * 2

        mid_n = int(n * MID_PERCENTILE)
        low_n = low_min
        high_n = n - (low_n + mid_n)

        while high_n >= high_min:
            group_counts.append((contrast_level, low_n, mid_n, high_n))

            # 1.1.3. Build annotation file with new groups
            annot_df_sorted_groups = deepcopy(annot_df_sorted)
            annot_df_sorted_groups[GOI_LEVEL_PREFIX] = (
                ["low"] * low_n + ["mid"] * mid_n + ["high"] * high_n
            )

            exp_prefix = (
                f"{contrast_level}_{GOI_LEVEL_PREFIX}_high_{high_n}+low_{low_n}_"
            )

            # 1.1.4. Inputs for each MSigDB category
            for msigdb_cat in MSIGDB_CATS:
                results_path = RESULTS_PATH.joinpath(msigdb_cat)
                results_path.mkdir(exist_ok=True, parents=True)

                # 1.1.5. Generate input collection for all arguments' combinations
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

            # 1.1.6. Increase counters
            low_n += 1
            high_n -= 1

    input_collection_optimal.append(
        dict(
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
    )

    print(
        "[Diff. Enrich] Number of input arguments collected so far: ",
        len(input_collection),
    )
    print(
        "[Optimal Split] Number of input arguments collected so far: ",
        len(input_collection_optimal),
    )

print("Finished collecting function inputs.")

# 2. Use differential enrichment results to find the best split
if __name__ == "__main__":
    freeze_support()
    # 2.1. Run differential enrichment
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=diff_enrich_gsva_limma),
            input_collection,
            threads=user_args["threads"],
            method="fork",
        )
    else:
        for ins in tqdm(input_collection):
            diff_enrich_gsva_limma(**ins)

    # 2.2. Find the optimal split
    if PARALLEL and len(input_collection_optimal) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=get_optimal_gsva_splits),
            input_collection_optimal,
            threads=user_args["threads"],
            method="fork",
        )
    else:
        for ins in tqdm(input_collection_optimal):
            get_optimal_gsva_splits(**ins)
