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
from pipelines.differential_enrichment.utils import diff_enrich_gsva
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
DATA_ROOT: Path = STORAGE.joinpath(f"PCTA_WCDT_{GOI_SYMBOL}")
RESULTS_PATH: Path = DATA_ROOT.joinpath("diff_gsva")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
GSVA_PATH: Path = STORAGE.joinpath("PCTA_WCDT").joinpath("data").joinpath("gsva")
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
        for test, control in [
            (("prim", "high"), ("prim", "low")),
            (("met", "high"), ("met", "low")),
        ]
    ]
)
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "prim": "#4A708B",
    "met": "#8B3A3A",
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

P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("up", "down", "all")
LFC_THS: Iterable[float] = (0.0,)
HEATMAP_TOP_N: int = 1000
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))
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

input_collection = []
for sample_cluster_contrast, msigdb_cat in product(
    SAMPLE_CLUSTER_CONTRAST_LEVELS, MSIGDB_CATS
):
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

    results_path = RESULTS_PATH.joinpath(msigdb_cat)
    results_path.mkdir(exist_ok=True, parents=True)

    plots_path = PLOTS_PATH.joinpath(msigdb_cat)
    plots_path.mkdir(exist_ok=True, parents=True)

    # 2. Generate input collection for all arguments' combinations
    input_collection.append(
        dict(
            gsva_matrix_path=gsva_matrices[msigdb_cat],
            msigdb_cat_meta_path=msigdb_cats_meta[msigdb_cat],
            annot_df_contrasts=annot_df_contrasts,
            contrast_factor=GOI_CLASS_PREFIX,
            results_path=results_path,
            plots_path=plots_path,
            exp_prefix=exp_prefix,
            contrasts_levels=[contrasts_levels],
            contrast_levels_colors=CONTRASTS_LEVELS_COLORS,
            p_cols=P_COLS,
            p_ths=P_THS,
            lfc_levels=LFC_LEVELS,
            lfc_ths=LFC_THS,
            heatmap_top_n=HEATMAP_TOP_N,
            design_factors=[GOI_CLASS_PREFIX],
        )
    )

# 3. Run differential enrichment
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=diff_enrich_gsva),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            diff_enrich_gsva(**ins)
