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
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.utils import map_gene_id
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm
from utils import run_func_dict

from data.utils import filter_df, parallelize_map
from pipelines.differential_methylation.biseq_utils import (
    differential_methylation_rrbs_regions,
)

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
DATA_ROOT: Path = STORAGE.joinpath(f"WCDT_MCRPC_{GOI_SYMBOL}")
RESULTS_PATH: Path = DATA_ROOT.joinpath("biseq")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
BISMARK_PATH: Path = STORAGE.joinpath("WCDT_MCRPC").joinpath("data").joinpath("bismark")
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
    "norm": "#9ACD32",
    "prim": "#4A708B",
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

N_THREADS: int = 16
FDR_THS: Iterable[float] = (0.05, 0.01)
MEAN_DIFF_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
MEAN_DIFF_THS: Iterable[float] = (10, 20, 30)
PARALLEL: bool = True

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
            n_threads=N_THREADS,
            fdr_ths=FDR_THS,
            mean_diff_levels=MEAN_DIFF_LEVELS,
            mean_diff_ths=MEAN_DIFF_THS,
        )
    )

# 3. Run differential methylation analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(
                run_func_dict, func=differential_methylation_rrbs_regions
            ),
            input_collection,
            threads=user_args["threads"] // 4,
        )
    else:
        for ins in tqdm(input_collection):
            differential_methylation_rrbs_regions(**ins)
