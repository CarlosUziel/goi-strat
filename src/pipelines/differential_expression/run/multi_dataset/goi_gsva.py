import argparse
import datetime
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
from tqdm.rich import tqdm
from utils import run_func_dict

from data.utils import filter_df, parallelize_map
from pipelines.differential_expression.utils import differential_expression

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
LFC_LEVELS: Iterable[str] = ("all", "up", "down")
LFC_THS: Iterable[float] = (1.0,)
HEATMAP_TOP_N: int = 1000
COMPUTE_VST: bool = True
COMPUTE_RLOG: bool = False
PARALLEL: bool = True
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
DATASETS_MARKERS: Dict[str, str] = {
    "TCGA-BRCA": "BRCA1",  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8998777/
    "TCGA-LUAD": "NKX2-1",  # https://www.nature.com/articles/nature09881
    "TCGA-THCA": "HMGA2",  # https://core.ac.uk/download/pdf/46919877.pdf#page=59
    "TCGA-UCEC": "PIK3CA",  # https://pubmed.ncbi.nlm.nih.gov/28860563/, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3060282/
    "TCGA-LUSC": "SOX2",  # https://www.cell.com/cancer-cell/fulltext/S1535-6108(16)30436-6
    "TCGA-KIRC": "CA9",  # https://www.sciencedirect.com/science/article/abs/pii/S0959804910006982
    "TCGA-HNSC": "TP63",  # https://aacrjournals.org/mcr/article/17/6/1279/270274/Loss-of-TP63-Promotes-the-Metastasis-of-Head-and
    "TCGA-LGG": "IDH1",  # https://www.neurology.org/doi/abs/10.1212/wnl.0b013e3181f96282
    "PCTA_WCDT": "FOLH1",  # https://www.nature.com/articles/nrurol.2016.26
}

# 1. Collect all function inputs for each tcga project
input_collection = []
input_collection_optimal = []

n_projects = len(DATASETS_MARKERS)
for i, (dataset_name, goi_symbol) in enumerate(DATASETS_MARKERS.items()):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{current_time}][{i}/{n_projects-1}] Processing inputs for {dataset_name}..."
    )

    # 1.1. Set GOI_SYMBOL and GOI_ENSEMBL
    GOI_SYMBOL = goi_symbol
    GOI_ENSEMBL = map_gene_id([GOI_SYMBOL], org_db, "SYMBOL", "ENSEMBL")[0]

    MAIN_ROOT: Path = STORAGE.joinpath(dataset_name)
    DATA_ROOT: Path = STORAGE.joinpath(f"{dataset_name}_{GOI_SYMBOL}")
    RESULTS_PATH: Path = DATA_ROOT.joinpath("deseq2")
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    PLOTS_PATH: Path = RESULTS_PATH.joinpath("plots")
    PLOTS_PATH.mkdir(exist_ok=True, parents=True)
    DATA_PATH: Path = DATA_ROOT.joinpath("data")
    ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}_gsva.csv")
    RAW_COUNTS_PATH: Path = MAIN_ROOT.joinpath("data").joinpath("raw_counts.csv")
    SAMPLE_CONTRAST_FACTOR: str = "sample_type"
    GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
    GOI_CLASS_PREFIX: str = f"{GOI_SYMBOL}_class"
    SAMPLE_CLUSTER_CONTRAST_LEVELS: Iterable[
        Iterable[Dict[str, Iterable[Union[int, str]]]]
    ] = tuple(
        [
            (
                {SAMPLE_CONTRAST_FACTOR: (test[0],), GOI_LEVEL_PREFIX: (test[1],)},
                {
                    SAMPLE_CONTRAST_FACTOR: (control[0],),
                    GOI_LEVEL_PREFIX: (control[1],),
                },
            )
            for test, control in [
                (("prim", "high"), ("prim", "low")),
            ]
        ]
    )

    with DATA_ROOT.joinpath("CONTRASTS_LEVELS_COLORS.json").open("w") as fp:
        json.dump(CONTRASTS_LEVELS_COLORS, fp, indent=True)

    annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
    counts_df = pd.read_csv(RAW_COUNTS_PATH, index_col=0)

    common_samples = annot_df.index.intersection(counts_df.columns)
    annot_df = annot_df.loc[common_samples, :]
    counts_df = counts_df.loc[:, common_samples]

    sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

    # 1.2. Generate input collection for each sample_cluster_contrast
    for sample_cluster_contrast in SAMPLE_CLUSTER_CONTRAST_LEVELS:
        # 1.2.1. Setup test_filters and control_filters
        test_filters, control_filters = sample_cluster_contrast

        # 1.2.2. Multi-level samples annotation
        annot_df_contrasts = deepcopy(annot_df)

        # 1.2.3. Keep only same sample type
        if (sample_type := test_filters[SAMPLE_CONTRAST_FACTOR][0]) == control_filters[
            SAMPLE_CONTRAST_FACTOR
        ][0]:
            annot_df_contrasts = annot_df_contrasts[
                annot_df_contrasts[SAMPLE_CONTRAST_FACTOR] == sample_type
            ]

        # 1.2.4. Annotation of test samples
        contrast_level_test = "_".join(chain(*test_filters.values()))
        annot_df_contrasts.loc[
            filter_df(annot_df_contrasts, test_filters).index, GOI_CLASS_PREFIX
        ] = contrast_level_test

        # 1.2.5. Annotation of control samples
        contrast_level_control = "_".join(chain(*control_filters.values()))
        annot_df_contrasts.loc[
            filter_df(annot_df_contrasts, control_filters).index, GOI_CLASS_PREFIX
        ] = contrast_level_control

        # 1.2.6. Set contrast levels and experiment prefix
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

        # 1.2.7. Append input collection
        input_collection.append(
            dict(
                annot_df_contrasts=annot_df_contrasts,
                counts_matrix=counts_df,
                results_path=RESULTS_PATH,
                plots_path=PLOTS_PATH,
                exp_prefix=exp_prefix,
                org_db=org_db,
                factors=[GOI_CLASS_PREFIX],
                design_factors=[GOI_CLASS_PREFIX],
                contrast_factor=GOI_CLASS_PREFIX,
                contrasts_levels=[contrasts_levels],
                contrast_levels_colors=CONTRASTS_LEVELS_COLORS,
                p_cols=P_COLS,
                p_ths=P_THS,
                lfc_levels=LFC_LEVELS,
                lfc_ths=LFC_THS,
                heatmap_top_n=HEATMAP_TOP_N,
                compute_vst=COMPUTE_VST,
                compute_rlog=COMPUTE_RLOG,
            )
        )

# 2. Run differential expression
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        # 2.1. Parallelize differential_expression function
        parallelize_map(
            functools.partial(run_func_dict, func=differential_expression),
            input_collection,
            threads=user_args["threads"] // 3,
            method="fork",
        )
    else:
        # 2.2. Run differential_expression function sequentially
        for ins in tqdm(input_collection):
            differential_expression(**ins)
