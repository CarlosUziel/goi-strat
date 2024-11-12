import argparse
import datetime
import functools
import json
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Union

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.functional_analysis.utils import functional_enrichment
from r_wrappers.utils import map_gene_id
from utils import run_func_dict

_ = traceback.install()
rpy2_logger.setLevel(logging.WARNING)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.WARNING)
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
    "TCGA-THCA": "HMGA2",  # https://pubmed.ncbi.nlm.nih.gov/17943974/
    "TCGA-UCEC": "PIK3CA",  # https://pubmed.ncbi.nlm.nih.gov/28860563/, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3060282/
    "TCGA-LUSC": "SOX2",  # https://www.cell.com/cancer-cell/fulltext/S1535-6108(16)30436-6
    "TCGA-KIRC": "CA9",  # https://www.sciencedirect.com/science/article/abs/pii/S0959804910006982
    "TCGA-HNSC": "TP63",  # https://aacrjournals.org/mcr/article/17/6/1279/270274/Loss-of-TP63-Promotes-the-Metastasis-of-Head-and
    "TCGA-LGG": "IDH1",  # https://www.neurology.org/doi/abs/10.1212/wnl.0b013e3181f96282
    "PCTA-WCDT": "FOLH1",  # https://www.nature.com/articles/nrurol.2016.26
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
    FUNC_PATH: Path = DATA_ROOT.joinpath("functional")
    FUNC_PATH.mkdir(exist_ok=True, parents=True)
    PLOTS_PATH: Path = FUNC_PATH.joinpath("plots")
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
        # 1. Setup
        test_filters, control_filters = sample_cluster_contrast

        # 1.1. Annotation of test samples
        contrast_level_test = "_".join(chain(*test_filters.values()))

        # 1.2. Annotation of control samples
        contrast_level_control = "_".join(chain(*control_filters.values()))

        # 1.3. Set contrast levels and experiment prefix
        contrasts_levels = (contrast_level_test, contrast_level_control)
        exp_prefix = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )

        # 1.4. Set experiment name
        exp_name = f"{exp_prefix}_{contrast_level_test}_vs_{contrast_level_control}"

        # 1.5. Load input DEGS
        results_file = DATA_ROOT.joinpath("deseq2").joinpath(
            f"{exp_name}_deseq_results_unique.csv"
        )

        if not results_file.exists():
            continue

        # 2. Add GSEA inputs
        input_collection.append(
            dict(
                data_type="diff_expr",
                func_path=FUNC_PATH,
                plots_path=PLOTS_PATH,
                results_file=results_file,
                exp_name=exp_name,
                org_db=org_db,
                numeric_col="log2FoldChange",
                analysis_type="gsea",
            )
        )

        # 3. Add ORA inputs
        for p_col, p_th, lfc_level, lfc_th in product(
            P_COLS, P_THS, LFC_LEVELS, LFC_THS
        ):
            p_thr_str = str(p_th).replace(".", "_")
            lfc_thr_str = str(lfc_th).replace(".", "_")
            input_collection.append(
                dict(
                    data_type="diff_expr",
                    func_path=FUNC_PATH,
                    plots_path=PLOTS_PATH,
                    results_file=results_file,
                    exp_name=f"{exp_name}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}",
                    org_db=org_db,
                    cspa_surfaceome_file=STORAGE.joinpath(
                        "CSPA_validated_surfaceome_proteins_human.csv"
                    ),
                    p_col=p_col,
                    p_th=p_th,
                    lfc_col="log2FoldChange",
                    lfc_level=lfc_level,
                    lfc_th=lfc_th,
                    numeric_col="log2FoldChange",
                    analysis_type="ora",
                )
            )


# 4. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=functional_enrichment),
            input_collection,
            threads=user_args["threads"] // 3,
            method="fork",
        )
    else:
        for ins in tqdm(input_collection):
            functional_enrichment(**ins)
