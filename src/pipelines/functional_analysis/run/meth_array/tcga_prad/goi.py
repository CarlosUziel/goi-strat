import argparse
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
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.utils import map_gene_id
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm import tqdm
from utils import run_func_dict

from data.utils import parallelize_map
from pipelines.functional_analysis.utils import functional_enrichment

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
DATA_ROOT: Path = STORAGE.joinpath(f"TCGA_PRAD_MethArray_{GOI_SYMBOL}")
FUNC_PATH: Path = DATA_ROOT.joinpath("functional")
FUNC_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH: Path = FUNC_PATH.joinpath("plots")
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
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
        for test, control in [(("prim", "high"), ("prim", "low"))]
    ]
)
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
with DATA_ROOT.joinpath("CONTRASTS_LEVELS_COLORS.json").open("w") as fp:
    json.dump(CONTRASTS_LEVELS_COLORS, fp, indent=True)

ID_COL: str = "sample_id"
NORM_TYPES: Iterable[str] = ("noob_quantile",)
GENE_ANNOTS: Iterable[str] = (
    f"{GENOME}_genes_1to5kb",
    f"{GENOME}_genes_exons",
    f"{GENOME}_genes_introns",
    f"{GENOME}_genes_promoters",
)
N_THREADS: int = 4
P_COLS: Iterable[str] = ("P.Value", "adj.P.Val")
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
LFC_THS: Iterable[float] = (1.0,)
MEAN_METH_DIFF_THS: Iterable[float] = (0.0, 0.1, 0.2)
SPECIES: str = "Homo sapiens"
PARALLEL: bool = True

annot_df = (
    pd.read_csv(ANNOT_PATH, index_col=0, dtype=str).rename_axis(ID_COL).reset_index()
)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for sample_cluster_contrast in SAMPLE_CLUSTER_CONTRAST_LEVELS:
    # 0. Setup
    test_filters, control_filters = sample_cluster_contrast

    # 1.1. Annotation of test samples
    contrast_level_test = "_".join(chain(*test_filters.values()))

    # 1.2. Annotation of control samples
    contrast_level_control = "_".join(chain(*control_filters.values()))

    # 1.3. Set contrast levels and experiment prefix
    contrasts_levels = (contrast_level_test, contrast_level_control)

    for (
        norm_type,
        gene_annot,
        mean_meth_diff_th,
    ) in product(NORM_TYPES, GENE_ANNOTS, MEAN_METH_DIFF_THS):
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        exp_prefix = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}__"
            f"diff_meth_probes_{norm_type}_top_table_"
            f"{contrast_level_test}_vs_{contrast_level_control}"
        )
        exp_name = f"{exp_prefix}_wrt_mean_diff_{mean_meth_diff_th_str}_{gene_annot}"
        results_file = DATA_ROOT.joinpath("minfi").joinpath(f"{exp_name}.csv")

        if not results_file.exists():
            continue

        # 2.1. Add GSEA inputs
        input_collection.append(
            dict(
                data_type="diff_meth",
                func_path=FUNC_PATH,
                plots_path=PLOTS_PATH,
                results_file=results_file,
                exp_name=exp_name,
                org_db=org_db,
                numeric_col="logFC",
                analysis_type="gsea",
            )
        )

        # 2.2. Add ORA inputs
        for p_col, p_th, lfc_level, lfc_th in product(
            P_COLS, P_THS, LFC_LEVELS, LFC_THS
        ):
            p_col_str = p_col.replace(".", "_")
            p_thr_str = str(p_th).replace(".", "_")
            lfc_thr_str = str(lfc_th).replace(".", "_")
            input_collection.append(
                dict(
                    data_type="diff_meth",
                    func_path=FUNC_PATH,
                    plots_path=PLOTS_PATH,
                    results_file=results_file,
                    exp_name=(
                        f"{exp_prefix}_sig_{p_col_str}_{p_thr_str}_{lfc_level}_"
                        f"{lfc_thr_str}_wrt_mean_diff_{mean_meth_diff_th_str}_"
                        f"{gene_annot}"
                    ),
                    org_db=org_db,
                    cspa_surfaceome_file=STORAGE.joinpath(
                        "CSPA_validated_surfaceome_proteins_human.csv"
                    ),
                    p_col=p_col,
                    p_th=p_th,
                    lfc_col="logFC",
                    lfc_level=lfc_level,
                    lfc_th=lfc_th,
                    numeric_col="logFC",
                    analysis_type="ora",
                )
            )

# 3. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=functional_enrichment),
            input_collection,
            threads=user_args["threads"] // 3,
        )
    else:
        for ins in tqdm(input_collection):
            functional_enrichment(**ins)
