import argparse
import functools
import logging
import multiprocessing
import warnings
from itertools import chain, product
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.integrative_analysis.utils import intersect_degss_gsea
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
PRIM_METH_ROOT: Path = STORAGE.joinpath(f"TCGA_PRAD_MethArray_{GOI_SYMBOL}")
MET_METH_ROOT: Path = STORAGE.joinpath(f"WCDT_MCRPC_{GOI_SYMBOL}")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
ANNOT_PATH: Path = DATA_PATH.joinpath(f"samples_annotation_{GOI_SYMBOL}.csv")
SAMPLE_CONTRAST_FACTOR: str = "sample_type"
GOI_LEVEL_PREFIX: str = f"{GOI_SYMBOL}_level"
CONTRAST_COMPARISONS: Dict[
    str,
    Iterable[
        Tuple[Dict[Iterable[str], Iterable[str]], Dict[Iterable[str], Iterable[str]]]
    ],
] = {
    "met_prim": (
        (
            {SAMPLE_CONTRAST_FACTOR: ["met"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["prim"], GOI_LEVEL_PREFIX: ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"], GOI_LEVEL_PREFIX: ["low"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["met"], GOI_LEVEL_PREFIX: ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["met"], GOI_LEVEL_PREFIX: ["low"]},
        ),
    ),
}
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("up", "down", "all")
LFC_THS: Iterable[float] = (0.0,)
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
sample_types_str = "+".join(sorted(set(annot_df[SAMPLE_CONTRAST_FACTOR])))

input_collection = []
for contrast_comparison, contrast_comparison_filters in CONTRAST_COMPARISONS.items():
    contrast_prefixes = {}
    # 1. Get contrast file prefixes
    for contrast_filters in contrast_comparison_filters:
        # 1.1. Setup
        test_filters, control_filters = contrast_filters

        # 1.2. Multi-level samples annotation
        # 1.2.1. Annotation of test samples
        contrast_level_test = "_".join(chain(*test_filters.values()))

        # 1.2.2. Annotation of control samples
        contrast_level_control = "_".join(chain(*control_filters.values()))

        # 1.3. Set experiment prefix and remove unnecesary samples
        contrasts_levels = (contrast_level_test, contrast_level_control)
        exp_prefix_rna = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )
        exp_prefix_meth_prim = (
            f"{SAMPLE_CONTRAST_FACTOR}_prim_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )
        exp_prefix_meth_met = (
            f"{SAMPLE_CONTRAST_FACTOR}_met_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )

        contrast_name_prefix = (
            f"{contrast_level_test}_vs_{contrast_level_control}",
            f"{exp_prefix_rna}_{contrast_level_test}_vs_{contrast_level_control}",
        )

        for msigdb_cat, p_col, p_th, lfc_level, lfc_th in product(
            MSIGDB_CATS, P_COLS, P_THS, LFC_LEVELS, LFC_THS
        ):
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")

            test_sample_type, control_sample_type = (
                contrast_level_test.split("_")[0],
                contrast_level_control.split("_")[0],
            )

            gsea_files = {
                "rna_gsea": (
                    DATA_ROOT.joinpath("functional")
                    .joinpath("MSIGDB")
                    .joinpath(f"{contrast_name_prefix[1]}_gsea_{msigdb_cat}.csv")
                )
            }
            if test_sample_type == control_sample_type:
                if test_sample_type == "prim":
                    gsea_files["meth_promoters_gsea"] = (
                        PRIM_METH_ROOT.joinpath("functional")
                        .joinpath("MSIGDB")
                        .joinpath(
                            f"{exp_prefix_meth_prim}_diff_meth_probes_"
                            "noob_quantile_top_table_"
                            f"{contrast_name_prefix[0]}_wrt_mean_diff_0_0_"
                            f"hg38_genes_promoters_gsea_{msigdb_cat}.csv"
                        )
                    )
                    gsea_files["meth_exons_gsea"] = (
                        PRIM_METH_ROOT.joinpath("functional")
                        .joinpath("MSIGDB")
                        .joinpath(
                            f"{exp_prefix_meth_prim}_diff_meth_probes_"
                            "noob_quantile_top_table_"
                            f"{contrast_name_prefix[0]}_wrt_mean_diff_0_0_"
                            f"hg38_genes_exons_gsea_{msigdb_cat}.csv"
                        )
                    )
                elif test_sample_type == "met":
                    gsea_files["meth_promoters_gsea"] = (
                        MET_METH_ROOT.joinpath("methylkit")
                        .joinpath("functional")
                        .joinpath("MSIGDB")
                        .joinpath(
                            f"{exp_prefix_meth_met}_diff_meth_"
                            f"{contrast_name_prefix[0]}_"
                            f"ann_genes_hg38_genes_promoters_gsea_{msigdb_cat}.csv"
                        )
                    )
                    gsea_files["meth_exons_gsea"] = (
                        MET_METH_ROOT.joinpath("methylkit")
                        .joinpath("functional")
                        .joinpath("MSIGDB")
                        .joinpath(
                            f"{exp_prefix_meth_met}_diff_meth_"
                            f"{contrast_name_prefix[0]}_"
                            f"ann_genes_hg38_genes_exons_gsea_{msigdb_cat}.csv"
                        )
                    )

            # 2. Generate input collection for all arguments' combinations
            input_collection.append(
                dict(
                    contrast_name_prefix=contrast_name_prefix[:2],
                    root_path=DATA_ROOT,
                    msigdb_cat=msigdb_cat,
                    gsea_files=gsea_files,
                    comparison_alias=(
                        f"{contrast_name_prefix[0]}_degss_"
                        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                        f"gsea_{msigdb_cat}"
                    ),
                    p_col=p_col,
                    p_th=p_th,
                    lfc_level=lfc_level,
                    lfc_th=lfc_th,
                )
            )

# 3. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=intersect_degss_gsea),
            input_collection,
            threads=user_args["threads"],
        )
    else:
        for ins in tqdm(input_collection):
            intersect_degss_gsea(**ins)
