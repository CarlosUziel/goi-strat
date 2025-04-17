"""
Functional enrichment analysis pipeline for RRBS methylation data using BiSeq results.

This script performs Gene Set Enrichment Analysis (GSEA) and Over-Representation Analysis (ORA)
on differentially methylated regions (DMRs) identified by BiSeq from WCDT-MCRPC RRBS data.
The analysis focuses on understanding the biological significance of DNA methylation differences
associated with FOLH1/PSMA expression levels in metastatic prostate cancer.

The script:
1. Processes BiSeq DMR results from multiple genomic contexts (promoters, exons, introns, etc.)
2. Performs GSEA using methylation differences (meth.diff) for gene ranking
3. Performs ORA on filtered methylation regions using various FDR and difference thresholds
4. Supports hyper- and hypo-methylation analysis separately or combined
5. Generates comprehensive functional annotations and visualizations
6. Integrates with multiple gene set databases (GO, KEGG, MSigDB, etc.)

This analysis provides insights into the epigenetic regulatory mechanisms associated with
FOLH1/PSMA expression patterns in metastatic castration-resistant prostate cancer.

Usage:
    python goi_biseq.py [--root-dir DIRECTORY] [--processes NUM_PROCESSES]
"""

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
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from data.utils import parallelize_map
from pipelines.functional_analysis.utils import functional_enrichment
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
    help="Root directory for data storage",
    nargs="?",
    default="/mnt/d/phd_data",
)
parser.add_argument(
    "--processes",
    type=int,
    help="Number of processes for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
GOI_ENSEMBL: str = "ENSG00000086205"  # ENSEMBL ID for FOLH1 (PSMA)
SPECIES: str = "Homo sapiens"
org_db = OrgDB(SPECIES)
GOI_SYMBOL = map_gene_id([GOI_ENSEMBL], org_db, "ENSEMBL", "SYMBOL")[0]
DATA_ROOT: Path = STORAGE.joinpath(f"WCDT-MCRPC_{GOI_SYMBOL}")
BISEQ_ROOT: Path = DATA_ROOT.joinpath("biseq")
FUNC_PATH: Path = BISEQ_ROOT.joinpath("functional")
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
        for test, control in [(("met", "high"), ("met", "low"))]
    ]
)
CONTRASTS_LEVELS_COLORS: Dict[str, str] = {
    "met": "#8B3A3A",
    # same colors but for GOI levels
    "high": "#EE3B3B",
    "low": "#8B2323",
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
GENE_ANNOTS: Iterable[str] = (
    f"{GENOME}_genes_1to5kb",
    f"{GENOME}_genes_exons",
    f"{GENOME}_genes_introns",
    f"{GENOME}_genes_promoters",
)
N_processes: int = 4
FDR_THS: Iterable[float] = (0.05, 0.01)
MEAN_DIFF_LEVELS: Iterable[str] = ("hyper", "hypo", "all")
MEAN_DIFF_THS: Iterable[float] = (5, 10, 20)
SPECIES: str = "Homo sapiens"
PARALLEL: bool = True

annot_df = pd.read_csv(ANNOT_PATH, index_col=0)
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

    exp_prefix = (
        f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
        f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
    )

    for gene_annot in GENE_ANNOTS:
        exp_name = (
            f"{exp_prefix}_dmr_{contrast_level_test}_vs_{contrast_level_control}_"
            f"ann_{gene_annot}"
        )
        results_file = BISEQ_ROOT.joinpath(f"{exp_name}.csv")

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
                numeric_col="meth.diff",
                analysis_type="gsea",
            )
        )

        # 2.2. Add ORA inputs
        for fdr_th, mean_diff_level, mean_meth_diff_th in product(
            FDR_THS, MEAN_DIFF_LEVELS, MEAN_DIFF_THS
        ):
            fdr_th_str = str(fdr_th).replace(".", "_")
            mean_diff_level_str = str(mean_diff_level).replace(".", "_")
            mean_diff_str = str(mean_meth_diff_th).replace(".", "_")
            input_collection.append(
                dict(
                    data_type="diff_meth",
                    func_path=FUNC_PATH,
                    plots_path=PLOTS_PATH,
                    results_file=results_file,
                    exp_name=(
                        f"{exp_prefix}_dmr_"
                        f"{contrast_level_test}_vs_{contrast_level_control}_"
                        f"{mean_diff_level}_fdr_{fdr_th_str}_diff_{mean_diff_str}_"
                        f"ann_{gene_annot}"
                    ),
                    org_db=org_db,
                    cspa_surfaceome_file=STORAGE.joinpath(
                        "CSPA_validated_surfaceome_proteins_human.csv"
                    ),
                    p_col="qvalue",
                    p_th=fdr_th,
                    lfc_col="meth.diff",
                    lfc_level=mean_diff_level,
                    lfc_th=mean_meth_diff_th,
                    numeric_col="meth.diff",
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
            processes=user_args["processes"] // 3,
        )
    else:
        for ins in tqdm(input_collection):
            functional_enrichment(**ins)
