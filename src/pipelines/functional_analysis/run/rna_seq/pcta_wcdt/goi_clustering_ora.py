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
import rpy2.robjects as ro
from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from components.functional_analysis.utils import run_all_ora_simple
from data.utils import parallelize_map
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
GOI_ENSEMBL: str = "ENSG00000086205"  # ENSEMBL ID for FOLH1 (PSMA)
SPECIES: str = "Homo sapiens"
org_db = OrgDB(SPECIES)
GOI_SYMBOL = map_gene_id([GOI_ENSEMBL], org_db, "ENSEMBL", "SYMBOL")[0]
DATA_ROOT: Path = STORAGE.joinpath(f"PCTA_WCDT_{GOI_SYMBOL}")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
DESEQ_PATH: Path = DATA_ROOT.joinpath("deseq2")
PPI_NETWORK_PATH: Path = DATA_ROOT.joinpath("degss_ppi_networks")
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
            {SAMPLE_CONTRAST_FACTOR: ["prim"], GOI_LEVEL_PREFIX: ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["prim"], GOI_LEVEL_PREFIX: ["low"]},
        ),
        (
            {SAMPLE_CONTRAST_FACTOR: ["met"], GOI_LEVEL_PREFIX: ["high"]},
            {SAMPLE_CONTRAST_FACTOR: ["met"], GOI_LEVEL_PREFIX: ["low"]},
        ),
    ),
}
PQ_PAIRS: Iterable[Tuple[float, float]] = (
    (1.0, 0.5),
    (1.0, 0.1),
    (0.5, 1.0),
    (0.1, 1.0),
)
P_COLS: Iterable[str] = ["padj"]
P_THS: Iterable[float] = (0.05,)
LFC_LEVELS: Iterable[str] = ("up", "down")
LFC_THS: Iterable[float] = (0.0,)
MSIGDB_CATS: Iterable[str] = ("H", *[f"C{i}" for i in range(1, 9)])
INTERACTION_SCORES: Iterable[int] = (500, 700)
NETWORK_TYPES: Iterable[str] = ("functional",)
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
        exp_prefix = (
            f"{SAMPLE_CONTRAST_FACTOR}_{sample_types_str}_"
            f"{GOI_LEVEL_PREFIX}_{'+'.join(sorted(contrasts_levels))}_"
        )
        # 1.4. Set experiment name
        exp_name = f"{exp_prefix}_{contrast_level_test}_vs_{contrast_level_control}"
        contrast_name_prefix = f"{contrast_level_test}_vs_{contrast_level_control}"

        # 2. Generate input collection for all arguments' combinations
        for (
            p_col,
            p_th,
            lfc_level,
            lfc_th,
            (p, q),
            interaction_score,
            network_type,
        ) in product(
            P_COLS,
            P_THS,
            LFC_LEVELS,
            LFC_THS,
            PQ_PAIRS,
            INTERACTION_SCORES,
            NETWORK_TYPES,
        ):
            # 2.1. Setup
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")
            p_str = str(p).replace(".", "_")
            q_str = str(q).replace(".", "_")
            root_path = (
                PPI_NETWORK_PATH.joinpath(
                    f"{contrast_name_prefix}_degss_"
                    f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                    f"gsea_{'_'.join(MSIGDB_CATS)}_union"
                )
                .joinpath(f"{network_type}_network_at_{interaction_score}_score")
                .joinpath("clustering")
                .joinpath(f"p_{p_str}_q_{q_str}")
            )
            func_path = root_path.joinpath("functional")
            plots_path = func_path.joinpath("plots")
            plots_path.mkdir(parents=True, exist_ok=True)

            # 2.2. Load gene clusters metadata
            protein_cluster_labels = pd.read_csv(
                root_path.joinpath("genes_cluster_ids_metadata.csv"), index_col=0
            ).sort_values("cluster_id", ascending=True)

            # 2.3. Determine background genes as all genes tested for diff. expr.
            deseq2_df = pd.read_csv(
                DESEQ_PATH.joinpath(f"{exp_name}_deseq_results_unique.csv"), index_col=0
            )
            genes_map = (
                deseq2_df.dropna()
                .set_index("SYMBOL")
                .sort_index()["ENTREZID"]
                .astype(str)
            )

            background_genes = ro.FloatVector(list(range(len(genes_map))))
            background_genes.names = ro.StrVector(genes_map.tolist())

            for cluster_id, cluster_genes in protein_cluster_labels.groupby(
                "cluster_id"
            ):
                common_ids = genes_map.index.intersection(cluster_genes.index)
                filtered_genes = ro.FloatVector(list(range(len(common_ids))))
                filtered_genes.names = ro.StrVector(genes_map[common_ids].tolist())

                input_collection.append(
                    dict(
                        exp_name=cluster_id,
                        background_genes=background_genes,
                        org_db=org_db,
                        filtered_genes=filtered_genes,
                        func_path=func_path,
                        plots_path=plots_path,
                        cspa_surfaceome_file=STORAGE.joinpath(
                            "CSPA_validated_surfaceome_proteins_human.csv"
                        ),
                    )
                )


# 4. Run functional enrichment analysis
if __name__ == "__main__":
    freeze_support()
    if PARALLEL and len(input_collection) > 1:
        parallelize_map(
            functools.partial(run_func_dict, func=run_all_ora_simple),
            input_collection,
            threads=user_args["threads"] // 3,
        )
    else:
        for ins in tqdm(input_collection):
            run_all_ora_simple(**ins)
