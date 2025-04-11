import gc
import json
import logging
from collections import defaultdict
from datetime import datetime as dt
from itertools import product
from pathlib import Path
from typing import Callable

import pandas as pd
import rpy2.robjects as ro

from components.functional_analysis.do import run_do_gsea, run_do_ora
from components.functional_analysis.go import run_go_gsea, run_go_ora
from components.functional_analysis.gprofiler2 import (
    run_gprofiler2_gsea,
    run_gprofiler2_ora,
)
from components.functional_analysis.kegg import run_kegg_gsea, run_kegg_ora
from components.functional_analysis.mkegg import run_mkegg_gsea, run_mkegg_ora
from components.functional_analysis.msigdb import run_msigdb_gsea, run_msigdb_ora
from components.functional_analysis.ncg import run_ncg_gsea, run_ncg_ora
from components.functional_analysis.orgdb import OrgDB
from components.functional_analysis.reactome import run_reactome_gsea, run_reactome_ora
from r_wrappers.msigdb import get_msigdbr, get_t2g
from r_wrappers.utils import map_gene_id, prepare_gene_list


def generate_gene_lists(
    degs_df: pd.DataFrame,
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_col: str = "log2FoldChange",
    lfc_th: float = 2.0,
    numeric_col: str = "log2FoldChange",
    from_type: str = "ENSEMBL",
):
    gene_lists = defaultdict(dict)

    for gene_id in ("ENTREZID", "SYMBOL"):
        gene_lists[gene_id]["filtered_genes"] = prepare_gene_list(
            genes=degs_df,
            from_type=from_type,
            to_type=gene_id,
            numeric_col=numeric_col,
        )
        gene_lists[gene_id]["degs_filtered"] = prepare_gene_list(
            genes=degs_df,
            from_type=from_type,
            to_type=gene_id,
            p_col=p_col,
            p_th=p_th,
            lfc_col=lfc_col,
            lfc_th=lfc_th,
            numeric_col=numeric_col,
        )

    return gene_lists


def run_all_ora(exp_name: str, get_func_input: Callable, cspa_surfaceome_file: Path):
    """
    Run all functional analysis functions.

    Args:
        exp_name: String name of the experiment, example format:
            {exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}
        get_func_input: Callable function that returns formatted inputs for functional
            enrichment analysis.
        cspa_surfaceome_file: File containing CSPA surface proteins annotation.
    """

    enrich_params = get_func_input("", "")

    if (
        enrich_params["background_genes"] is None
        or len(enrich_params["background_genes"]) == 0
    ):
        logging.error(
            f"[{dt.now()}][{exp_name}]: Background genes cannot be empty, returning"
        )
        return

    if (
        enrich_params["filtered_genes"] is None
        or len(enrich_params["filtered_genes"]) == 0
    ):
        logging.error(
            f"[{dt.now()}][{exp_name}]: No DEGs under specified thresholds, returning."
        )
        return

    with (
        enrich_params["files_prefix"]
        .parent.joinpath(f"{exp_name}_ora_params.json")
        .open("w") as fp
    ):
        json.dump(
            {
                "background_genes": list(enrich_params["background_genes"].names),
                "filtered_genes": list(enrich_params["filtered_genes"].names),
                "species": enrich_params["org_db"].species,
                "files_prefix": str(enrich_params["files_prefix"]),
                "plots_prefix": str(enrich_params["plots_prefix"]),
            },
            fp,
            indent=4,
        )

    ####################################################################################
    # 1. KEGG Pathways
    logging.info(f"[{dt.now()}][{exp_name}]: Processing KEGG pathways...")
    run_kegg_ora(**get_func_input("KEGG", "ora"))
    gc.collect()

    ####################################################################################
    # 2. KEGG Pathways Modules
    logging.info(f"[{dt.now()}][{exp_name}]: Processing MKEGG Pathways modules...")
    run_mkegg_ora(**get_func_input("MKEGG", "ora"))
    gc.collect()

    ####################################################################################
    # 3. Gene Ontology (GO)
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Gene Ontology...")
    onts = ("MF", "BP", "CC")

    inputs = [
        (
            inputs["background_genes"],
            inputs["org_db"],
            inputs["filtered_genes"],
            Path(f"{inputs['files_prefix']}_{ont}"),
            Path(f"{inputs['plots_prefix']}_{ont}"),
            ont,
        )
        for inputs, ont in product([get_func_input("GO", "ora")], onts)
    ]
    for args in inputs:
        run_go_ora(*args)
    gc.collect()

    ####################################################################################
    # 4. Disease Ontology (DO)
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Disease Ontology...")
    run_do_ora(**get_func_input("DO", "ora"))
    gc.collect()

    ####################################################################################
    # 5. Network of Cancer Genes (NCG)
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Network of Cancer Genes...")
    run_ncg_ora(**get_func_input("NCG", "ora"))
    gc.collect()

    ####################################################################################
    # 6. Molecular Signatures Database (MSigDB)
    logging.info(
        f"[{dt.now()}][{exp_name}]: Processing Molecular Signatures Database"
        " (MSigDB)..."
    )

    # category-description of gene sets
    categories = {
        "H": "hallmark gene sets",
        "C1": "positional gene sets",
        "C2": "curated gene sets",
        "C3": "motif gene sets",
        "C4": "computational gene sets",
        "C5": "GO gene sets",
        "C6": "oncogenic signatures",
        "C7": "immunologic signatures",
        "C8": "cell type signatures",
    }
    mesigdbr_cat = {cat: get_msigdbr(category=cat) for cat in categories.keys()}

    inputs = [
        (
            inputs["background_genes"],
            inputs["org_db"],
            inputs["filtered_genes"],
            Path(f"{inputs['files_prefix']}_{cat}"),
            Path(f"{inputs['plots_prefix']}_{cat}"),
            get_t2g(mesigdbr_cat[cat], gene_id_col="entrez_gene"),
        )
        for inputs, cat in product([get_func_input("MSIGDB", "ora")], categories.keys())
    ]

    for args in inputs:
        run_msigdb_ora(*args)
    gc.collect()

    ####################################################################################
    # 7. Reactome
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Reactome...")
    run_reactome_ora(**get_func_input("REACTOME", "ora"))
    gc.collect()

    ####################################################################################
    # 8. Gprofiler2
    logging.info(f"[{dt.now()}][{exp_name}]: Processing GPROFILER2...")
    run_gprofiler2_ora(**get_func_input("GPROFILER2", "ora"))
    gc.collect()

    ####################################################################################
    # 9. Cell surface protein receptors
    # Source: http://wlab.ethz.ch/cspa/#downloads
    # Download http://wlab.ethz.ch/cspa/data/S2_File.xlsx, and manually
    # convert to .csv. First page is human, second one is mouse,
    # convert them separately.
    logging.info(
        f"[{dt.now()}][{exp_name}]: Processing Cell surface protein receptors..."
    )
    cspa_surfaceome = pd.read_csv(cspa_surfaceome_file, index_col=0)

    # 9.1. Annotate DEGs with corresponding surface proteins
    enrich_params["files_prefix"].parent.joinpath("CSPA_SURFACEOME").mkdir(
        exist_ok=True, parents=True
    )

    deg_cspa_df = (
        map_gene_id(
            list(enrich_params["filtered_genes"].names),
            org_db=get_func_input("", "")["org_db"],
            from_type="ENTREZID",
            to_type="SYMBOL",
        )
        .to_frame()
        .set_index("SYMBOL")
        .join(cspa_surfaceome.set_index("ENTREZ gene symbol"))
        .dropna()
    )

    # 9.2. Save results
    deg_cspa_df.to_csv(
        enrich_params["files_prefix"]
        .parent.joinpath("CSPA_SURFACEOME")
        .joinpath(f"{exp_name}_cspa_surfaceome.csv")
    )


def run_all_ora_simple(
    exp_name: str,
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    func_path: Path,
    plots_path: Path,
    cspa_surfaceome_file: Path,
):
    """
    Wrapper for `run_all_ora` when the `get_func_input` argument cannot be pickled.
    """

    def get_func_input(db_type: str, analysis_type: str):
        return dict(
            background_genes=background_genes,
            org_db=org_db,
            filtered_genes=filtered_genes,
            files_prefix=func_path.joinpath(db_type).joinpath(
                f"{exp_name}_{analysis_type}"
            ),
            plots_prefix=plots_path.joinpath(db_type).joinpath(
                f"{exp_name}_{analysis_type}"
            ),
        )

    run_all_ora(exp_name, get_func_input, cspa_surfaceome_file)


def run_all_gsea(exp_name: str, get_func_input: Callable):
    """
    Run all functional analysis functions.

    Args:
        exp_name: String name of the experiment, example format:
            {exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}
        get_func_input: Callable function that returns formatted inputs for functional
            enrichment analysis.
    """
    enrich_params = get_func_input("", "")

    if (
        enrich_params["background_genes"] is None
        or len(enrich_params["background_genes"]) == 0
    ):
        logging.error(
            f"[{dt.now()}][{exp_name}]: Background genes cannot be 0, returning"
        )
        return

    with (
        enrich_params["files_prefix"]
        .parent.joinpath(f"{exp_name}_gsea_params.json")
        .open("w") as fp
    ):
        json.dump(
            {
                "background_genes": list(enrich_params["background_genes"].names),
                "filtered_genes": None,
                "species": enrich_params["org_db"].species,
                "files_prefix": str(enrich_params["files_prefix"]),
                "plots_prefix": str(enrich_params["plots_prefix"]),
            },
            fp,
            indent=4,
        )

    ####################################################################################
    # 1. KEGG Pathways
    logging.info(f"[{dt.now()}][{exp_name}]: Processing KEGG pathways...")
    run_kegg_gsea(**get_func_input("KEGG", "gsea"))
    gc.collect()

    ####################################################################################
    # 2. KEGG Pathways Modules
    logging.info(f"[{dt.now()}][{exp_name}]: Processing MKEGG Pathways modules...")
    run_mkegg_gsea(**get_func_input("MKEGG", "gsea"))
    gc.collect()

    ####################################################################################
    # 3. Gene Ontology (GO)
    logging.info("Processing Gene Ontology:")
    onts = ["MF", "BP", "CC", "ALL"]

    inputs = [
        (
            inputs["background_genes"],
            inputs["org_db"],
            inputs["filtered_genes"],
            Path(f"{inputs['files_prefix']}_{ont}"),
            Path(f"{inputs['plots_prefix']}_{ont}"),
            ont,
        )
        for inputs, ont in product([get_func_input("GO", "gsea")], onts)
    ]
    for args in inputs:
        run_go_gsea(*args)
    gc.collect()

    ####################################################################################
    # 4. Disease Ontology (DO)
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Disease Ontology...")
    run_do_gsea(**get_func_input("DO", "gsea"))
    gc.collect()

    ####################################################################################
    # 5. Network of Cancer Genes (NCG)
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Network of Cancer Genes...")
    run_ncg_gsea(**get_func_input("NCG", "gsea"))
    gc.collect()

    ####################################################################################
    # 6. Molecular Signatures Database (MSigDB)
    logging.info(
        f"[{dt.now()}][{exp_name}]: Processing Molecular Signatures Database"
        " (MSigDB)..."
    )

    # category-description of gene sets
    categories = {
        "H": "hallmark gene sets",
        "C1": "positional gene sets",
        "C2": "curated gene sets",
        "C3": "motif gene sets",
        "C4": "computational gene sets",
        "C5": "GO gene sets",
        "C6": "oncogenic signatures",
        "C7": "immunologic signatures",
        "C8": "cell type signatures",
    }
    mesigdbr_cat = {cat: get_msigdbr(category=cat) for cat in categories.keys()}

    inputs = [
        (
            inputs["background_genes"],
            inputs["org_db"],
            inputs["filtered_genes"],
            Path(f"{inputs['files_prefix']}_{cat}"),
            Path(f"{inputs['plots_prefix']}_{cat}"),
            get_t2g(mesigdbr_cat[cat], gene_id_col="entrez_gene"),
        )
        for inputs, cat in product(
            [get_func_input("MSIGDB", "gsea")], categories.keys()
        )
    ]

    for args in inputs:
        run_msigdb_gsea(*args)
    gc.collect()

    ####################################################################################
    # 7. Reactome
    logging.info(f"[{dt.now()}][{exp_name}]: Processing Reactome...")
    run_reactome_gsea(**get_func_input("REACTOME", "gsea"))
    gc.collect()

    ####################################################################################
    # 8. Gprofiler2
    logging.info(f"[{dt.now()}][{exp_name}]: Processing GPROFILER2...")
    run_gprofiler2_gsea(**get_func_input("GPROFILER2", "gsea"))
    gc.collect()
