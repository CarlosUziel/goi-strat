from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro

from components.functional_analysis.orgdb import OrgDB
from components.functional_analysis.utils import run_all_gsea, run_all_ora
from r_wrappers.utils import prepare_gene_list


def prepare_gene_lists(
    data_type: str,
    results_file: Path,
    org_db: OrgDB,
    filtered_results_file: Path = None,
    p_col: str = None,
    p_th: float = None,
    lfc_col: str = "log2FoldChange",
    lfc_level: str = None,
    lfc_th: float = None,
    numeric_col: str = "log2FoldChange",
    shap_th: float = None,
    analysis_type: str = "gsea",
) -> Tuple[ro.FloatVector, ro.FloatVector]:
    """
    Prepare gene lists for functional enrichment analysis. One gene list containing
        background genes, and another containing genes filtered by some criteria, such
        as log 2 fold change.

    Args:
        data_type: Keyword representing source of the data.
        results_file: Results file containing genes to analyse.
        org_db: Organism annotation database.
        filtered_results_file: A file containing already-filtered genes.
        p_col: P-value column to be used as filter (e.g., pvalue, padj)
        p_th: P-value filter threshold.
        lfc_level: Keep only genes of that log2FoldChange level (all, up-regulated,
            down-regulated)
        lfc_th: Log2FoldChange threshold.
        numeric_col: Which column name used to rank genes.
        shap_th: SHAP value threshold, only for ML results.
        analysis_type: Either "ora" (for ORA) or "gsea" (for GSEA).
    """
    filtered_genes = None
    if data_type == "diff_expr":
        deseq_results_df = pd.read_csv(
            results_file,
            index_col=0,
        )

        background_genes = prepare_gene_list(
            genes=deseq_results_df,
            org_db=org_db,
            from_type="ENSEMBL",
            to_type="ENTREZID",
            numeric_col=numeric_col,
        )

        if analysis_type == "ora":
            filtered_genes = prepare_gene_list(
                genes=deseq_results_df,
                org_db=org_db,
                from_type="ENSEMBL",
                to_type="ENTREZID",
                p_col=p_col,
                p_th=p_th,
                lfc_col=lfc_col,
                lfc_level=lfc_level,
                lfc_th=lfc_th,
                numeric_col=numeric_col,
            )

    elif data_type == "diff_meth":
        diff_meth_genes = (
            pd.read_csv(results_file, index_col=0)
            .replace("NA_character_", np.nan)
            .dropna(subset=["annot.gene_id"])
            .groupby("annot.gene_id")
            .mean(numeric_only=True)
        )
        diff_meth_genes.index = diff_meth_genes.index.astype(int).astype(str)

        background_genes = prepare_gene_list(
            genes=diff_meth_genes,
            numeric_col=numeric_col,
        )

        if analysis_type == "ora":
            filtered_genes = prepare_gene_list(
                genes=diff_meth_genes,
                p_col=p_col,
                p_th=p_th,
                lfc_col=lfc_col,
                lfc_level=lfc_level,
                lfc_th=lfc_th,
                numeric_col=numeric_col,
            )

    elif data_type in ("diff_expr_ml", "diff_meth_ml"):
        if data_type == "diff_expr_ml":
            inputs_df = pd.read_csv(
                results_file,
                index_col=0,
            ).sort_values("shap_value", ascending=False)
        else:
            genes_shap = (
                pd.read_csv(
                    results_file,
                    index_col=0,
                )
                .replace("NA_character_", np.nan)
                .dropna(subset=["annot.gene_id"])
                .groupby("annot.gene_id")["shap_value"]
                .mean()
                .sort_values(ascending=False)
            )

        background_genes = ro.FloatVector(genes_shap.tolist())
        background_genes.names = inputs_df.index.tolist()

        if analysis_type == "ora":
            inputs_df_filtered = genes_shap[genes_shap > shap_th]
            filtered_genes = ro.FloatVector(inputs_df_filtered.tolist())
            filtered_genes.names = inputs_df_filtered.index.tolist()

    elif data_type in ("diff_expr_wgcna", "diff_meth_wgcna"):
        if data_type == "diff_expr_wgcna":
            inputs_df = pd.read_csv(
                results_file,
                index_col=0,
            ).sort_values(["ClusterCoef", "Connectivity"], ascending=False)
        else:
            inputs_df = (
                pd.read_csv(
                    results_file,
                    index_col=0,
                )
                .replace("NA_character_", np.nan)
                .dropna(subset=["annot.gene_id"])
                .groupby("annot.gene_id")[["ClusterCoef", "Connectivity"]]
                .mean()
                .sort_values(["ClusterCoef", "Connectivity"], ascending=False)
            )

        background_genes = ro.FloatVector(inputs_df["ClusterCoef"].tolist())
        background_genes.names = inputs_df.index.tolist()

        if analysis_type == "ora":
            if data_type == "diff_expr_wgcna":
                module_members = pd.read_csv(
                    filtered_results_file,
                    index_col=0,
                ).sort_values(["ClusterCoef", "Connectivity"], ascending=False)
            else:
                module_members = (
                    pd.read_csv(
                        filtered_results_file,
                        index_col=0,
                    )
                    .replace("NA_character_", np.nan)
                    .dropna(subset=["annot.gene_id"])
                    .groupby("annot.gene_id")[["ClusterCoef", "Connectivity"]]
                    .mean()
                    .sort_values(["ClusterCoef", "Connectivity"], ascending=False)
                )

            filtered_genes = ro.FloatVector(
                inputs_df.loc[module_members.index, "ClusterCoef"].tolist()
            )
            filtered_genes.names = module_members.index.tolist()

    return background_genes, filtered_genes


def functional_enrichment(
    data_type: str,
    func_path: Path,
    plots_path: Path,
    results_file: Path,
    exp_name: str,
    org_db: OrgDB,
    filtered_results_file: Path = None,
    p_col: str = None,
    p_th: float = None,
    lfc_col: str = "log2FoldChange",
    lfc_level: str = None,
    lfc_th: float = None,
    numeric_col: str = "log2FoldChange",
    shap_th: float = None,
    analysis_type: str = "gsea",
    cspa_surfaceome_file: Path = None,
) -> None:
    """
    Run all functional analysis functions for differential expression results.

    Args:
        data_type: Keyword representing source of the data.
        func_path: Where to store functional results.
        plots_path: Where to store functional plots.
        results_file: Results file containing genes to analyse.
        exp_name: Name of the experiment.
        org_db: Organism annotation database.
        filtered_results_file: A file containing already-filtered genes.
        p_col: P-value column to be used as filter (e.g., pvalue, padj)
        p_th: P-value filter threshold.
        lfc_level: Keep only genes of that log2FoldChange level (all, up-regulated,
            down-regulated)
        lfc_th: Log2FoldChange threshold.
        numeric_col: Which column name used to rank genes.
        shap_th: SHAP value threshold, only for ML results.
        analysis_type: Either "ora" (for ORA) or "gsea" (for GSEA)
        cspa_surfaceome_file: File containing CSPA surface proteins annotation.
    """
    assert analysis_type in ["ora", "gsea"], "analysis_type can only be 'ora' or 'gsea'"

    ####################################################################################
    # 1. Prepare gene lists
    background_genes, filtered_genes = prepare_gene_lists(
        data_type=data_type,
        results_file=results_file,
        org_db=org_db,
        filtered_results_file=filtered_results_file,
        p_col=p_col,
        p_th=p_th,
        lfc_col=lfc_col,
        lfc_level=lfc_level,
        lfc_th=lfc_th,
        numeric_col=numeric_col,
        shap_th=shap_th,
        analysis_type=analysis_type,
    )

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

    ####################################################################################
    # 2. Run all functional enrichment analysis functions
    if analysis_type == "ora":
        run_all_ora(
            exp_name,
            get_func_input,
            cspa_surfaceome_file,
        )
    else:
        run_all_gsea(exp_name, get_func_input)
