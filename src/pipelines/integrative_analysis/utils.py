import json
import logging
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import PositiveFloat
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from upsetplot import UpSet, from_contents

from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation


def intersect_degs(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    genes_id: str = "ENTREZID",
) -> None:
    """
    Compute all possible intersecting sets of DEGs between a given list of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
    """
    # 0. Setup
    deseq2_path = root_path.joinpath("deseq2")
    save_path = root_path.joinpath("integrative_analysis").joinpath("intersecting_degs")
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
    ).replace(" ", "_")

    # 1. Get DEGs IDs sets for each comparison
    degs_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_dfs[contrast] = pd.read_csv(
                deseq2_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                    "_deseq_results_unique.csv"
                ),
                index_col=0,
                dtype={"ENTREZID": str},
            ).dropna(subset=[genes_id])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degs_dfs[contrast] = pd.DataFrame(columns=[genes_id])

    if all([df.empty for df in degs_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All deseq2 results files were empty. "
            "No intersection possible."
        )
        return

    degs_intersections = from_contents(
        {
            contrast: set(degs_df[genes_id].values)
            for contrast, degs_df in degs_dfs.items()
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degs_intersections.loc[tuple([True] * degs_intersections.index.nlevels)]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Annotate DEGs intersection dataframe and save to disk
    degs_intersections.reset_index().set_index("id").rename_axis(genes_id).join(
        pd.concat([df.set_index(genes_id) for df in degs_dfs.values()]).drop(
            columns=["baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"]
        )
    ).reset_index().drop_duplicates(subset=[genes_id]).astype(
        {"ENTREZID": int}
    ).set_index(
        genes_id
    ).sort_values(
        [*degs_intersections.index.names, genes_id],
        ascending=[False] * len(degs_intersections.index.names) + [True],
    ).to_csv(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs_{genes_id}_{n_all_common}.csv"
        )
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degs_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
    ).plot(fig=fig)
    plt.suptitle(
        f"Intersecting {genes_id} DEGs \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th})",
    )
    plt.savefig(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs"
            f"_{genes_id}_{n_all_common}_upsetplot.pdf"
        )
    )
    plt.close()


def intersect_degs_external(
    contrast_prefixes: Dict[str, str],
    external_genes: Dict[str, Set[str]],
    root_path: Path,
    external_annot_df: Optional[pd.DataFrame] = None,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    genes_id: str = "ENTREZID",
) -> None:
    """
    Compute all possible intersecting sets of DEGs between a given list of contrasts, as
    well as external user-provided lists gene IDs.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        external_genes: A dictionary with keys being gene list names and
            values being `genes_id` genes.
        root_path: Root path of the RNASeq analysis.
        external_annot_df: Additional annotation for external DEGs.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
    """
    # 0. Setup
    deseq2_path = root_path.joinpath("deseq2")
    save_path = root_path.joinpath("integrative_analysis").joinpath(
        "intersecting_degs_external"
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
    ).replace(" ", "_")

    # 1. Get DEGs sets for each comparison
    degs_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_dfs[contrast] = pd.read_csv(
                deseq2_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                    "_deseq_results_unique.csv"
                ),
                index_col=0,
                dtype={"ENTREZID": str},
            ).dropna(subset=[genes_id])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degs_dfs[contrast] = pd.DataFrame(columns=[genes_id])

    if all([df.empty for df in degs_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All deseq2 results files were empty. "
            "No intersection possible."
        )
        return

    degs_intersections = from_contents(
        {
            **{
                contrast: set(degs_df[genes_id].values)
                for contrast, degs_df in degs_dfs.items()
            },
            **external_genes,
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degs_intersections.loc[tuple([True] * degs_intersections.index.nlevels)]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Save DEGs intersection dataframe to disk
    degs_intersections.reset_index().set_index("id").rename_axis(genes_id).join(
        pd.concat(
            [df.set_index(genes_id) for df in degs_dfs.values()]
            + [external_annot_df.set_index(genes_id)]
        ).drop(columns=["baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"])
    ).reset_index().drop_duplicates(subset=[genes_id]).astype(
        {"ENTREZID": int}
    ).set_index(
        genes_id
    ).sort_values(
        [*degs_intersections.index.names, genes_id],
        ascending=[False] * len(degs_intersections.index.names) + [True],
    ).to_csv(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs_{genes_id}_{n_all_common}.csv"
        )
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degs_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
    ).plot(fig=fig)
    plt.suptitle(
        f"Intersecting {genes_id} DEGs \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th})",
    )

    plt.savefig(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs"
            f"_{genes_id}_{n_all_common}_upsetplot.pdf"
        )
    )
    plt.close()


def intersect_degs_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
    genes_id: str = "ENTREZID",
) -> None:
    """
    Compute all possible intersecting sets of DEGs (filtered by SHAP values) between
    a given list of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degs_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"shap_values_{shap_th_str}"
    ).replace(" ", "_")

    # 1. Get DEGs (filtered by SHAP values) sets for each comparison
    degs_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                dtype={"ENTREZID": str},
            ).dropna(subset=[genes_id])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degs_shap_dfs[contrast] = pd.DataFrame(columns=[genes_id])

    if all([df.empty for df in degs_shap_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All bootstrap results files were empty. "
            "No intersection possible."
        )
        return

    degs_shap_intersections = from_contents(
        {
            contrast: set(degs_shap_df[genes_id].values)
            for contrast, degs_shap_df in degs_shap_dfs.items()
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degs_shap_intersections.loc[
                tuple([True] * degs_shap_intersections.index.nlevels)
            ]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Save shap DEGs intersection dataframe to disk
    degs_shap_intersections.reset_index().set_index("id").rename_axis(genes_id).join(
        pd.concat([df.set_index(genes_id) for df in degs_shap_dfs.values()]).drop(
            columns=[
                "baseMean",
                "log2FoldChange",
                "lfcSE",
                "pvalue",
                "padj",
                "shap_value",
            ]
        )
    ).reset_index().drop_duplicates(subset=[genes_id]).astype(
        {"ENTREZID": int}
    ).set_index(
        genes_id
    ).sort_values(
        [*degs_shap_intersections.index.names, genes_id],
        ascending=[False] * len(degs_shap_intersections.index.names) + [True],
    ).to_csv(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs_shap_{genes_id}_{n_all_common}.csv"
        )
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degs_shap_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
    ).plot(fig=fig)
    plt.suptitle(
        f"Intersecting {genes_id} DEGs \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th}, SHAP > {shap_th})",
    )

    plt.savefig(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs_shap"
            f"_{genes_id}_{n_all_common}_upsetplot.pdf"
        )
    )
    plt.close()


def intersect_degs_shap_external(
    contrast_prefixes: Dict[str, str],
    external_genes: Dict[str, Set[str]],
    root_path: Path,
    external_annot_df: Optional[pd.DataFrame] = None,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
    genes_id: str = "ENTREZID",
) -> None:
    """
    Compute all possible intersecting sets of DEGs (filtered by SHAP values) between
    a given list of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        external_genes: A dictionary with keys being gene list names and
            values being `genes_id` genes.
        root_path: Root path of the RNASeq analysis.
        external_annot_df: Additional annotation for external DEGs.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degs_shap_external")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"shap_values_{shap_th_str}"
    ).replace(" ", "_")

    # 1. Get DEGs (filtered by SHAP values) sets for each comparison
    degs_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                dtype={"ENTREZID": str},
            ).dropna(subset=[genes_id])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degs_shap_dfs[contrast] = pd.DataFrame(columns=[genes_id])

    if all([df.empty for df in degs_shap_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All bootstrap results files were empty. "
            "No intersection possible."
        )
        return

    degs_shap_intersections = from_contents(
        {
            **{
                contrast: set(degs_shap_df[genes_id].values)
                for contrast, degs_shap_df in degs_shap_dfs.items()
            },
            **external_genes,
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degs_shap_intersections.loc[
                tuple([True] * degs_shap_intersections.index.nlevels)
            ]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Save shap DEGs intersection dataframe to disk
    degs_shap_intersections.reset_index().set_index("id").rename_axis(genes_id).join(
        pd.concat([df.set_index(genes_id) for df in degs_shap_dfs.values()]).drop(
            columns=[
                "baseMean",
                "log2FoldChange",
                "lfcSE",
                "pvalue",
                "padj",
                "shap_value",
            ]
        )
    ).combine_first(
        external_annot_df.set_index(genes_id)
    ).reset_index().drop_duplicates(
        subset=[genes_id]
    ).astype(
        {"ENTREZID": int}
    ).sort_values(
        [*degs_shap_intersections.index.names, genes_id],
        ascending=[False] * len(degs_shap_intersections.index.names) + [True],
    ).set_index(
        [genes_id, *degs_shap_intersections.index.names]
    ).to_csv(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs_shap_{genes_id}_{n_all_common}.csv"
        )
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degs_shap_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
    ).plot(fig=fig)
    plt.suptitle(
        f"Intersecting {genes_id} DEGs \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th}, SHAP > {shap_th})",
    )

    plt.savefig(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_degs_shap"
            f"_{genes_id}_{n_all_common}_upsetplot.pdf"
        )
    )
    plt.close()


def intersect_pathways(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: Optional[str] = None,
    p_th: Optional[float] = None,
    lfc_level: Optional[str] = None,
    lfc_th: Optional[float] = None,
    func_db: str = "KEGG",
    analysis_type: str = "gsea",
) -> None:
    """
    Compute all possible intersecting sets of enriched pathways between a given list
    of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        func_db: Functional database the pathways are enriched in.
        analysis_type: Type of enrichment analysis that was performed, either "ora"
            or "gsea".
    """
    # 0. Setup
    func_path = root_path.joinpath("functional")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_pathways")
        .joinpath(func_db)
    )
    save_path.mkdir(exist_ok=True, parents=True)

    assert (analysis_type == "gsea") or (
        (
            analysis_type == "ora"
            and p_col is not None
            and p_th is not None
            and lfc_level is not None
            and lfc_th is not None
        )
    )

    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    filters_str = (
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        if analysis_type == "ora"
        else ""
    )
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{filters_str}{analysis_type}"
    ).replace(" ", "_")

    # 1. Get pathways sets for each comparison
    pathways_dfs = {}
    func_suffix = f"_{func_db.split('_')[1]}" if "GO" in func_db else ""
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            pathways_dfs[contrast] = pd.read_csv(
                func_path.joinpath(func_db.split("_")[0]).joinpath(
                    (
                        f"{contrast_prefix}_{p_col}_{p_th_str}_"
                        f"{lfc_level}_{lfc_th_str}_ora{func_suffix}.csv"
                        if analysis_type == "ora"
                        else f"{contrast_prefix}_gsea{func_suffix}.csv"
                    )
                )
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            pathways_dfs[contrast] = pd.DataFrame(
                columns=["Description", "GeneRatio", "geneID"]
            )

    if all([df.empty for df in pathways_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All functional results files were empty. "
            "No intersection possible."
        )
        return

    pathways_intersections = from_contents(
        {
            contrast: set(pathway_df["Description"].values)
            for contrast, pathway_df in pathways_dfs.items()
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            pathways_intersections.loc[
                tuple([True] * pathways_intersections.index.nlevels)
            ]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Get gene ratio for each pathway
    gene_ratio_metadata = pd.DataFrame(
        {
            f"{key}_GeneRatio_setSize": [
                pathways_df.set_index("Description")[
                    "GeneRatio" if "GeneRatio" in pathways_df.columns else "setSize"
                ].get(pathway, "")
                for pathway in pathways_intersections["id"].values
            ]
            for key, pathways_df in pathways_dfs.items()
        }
    ).set_index(pathways_intersections.index)

    # 1.2. Get gene content for each pathway (list of genes assigned to pathway)
    gene_IDs_metadata = pd.DataFrame(
        {
            f"{key}_geneID": [
                pathways_df.set_index("Description")[
                    "geneID" if "geneID" in pathways_df.columns else "core_enrichment"
                ].get(pathway, "")
                for pathway in pathways_intersections["id"].values
            ]
            for key, pathways_df in pathways_dfs.items()
        }
    ).set_index(pathways_intersections.index)

    # 1.3. Save pathways intersection and metadata dataframe to disk
    pd.concat(
        (pathways_intersections, gene_ratio_metadata, gene_IDs_metadata), axis=1
    ).to_csv(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_pathways_{n_all_common}.csv"
        )
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        pathways_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
    ).plot(fig=fig)
    plt.suptitle(
        "Intersecting pathways \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
        f"{p_col} < {p_th}, LFC > {lfc_th}, {func_db} {analysis_type})",
    )

    plt.savefig(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_pathways_{n_all_common}_upsetplot.pdf"
        )
    )
    plt.close()


def intersect_pathways_genes(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: Optional[str] = None,
    p_th: Optional[float] = None,
    lfc_level: Optional[str] = None,
    lfc_th: Optional[float] = None,
    func_db: str = "KEGG",
    analysis_type: str = "gsea",
) -> None:
    """
    Compute all possible intersecting sets between genes of shared enriched pathways
    among a given list of contrasts. That is, for each (not) shared pathway between
    contrasts, also compute the intersection of the genes that were (not) assigned to
    said pathway.

    Even if a pathway is shared between two contrasts, it is expected that the gene
    content differs, hence making these comparisons interesting to look at.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        func_db: Functional database the pathways are enriched in.
        analysis_type: Type of enrichment analysis that was performed, either "ora"
            or "gsea".
    """
    # 0. Setup
    func_path = root_path.joinpath("functional")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_pathways_genes")
        .joinpath(func_db)
    )
    save_path.mkdir(exist_ok=True, parents=True)

    assert (analysis_type == "gsea") or (
        (
            analysis_type == "ora"
            and p_col is not None
            and p_th is not None
            and lfc_level is not None
            and lfc_th is not None
        )
    )

    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    filters_str = (
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        if analysis_type == "ora"
        else ""
    )
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{filters_str}{analysis_type}"
    ).replace(" ", "_")
    pathways_path = save_path.joinpath(comparison_alias)
    pathways_path.mkdir(exist_ok=True, parents=True)

    # 1. Get pathways sets for each comparison
    pathways_dfs = {}
    func_suffix = f"_{func_db.split('_')[1]}" if "GO" in func_db else ""
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            pathways_dfs[contrast] = pd.read_csv(
                func_path.joinpath(func_db.split("_")[0]).joinpath(
                    (
                        f"{contrast_prefix}_{p_col}_{p_th_str}_"
                        f"{lfc_level}_{lfc_th_str}_ora{func_suffix}.csv"
                        if analysis_type == "ora"
                        else f"{contrast_prefix}_gsea{func_suffix}.csv"
                    )
                )
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            pathways_dfs[contrast] = pd.DataFrame(
                columns=["Description", "GeneRatio", "geneID"]
            )

    if all([df.empty for df in pathways_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All functional results files were empty. "
            "No intersection possible."
        )
        return

    pathways_intersections = from_contents(
        {
            contrast: set(pathway_df["Description"].values)
            for contrast, pathway_df in pathways_dfs.items()
        }
    ).sort_index(ascending=False)

    # 2. Intersect genes from different contrasts assigned to each pathway
    for pathway in pathways_intersections["id"].values:
        pathway_genes_intersections = from_contents(
            {
                f"{key}_genes": set(
                    pathways_df.set_index("Description")[
                        (
                            "geneID"
                            if "geneID" in pathways_df.columns
                            else "core_enrichment"
                        )
                    ]
                    .get(pathway, "")
                    .split("/")
                )
                for key, pathways_df in pathways_dfs.items()
            }
        ).sort_index(ascending=False)

        try:
            n_all_common = len(
                pathway_genes_intersections.loc[
                    tuple([True] * pathway_genes_intersections.index.nlevels)
                ]
            )
        except KeyError:
            n_all_common = 0

        pathway_name = (
            pathway.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
        )
        pathway_genes_intersections.to_csv(
            pathways_path.joinpath(
                f"{pathway_name}_intersecting_genes_{n_all_common}.csv"
            )
        )

        fig = plt.figure(figsize=(15, 5), dpi=300)
        UpSet(
            pathway_genes_intersections,
            subset_size="count",
            element_size=None,
            show_counts=True,
            show_percentages=True,
        ).plot(fig=fig)
        plt.suptitle(
            f"{pathway} \n("
            f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
            f"{p_col} < {p_th}, LFC > {lfc_th}, {func_db})",
        )

        plt.savefig(
            pathways_path.joinpath(
                f"{pathway_name}_intersecting_genes_{n_all_common}_upsetplot.pdf"
            )
        )
        plt.close()


def intersect_pathways_genes_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: Optional[str] = None,
    p_th: Optional[float] = None,
    lfc_level: Optional[str] = None,
    lfc_th: Optional[float] = None,
    func_db: str = "KEGG",
    analysis_type: str = "gsea",
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
) -> None:
    """
    Compute all possible intersecting sets between genes (filtered by SHAP values)
    of shared enriched pathways among a given list of contrasts. That is, for each
    (not) shared pathway between contrasts, also compute the intersection of the genes
    that were (not) assigned to said pathway.

    Even if a pathway is shared between two contrasts, it is expected that the gene
    content differs, hence making these comparisons interesting to look at.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        func_db: Functional database the pathways are enriched in.
        analysis_type: Type of enrichment analysis that was performed, either "ora"
            or "gsea".
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
    """
    # 0. Setup
    func_path = root_path.joinpath("functional")
    ml_path = root_path.joinpath("ml_classifiers")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_pathways_genes_shap")
        .joinpath(func_db)
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
    )
    save_path.mkdir(exist_ok=True, parents=True)

    assert (analysis_type == "gsea") or (
        (
            analysis_type == "ora"
            and p_col is not None
            and p_th is not None
            and lfc_level is not None
            and lfc_th is not None
        )
    )

    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    filters_str = (
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        if analysis_type == "ora"
        else ""
    )
    shap_th_str = str(shap_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{filters_str}{analysis_type}_shap_values_{shap_th_str}"
    ).replace(" ", "_")
    pathways_path = save_path.joinpath(comparison_alias)
    pathways_path.mkdir(exist_ok=True, parents=True)

    # 1. Get pathways sets for each comparison
    pathways_dfs = {}
    func_suffix = f"_{func_db.split('_')[1]}" if "GO" in func_db else ""
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            pathways_dfs[contrast] = pd.read_csv(
                func_path.joinpath(func_db.split("_")[0]).joinpath(
                    (
                        f"{contrast_prefix}_{p_col}_{p_th_str}_"
                        f"{lfc_level}_{lfc_th_str}_ora{func_suffix}.csv"
                        if analysis_type == "ora"
                        else f"{contrast_prefix}_gsea{func_suffix}.csv"
                    )
                )
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            pathways_dfs[contrast] = pd.DataFrame(
                columns=["Description", "GeneRatio", "geneID"]
            )

    if all([df.empty for df in pathways_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All functional results files were empty. "
            "No intersection possible."
        )
        return

    pathways_intersections = from_contents(
        {
            contrast: set(pathway_df["Description"].values)
            for contrast, pathway_df in pathways_dfs.items()
        }
    ).sort_index(ascending=False)

    # 2. Get DEGs (filtered by SHAP values) SYMBOL sets for each comparison
    degs_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                index_col=0,
            ).dropna(subset=["SYMBOL"])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degs_shap_dfs[contrast] = pd.DataFrame(columns=["SYMBOL"])

    if all([df.empty for df in degs_shap_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All bootstrap results files were empty. "
            "No intersection possible."
        )
        return

    # 3. Intersect genes (above SHAP value) from different contrasts assigned to each
    # pathway
    for pathway in pathways_intersections["id"].values:
        pathway_genes_shap_intersections = from_contents(
            {
                f"{key}_genes": set(
                    pathways_df.set_index("Description")[
                        (
                            "geneID"
                            if "geneID" in pathways_df.columns
                            else "core_enrichment"
                        )
                    ]
                    .get(pathway, "")
                    .split("/")
                ).intersection(degs_shap_dfs[key]["SYMBOL"].values)
                for key, pathways_df in pathways_dfs.items()
            }
        ).sort_index(ascending=False)

        if pathway_genes_shap_intersections.empty:
            continue

        try:
            n_all_common = len(
                pathway_genes_shap_intersections.loc[
                    tuple([True] * pathway_genes_shap_intersections.index.nlevels)
                ]
            )
        except KeyError:
            n_all_common = 0

        pathway_name = (
            pathway.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
        )
        pathway_genes_shap_intersections.to_csv(
            pathways_path.joinpath(
                f"{pathway_name}_intersecting_genes_shap_{n_all_common}.csv"
            )
        )

        fig = plt.figure(figsize=(15, 5), dpi=300)
        UpSet(
            pathway_genes_shap_intersections,
            subset_size="count",
            element_size=None,
            show_counts=True,
            show_percentages=True,
        ).plot(fig=fig)
        plt.suptitle(
            f"{pathway} \n("
            f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
            f"{p_col} < {p_th}, LFC > {lfc_th}, SHAP > {shap_th}, {func_db})",
        )
        plt.savefig(
            pathways_path.joinpath(
                f"{pathway_name}_intersecting_genes_shap_{n_all_common}_upsetplot.pdf"
            )
        )
        plt.close()


def intersect_wgcna(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    correlation_type: str = "bicor",
    network_type: str = "signed",
    genes_id: str = "ENTREZID",
) -> None:
    """
    Compute all possible intersecting sets of WGCNA modules between a given list of
    contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        network_type: Type of the network to compute.
        correlation_type: Co-expression metric.
    """
    # 0. Setup
    wgcna_path = root_path.joinpath("wgcna")
    save_path = root_path.joinpath("integrative_analysis").joinpath(
        "intersecting_wgcna"
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"{correlation_type}_{network_type}"
    ).replace(" ", "_")

    # 1. Get WGCNA module genes sets for each comparison
    wgcna_modules_dfs = defaultdict(dict)
    for contrast, contrast_prefix in contrast_prefixes.items():
        exp_path = (
            wgcna_path.joinpath(
                f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
            )
            .joinpath("standard")
            .joinpath("results")
        )
        for module_file in exp_path.glob(
            f"{correlation_type}_{network_type}_M*_genes.csv"
        ):
            module_name = module_file.stem.split("_")[2]
            if module_name != "M0":
                wgcna_modules_dfs[contrast][module_name] = pd.read_csv(
                    module_file
                ).dropna(subset=[genes_id])

    # 1.1. Check that not all contrast modules are empty.
    if all(
        [
            df.empty
            for contrast_modules in wgcna_modules_dfs.values()
            for df in contrast_modules.values()
        ]
    ):
        logging.warning(
            f"[{comparison_alias}] No modules detected for any comparisons. "
            "No intersection possible."
        )
        return

    # 2. Get WGCNA modules intersections
    module_names_sets = [
        tuple(wgcna_modules.keys()) for wgcna_modules in wgcna_modules_dfs.values()
    ]

    if len(module_names_sets) == 1:
        logging.warning(
            f"[{comparison_alias}]: Not enough WGCNA modules for comparison. Returning."
        )
        return

    for module_names_set in product(*module_names_sets):
        modules_intersections = from_contents(
            {
                f"{contrast}_{module_name}": set(
                    wgcna_modules[module_name][genes_id].values
                )
                for (contrast, wgcna_modules), module_name in zip(
                    wgcna_modules_dfs.items(), module_names_set
                )
            }
        ).sort_index(ascending=False)

        if modules_intersections.empty:
            continue

        try:
            n_all_common = len(
                modules_intersections.loc[
                    tuple([True] * modules_intersections.index.nlevels)
                ]
            )
        except KeyError:
            n_all_common = 0

        # 2.1. Save WGCNA modules intersection dataframe to disk
        module_names_set_str = "+".join(module_names_set)
        modules_intersections.reset_index().set_index("id").rename_axis(genes_id).join(
            pd.concat(
                [
                    df[module_name].set_index(genes_id)
                    for df, module_name in zip(
                        wgcna_modules_dfs.values(), module_names_set
                    )
                ]
            ).drop(
                columns=[
                    "baseMean",
                    "log2FoldChange",
                    "lfcSE",
                    "pvalue",
                    "padj",
                    "ClusterCoef",
                    "Connectivity",
                ]
            )
        ).reset_index().drop_duplicates(subset=[genes_id]).set_index(
            genes_id
        ).sort_values(
            modules_intersections.index.names, ascending=False
        ).to_csv(
            save_path.joinpath(
                f"{comparison_alias}_intersecting_wgcna"
                f"_{module_names_set_str}_{genes_id}_{n_all_common}.csv"
            )
        )

        # 2.2. Generate UpSet plot
        fig = plt.figure(figsize=(15, 5), dpi=300)
        UpSet(
            modules_intersections,
            subset_size="count",
            element_size=None,
            show_counts=True,
            show_percentages=True,
        ).plot(fig=fig)
        plt.suptitle(
            "Intersecting WGCNA modules' genes \n("
            f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
            f"{p_col} < {p_th}, LFC > {lfc_th})",
        )
        plt.savefig(
            save_path.joinpath(
                f"{comparison_alias}_intersecting_wgcna"
                f"_{module_names_set_str}_{genes_id}_{n_all_common}_upsetplot.pdf"
            )
        )
        plt.close()


def intersect_wgcna_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    correlation_type: str = "bicor",
    network_type: str = "signed",
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
) -> None:
    """
    Compute all possible intersecting sets of WGCNA modules between a given list of
    contrasts. Only consider module genes above a certain SHAP threshold.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        network_type: Type of the network to compute.
        correlation_type: Co-expression metric.
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    wgcna_path = root_path.joinpath("wgcna")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_wgcna_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"{correlation_type}_{network_type}_"
        f"{classifier_name}_{bootstrap_iterations}_shap_values_{shap_th_str}"
    ).replace(" ", "_")

    # 1. Get WGCNA module genes SYMBOL sets for each comparison
    wgcna_modules_dfs = defaultdict(dict)
    for contrast, contrast_prefix in contrast_prefixes.items():
        exp_path = (
            wgcna_path.joinpath(
                f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
            )
            .joinpath("standard")
            .joinpath("results")
        )
        for module_file in exp_path.glob(
            f"{correlation_type}_{network_type}_M*_genes.csv"
        ):
            module_name = module_file.stem.split("_")[2]
            if module_name != "M0":
                wgcna_modules_dfs[contrast][module_name] = pd.read_csv(
                    module_file, index_col=0
                ).dropna(subset=["SYMBOL"])

    if all(
        [
            df.empty
            for contrast_modules in wgcna_modules_dfs.values()
            for df in contrast_modules.values()
        ]
    ):
        logging.warning(
            f"[{comparison_alias}] No modules detected for any comparisons. "
            "No intersection possible."
        )
        return

    # 2. Get DEGs (filtered by SHAP values) SYMBOL sets for each comparison
    degs_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                index_col=0,
            ).dropna(subset=["SYMBOL"])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degs_shap_dfs[contrast] = pd.DataFrame(columns=["SYMBOL"])

    if all([df.empty for df in degs_shap_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All bootstrap results files were empty. "
            "No intersection possible."
        )
        return

    contrasts_degs_shap = {
        contrast: set(degs_shap_df["SYMBOL"].values)
        for contrast, degs_shap_df in degs_shap_dfs.items()
    }

    # 2. Get WGCNA modules intersections
    module_names_sets = [
        tuple(wgcna_modules.keys()) for wgcna_modules in wgcna_modules_dfs.values()
    ]

    if len(module_names_sets) == 1:
        logging.warning(
            f"[{comparison_alias}]: Not enough WGCNA modules for comparison. Returning."
        )
        return

    for module_names_set in product(*module_names_sets):
        modules_intersections = from_contents(
            {
                f"{contrast}_{module_name}": set(
                    wgcna_modules[module_name]["SYMBOL"].values
                ).intersection(contrasts_degs_shap[contrast])
                for (contrast, wgcna_modules), module_name in zip(
                    wgcna_modules_dfs.items(), module_names_set
                )
            }
        ).sort_index(ascending=False)

        if modules_intersections.empty:
            continue

        try:
            n_all_common = len(
                modules_intersections.loc[
                    tuple([True] * modules_intersections.index.nlevels)
                ]
            )
        except KeyError:
            n_all_common = 0

        # 1.1. Save WGCNA modules intersection dataframe to disk
        module_names_set_str = "+".join(module_names_set)
        modules_intersections.to_csv(
            save_path.joinpath(
                f"{comparison_alias}_intersecting_wgcna_{module_names_set_str}_"
                f"{n_all_common}.csv"
            )
        )

        # 2. Generate UpSet plot
        fig = plt.figure(figsize=(15, 5), dpi=300)
        UpSet(
            modules_intersections,
            subset_size="count",
            element_size=None,
            show_counts=True,
            show_percentages=True,
        ).plot(fig=fig)
        plt.suptitle(
            "Intersecting WGCNA modules \n("
            f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
            f"{p_col} < {p_th}, LFC > {lfc_th}, SHAP > {shap_th})",
        )
        plt.savefig(
            save_path.joinpath(
                f"{comparison_alias}_intersecting_wgcna_{module_names_set_str}_"
                f"{n_all_common}_upsetplot.pdf"
            )
        )
        plt.close()


def intersect_wgcna_pathways(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    correlation_type: str = "bicor",
    network_type: str = "signed",
    func_db: str = "KEGG",
) -> None:
    """
    Compute all possible intersecting sets of enriched pathways between a given list
    of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        network_type: Type of the network to compute.
        correlation_type: Co-expression metric.
        func_db: Functional database the pathways are enriched in.
    """
    # 0. Setup
    wgcna_path = root_path.joinpath("wgcna")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_wgcna_pathways")
        .joinpath(func_db)
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"{correlation_type}_{network_type}"
    ).replace(" ", "_")
    func_db_str = func_db.split("_")[0]

    # 1. Get pathways sets for each comparison
    modules_pathways_dfs = defaultdict(dict)
    for contrast, contrast_prefix in contrast_prefixes.items():
        exp_path = (
            wgcna_path.joinpath(
                f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
            )
            .joinpath("standard")
            .joinpath("functional")
            .joinpath(func_db_str)
        )
        for module_file in exp_path.glob(
            f"{correlation_type}_{network_type}_M*_ora.csv"
        ):
            module_name = module_file.stem.split("_")[2]
            if module_name != "M0":
                modules_pathways_dfs[contrast][module_name] = pd.read_csv(module_file)

    # 1.1. Check that not all contrast modules pathways are empty.
    if all(
        [
            df.empty
            for contrast_modules in modules_pathways_dfs.values()
            for df in contrast_modules.values()
        ]
    ):
        logging.warning(
            f"[{comparison_alias}, {func_db_str}]: "
            "No modules detected for any comparisons. No intersection possible."
        )
        return

    # 2. Get WGCNA modules pathways intersections
    module_names_sets = [
        tuple(wgcna_modules.keys()) for wgcna_modules in modules_pathways_dfs.values()
    ]

    if len(module_names_sets) == 1:
        logging.warning(
            f"[{comparison_alias}, {func_db_str}]: "
            "Not enough WGCNA modules for comparison. Returning."
        )
        return

    for module_names_set in product(*module_names_sets):
        modules_pathways_intersections = from_contents(
            {
                f"{contrast}_{module_name}": set(
                    wgcna_modules[module_name]["Description"].values
                )
                for (contrast, wgcna_modules), module_name in zip(
                    modules_pathways_dfs.items(), module_names_set
                )
            }
        ).sort_index(ascending=False)

        if modules_pathways_intersections.empty:
            continue

        try:
            n_all_common = len(
                modules_pathways_intersections.loc[
                    tuple([True] * modules_pathways_intersections.index.nlevels)
                ]
            )
        except KeyError:
            n_all_common = 0

        # 2.1. Get gene ratio for each pathway
        gene_ratio_metadata = pd.DataFrame(
            {
                f"{contrast}_GeneRatio_setSize": [
                    wgcna_modules[module_name]
                    .set_index("Description")[
                        (
                            "GeneRatio"
                            if "GeneRatio" in wgcna_modules[module_name].columns
                            else "setSize"
                        )
                    ]
                    .get(pathway, "")
                    for pathway in modules_pathways_intersections["id"].values
                ]
                for (contrast, wgcna_modules), module_name in zip(
                    modules_pathways_dfs.items(), module_names_set
                )
            }
        ).set_index(modules_pathways_intersections.index)

        # 2.2. Get gene content for each pathway (list of genes assigned to pathway)
        gene_IDs_metadata = pd.DataFrame(
            {
                f"{contrast}_geneID": [
                    wgcna_modules[module_name]
                    .set_index("Description")[
                        (
                            "geneID"
                            if "geneID" in wgcna_modules[module_name].columns
                            else "core_enrichment"
                        )
                    ]
                    .get(pathway, "")
                    for pathway in modules_pathways_intersections["id"].values
                ]
                for (contrast, wgcna_modules), module_name in zip(
                    modules_pathways_dfs.items(), module_names_set
                )
            }
        ).set_index(modules_pathways_intersections.index)

        # 2.3. Save pathways intersection and metadata dataframe to disk
        module_names_set_str = "+".join(module_names_set)
        pd.concat(
            (modules_pathways_intersections, gene_ratio_metadata, gene_IDs_metadata),
            axis=1,
        ).to_csv(
            save_path.joinpath(
                f"{comparison_alias}_intersecting_wgcna_pathways"
                f"_{module_names_set_str}_{n_all_common}.csv"
            )
        )

        # 2. Generate UpSet plot
        fig = plt.figure(figsize=(15, 5), dpi=300)
        UpSet(
            modules_pathways_intersections,
            subset_size="count",
            element_size=None,
            show_counts=True,
            show_percentages=True,
        ).plot(fig=fig)
        plt.suptitle(
            "Intersecting WGCNA pathways \n("
            f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
            f"{p_col} < {p_th}, LFC > {lfc_th}, {func_db})",
        )

        plt.savefig(
            save_path.joinpath(
                f"{comparison_alias}_intersecting_wgcna_pathways"
                f"_{module_names_set_str}_{n_all_common}_upsetplot.pdf"
            )
        )
        plt.close()


def test_prognostic_biomarkers(
    biomarkers: Iterable[str],
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    annot_df: pd.DataFrame,
    random_seed: int = 8080,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Compute prognostic metrics of each biomarker.

    Args:
        biomarkers: List of potential gene biomarkers.
        contrast_factor: Comparisons contrast factor.
        contrasts_levels: Iterable of tuples indicating pairwise contrasts comparisons.
        annot_df: Annotation dataframe.
        random_seed: Random seed.
    """
    prognostic_metrics = defaultdict(dict)
    for gene in biomarkers:
        for test, control in contrasts_levels:
            annot_df_contrasts = annot_df[
                annot_df[contrast_factor].isin((test, control))
            ]
            X = np.ascontiguousarray(annot_df_contrasts[[gene]])
            y = LabelEncoder().fit_transform(annot_df_contrasts[contrast_factor])

            cv_scores = cross_validate(
                LogisticRegressionCV(random_state=random_seed, max_iter=1000),
                X,
                y,
                scoring=[
                    "f1_weighted",
                    "balanced_accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "roc_auc_ovo_weighted",
                ],
                cv=10,
                n_jobs=-1,
            )

            cv_scores_df = pd.DataFrame(cv_scores)
            prognostic_metrics[f"{test}/{control}"][gene] = {
                metric: f"{mean:.2f} \u00B1 {std:.2f}"
                for metric, mean, std in zip(
                    cv_scores_df.columns,
                    cv_scores_df.mean().values,
                    cv_scores_df.std().values,
                )
            }
    return prognostic_metrics


def test_biomarkers_violin_plot(
    biomarkers: Iterable[Union[str, int]],
    contrast_factor: str,
    annot_df: pd.DataFrame,
    exp_prefix: str,
    save_path: Path,
    contrasts_levels_colors: Dict[str, str],
    metrics_summary: Dict,
    contrasts_levels_order: Iterable[str] = ("norm", "prim", "met_bb"),
    biomarkers_alias: str = "biomarkers",
) -> None:
    """Generate biomarkers violin plot.

    Args:
        biomarkers: List of potential gene biomarkers.
        contrast_factor: Comparisons contrast factor.
        annot_df: Annotation dataframe.
        exp_prefix: Experiment file prefix.
        save_path: Path to store results.
        contrasts_levels_colors: Colors of each contrast level.
        metrics_summary: A dataframe containing relevant metrics to include as
            annotations.
        biomarkers_alias: A name to uniquely identify sets of biomarkers.
        random_seed: Random seed.

    """
    # 0. Setup
    biomarkers_alias_str = (
        biomarkers_alias.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # 1. Format annotation dataframe
    annot_df_melted = pd.melt(
        annot_df[[*biomarkers, contrast_factor]],
        id_vars=[contrast_factor],
        var_name="gene",
        value_name="VST",
    ).rename(columns={contrast_factor: "Sample Type"})

    # 2. Generate violin plot
    plt.figure(figsize=(5 * len(biomarkers), 5), dpi=300)  # set high dpi for publishing
    ax = sns.violinplot(
        y="VST",
        x="gene",
        hue="Sample Type",
        data=annot_df_melted,
        palette=contrasts_levels_colors,
    )
    sns.move_legend(ax, "upper left")
    plt.title(biomarkers_alias, fontsize=18)
    plt.xlabel("Gene")
    plt.ylabel("Variance-stabilized transformed counts")

    # 3. Save plot
    plt.savefig(
        save_path.joinpath(f"{exp_prefix}_{biomarkers_alias_str}_violin_plot.pdf")
    )
    plt.savefig(
        save_path.joinpath(f"{exp_prefix}_{biomarkers_alias_str}_violin_plot.svg")
    )
    plt.close()


def test_biomarkers(
    biomarkers: Iterable[Union[str, int]],
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    vst_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    exp_prefix: str,
    deseq2_path: Path,
    save_path: Path,
    contrasts_levels_colors: Dict[str, str],
    contrasts_levels_order: Iterable[str] = ("norm", "prim", "met_bb"),
    plot_type: str = "violin",
    biomarkers_alias: str = "biomarkers",
    lfc_th: PositiveFloat = 1.0,
    p_th: float = 0.05,
    random_seed: int = 8080,
) -> None:
    """
    Given a list of biomarkers and a list of comparisons' contrasts, test prognostic
    capability of each biomarker

    Also generate violin plots or heatmap with annotated information.

    Args:
        biomarkers: List of potential gene biomarkers as SYMBOL IDs.
        contrast_factor: Comparisons contrast factor.
        contrasts_levels: Iterable of tuples indicating pairwise contrasts comparisons.
        vst_df: Dataframe of VST-normalized gene expression counts with gene SYMBOL IDs.
        annot_df: Annotation dataframe.
        exp_prefix: Experiment file prefix.
        deseq2_path: Path to a deseq2 experiment.
        save_path: Path to store results.
        contrasts_levels_colors: Colors of each contrast level.
        contrasts_levels_order: In which order the contrasts levels should appear in the
            plots.
        plot_type: Type of plot to generate.
        biomarkers_alias: A name to uniquely identify sets of biomarkers.
        lfc_th: LFC threshold to consider results significant.
        p_th: P-value threshold to consider results significant.
        random_seed: Random seed.
    """
    # 0. Setup
    if (not_available := set(biomarkers) - set(vst_df.index)) != {}:
        logging.warning(
            f"There is no expression data for some biomarkers: ({not_available}). They"
            " will be ignored."
        )
        biomarkers = vst_df.index.intersection(set(biomarkers))

    # 0.1. Add biomarker gene expression values to annotation file
    for gene in biomarkers:
        annot_df[gene] = vst_df.loc[gene, annot_df.index]

    # 0.2. Filter annotation to include only relevant contrast levels in the right order
    assert (
        set(annot_df[contrast_factor]) - set(contrasts_levels_order) == set()
    ), "contrasts_levels_order must contain all contrasts levels available."

    annot_df = pd.concat(
        [
            annot_df[annot_df[contrast_factor] == contrast_level]
            for contrast_level in contrasts_levels_order
        ]
    )

    # 0.3. Load biomarkers DESeq2 results data frames per comparison
    deseq2_results = dict()
    for test, control in contrasts_levels:
        res_df = (
            pd.read_csv(
                deseq2_path.joinpath(
                    f"{exp_prefix}_{test}_vs_{control}_deseq_results_unique.csv"
                ),
                index_col=0,
            )
            .dropna(subset=["ENTREZID", "SYMBOL"])
            .drop_duplicates(subset=["ENTREZID", "SYMBOL"], keep=False)
            .set_index("SYMBOL")
        )
        if (missing := set(biomarkers) - set(res_df.index)) != set():
            logging.warning(
                "Some biomarkers are missing in the DESeq2 results dataframe for "
                f"{test}/{control} ({missing}). The intersection will be used instead."
            )
        if missing == set(biomarkers):
            logging.error(
                "No biomarkers intersect with DESeq2 results dataframe for "
                f"{test}/{control}. Biomarkers test is cancelled."
            )
            return

        deseq2_results[f"{test}/{control}"] = res_df.loc[
            res_df.index.intersection(set(biomarkers))
        ]

    # 0.4. Standarize biomarker alias
    biomarkers_alias_str = (
        biomarkers_alias.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # 1. Fit a model for each biomarker and contrast to assess prognostic power
    prognostic_metrics = test_prognostic_biomarkers(
        biomarkers, contrast_factor, contrasts_levels, annot_df, random_seed
    )

    # 2. Gather metrics for each biomarker
    comparisons_lfc = {
        f"{test}/{control}": {
            gene: np.round(
                deseq2_results[f"{test}/{control}"].loc[gene, "log2FoldChange"], 2
            )
            for gene in biomarkers
        }
        for test, control in contrasts_levels
    }
    comparisons_padj = {
        f"{test}/{control}": {
            gene: np.round(deseq2_results[f"{test}/{control}"].loc[gene, "padj"], 2)
            for gene in biomarkers
        }
        for test, control in contrasts_levels
    }
    metrics_summary = {
        gene: {
            f"{test}/{control}": {
                "lfc": comparisons_lfc[f"{test}/{control}"][gene],
                "padj": comparisons_padj[f"{test}/{control}"][gene],
                "sig_up": (
                    (comparisons_lfc[f"{test}/{control}"][gene] > lfc_th)
                    and (comparisons_padj[f"{test}/{control}"][gene] < p_th)
                ),
                "sig_down": (
                    (comparisons_lfc[f"{test}/{control}"][gene] < -lfc_th)
                    and (comparisons_padj[f"{test}/{control}"][gene] < p_th)
                ),
                "precision": prognostic_metrics[f"{test}/{control}"][gene][
                    "test_precision_weighted"
                ],
                "recall": prognostic_metrics[f"{test}/{control}"][gene][
                    "test_recall_weighted"
                ],
                "f1": prognostic_metrics[f"{test}/{control}"][gene]["test_f1_weighted"],
                "balanced_accuracy": prognostic_metrics[f"{test}/{control}"][gene][
                    "test_balanced_accuracy"
                ],
                "au_roc": prognostic_metrics[f"{test}/{control}"][gene][
                    "test_roc_auc_ovo_weighted"
                ],
            }
            for test, control in contrasts_levels
        }
        for gene in biomarkers
    }
    df = pd.DataFrame.from_dict(metrics_summary, orient="index").stack().to_frame()
    metrics_summary_df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    metrics_summary_df.to_csv(
        save_path.joinpath(f"{exp_prefix}_{biomarkers_alias_str}_metrics_summary.csv")
    )

    # 3. Plot biomarkers
    if plot_type == "violin":
        test_biomarkers_violin_plot(
            biomarkers,
            contrast_factor,
            annot_df,
            exp_prefix,
            save_path,
            contrasts_levels_colors,
            metrics_summary,
            contrasts_levels_order,
            biomarkers_alias,
        )
    elif plot_type == "heatmap":
        annot_df_heatmap = annot_df.rename(columns={contrast_factor: "Sample Type"})
        counts_matrix = deepcopy(vst_df.loc[biomarkers])[annot_df.index]
        counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

        ha_column = heatmap_annotation(
            df=annot_df_heatmap[["Sample Type"]],
            col={"Sample Type": contrasts_levels_colors},
            show_annotation_name=True,
            annotation_legend_param=ro.ListVector(
                {
                    "Sample Type": ro.ListVector(
                        {"at": ro.StrVector(contrasts_levels_order)}
                    )
                }
            ),
        )
        ha_row_df = (
            metrics_summary_df["sig_up"]
            .unstack(level=1)
            .loc[
                counts_matrix.index,
                [f"{test}/{control}" for test, control in contrasts_levels],
            ]
        ).applymap(lambda x: f"LFC > {lfc_th}" if x else f"LFC < {lfc_th}")
        ha_row_df.sort_values(ha_row_df.columns.tolist(), inplace=True)
        ha_row = heatmap_annotation(
            df=ha_row_df,
            col={
                c: {f"LFC > {lfc_th}": "black", f"LFC < {lfc_th}": "grey"}
                for c in ha_row_df.columns
            },
            show_annotation_name=True,
            which="row",
        )

        complex_heatmap(
            counts_matrix.loc[ha_row_df.index],
            save_path=save_path.joinpath(
                f"{exp_prefix}_{biomarkers_alias_str}_heatmap.pdf"
            ),
            width=10,
            height=0.25 * len(biomarkers),
            name="VST",
            column_title=f"{biomarkers_alias}",
            top_annotation=ha_column,
            left_annotation=ha_row,
            show_row_names=True,
            show_column_names=False,
            cluster_columns=False,
            cluster_rows=False,
            column_split=ro.r.factor(
                ro.StrVector(
                    annot_df_heatmap["Sample Type"].loc[counts_matrix.columns].tolist()
                ),
                levels=ro.StrVector(contrasts_levels_order),
            ),
            cluster_column_slices=False,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )
    else:
        raise NotImplementedError(f"plot_type '{plot_type}' is not supported.")


def intersect_msigdb_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    classifier_name: str = "random_forest",
    msigdb_cat: str = "H",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
) -> None:
    """
    Compute all possible intersecting sets of MSigDB pathways of a given category,
    (filtered by SHAP values) between a given list of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_msigdb_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
        .joinpath(msigdb_cat)
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"shap_values_{shap_th_str}"
    ).replace(" ", "_")

    # 1. Get MSigDB pathways per category (filtered by SHAP values) for each comparison
    msig_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            msig_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("gene_sets_features")
                .joinpath(msigdb_cat)
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            msig_shap_dfs[contrast] = pd.DataFrame()

    if all([df.empty for df in msig_shap_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All bootstrap results files were empty. "
            "No intersection possible."
        )
        return

    msig_shap_intersections = from_contents(
        {
            contrast: set(msig_shap_df.index)
            for contrast, msig_shap_df in msig_shap_dfs.items()
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            msig_shap_intersections.loc[
                tuple([True] * msig_shap_intersections.index.nlevels)
            ]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Get gene membership
    msig_genes = pd.DataFrame(
        {
            f"{contrast}_genes": [
                msig_shap_df["degs_symbol"].dropna().get(gene_set, "")
                for gene_set in msig_shap_intersections["id"].values
            ]
            for contrast, msig_shap_df in msig_shap_dfs.items()
        }
    ).set_index(msig_shap_intersections.index)

    # 1.2. Save MSIGDB gene sets intersection dataframe to disk
    pd.concat((msig_shap_intersections, msig_genes), axis=1).to_csv(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_msig_shap_{n_all_common}.csv"
        )
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        msig_shap_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
    ).plot(fig=fig)
    plt.suptitle(
        f"Intersecting MSigDB {msigdb_cat} Gene Sets \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th}, SHAP > {shap_th})",
    )

    plt.savefig(
        save_path.joinpath(
            f"{comparison_alias}_intersecting_msig_shap_{n_all_common}_upsetplot.pdf"
        )
    )
    plt.close()


def intersect_msigdb_shap_genes(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 1.0,
    classifier_name: str = "random_forest",
    msigdb_cat: str = "H",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
) -> None:
    """
    Compute all possible intersecting sets of MSigDB pathways of a given category,
    (filtered by SHAP values) between a given list of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_msigdb_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
        .joinpath(msigdb_cat)
    )
    save_path.mkdir(exist_ok=True, parents=True)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}_"
        f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
        f"shap_values_{shap_th_str}"
    ).replace(" ", "_")
    gene_sets_path = save_path.joinpath(comparison_alias)
    gene_sets_path.mkdir(exist_ok=True, parents=True)

    # 1. Get MSigDB pathways per category (filtered by SHAP values) for each comparison
    msig_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            msig_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("gene_sets_features")
                .joinpath(msigdb_cat)
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            msig_shap_dfs[contrast] = pd.DataFrame()

    if all([df.empty for df in msig_shap_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] All bootstrap results files were empty. "
            "No intersection possible."
        )
        return

    msig_shap_intersections = from_contents(
        {
            contrast: set(msig_shap_df.index)
            for contrast, msig_shap_df in msig_shap_dfs.items()
        }
    ).sort_index(ascending=False)

    # 2. Intersect genes from different contrasts assigned to each gene set
    for gene_set in msig_shap_intersections["id"].values:
        gene_set_genes_intersections = from_contents(
            {
                f"{contrast}_genes": set(
                    msig_shap_df["degs_symbol"].dropna().get(gene_set, "").split("/")
                )
                for contrast, msig_shap_df in msig_shap_dfs.items()
            }
        ).sort_index(ascending=False)

        try:
            n_all_common = len(
                gene_set_genes_intersections.loc[
                    tuple([True] * gene_set_genes_intersections.index.nlevels)
                ]
            )
        except KeyError:
            n_all_common = 0

        gene_set_name = (
            gene_set.replace(" ", "_")
            .replace("/", "")
            .replace("(", "")
            .replace(")", "")
        )
        gene_set_genes_intersections.to_csv(
            gene_sets_path.joinpath(
                f"{gene_set_name}_intersecting_genes_{n_all_common}.csv"
            )
        )

        fig = plt.figure(figsize=(15, 5), dpi=300)
        UpSet(
            gene_set_genes_intersections,
            subset_size="count",
            element_size=None,
            show_counts=True,
            show_percentages=True,
        ).plot(fig=fig)
        plt.suptitle(
            f"Intersecting DEGs in {gene_set} ({msigdb_cat}) \n("
            f"{'de' if lfc_level == 'all' else lfc_level}-regulated DEGs, "
            f"{p_col} < {p_th}, LFC > {lfc_th}, SHAP > {shap_th})"
        )

        plt.savefig(
            gene_sets_path.joinpath(
                f"{gene_set_name}_intersecting_genes_{n_all_common}_upsetplot.pdf"
            )
        )
        plt.close()


def intersect_degss(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    msigdb_cat: str,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 0.0,
) -> None:
    """
    Compute all possible intersecting sets of DEGSs (differentially enriched gene sets)
        between a given list of contrasts.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
    """
    # 0. Setup
    diff_path = root_path.joinpath("diff_gsva").joinpath(msigdb_cat)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss")
        .joinpath(msigdb_cat)
        .joinpath(f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}")
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}"
    ).replace(" ", "_")

    # 1. Get DEGSs names sets for each comparison
    degss_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degss_dfs[contrast] = pd.read_csv(
                diff_path.joinpath(
                    f"{contrast_prefix}_top_table_"
                    f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degss_dfs[contrast] = pd.DataFrame(
                columns=["gs_description", "entrez_gene", "gene_symbol"]
            )

    if all([df.empty for df in degss_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] "
            "All differential enrichment results files were empty. "
            "No intersection possible."
        )
        return

    degss_intersections = from_contents(
        {contrast: set(degss_df.index) for contrast, degss_df in degss_dfs.items()}
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degss_intersections.loc[tuple([True] * degss_intersections.index.nlevels)]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Annotate DEGSs intersection dataframe and save to disk
    degss_intersections.reset_index().set_index("id").join(
        pd.concat(degss_dfs.values())[
            ["gs_description", "entrez_gene", "gene_symbol"]
        ].drop_duplicates()
    ).sort_values(degss_intersections.index.names, ascending=False).to_csv(
        save_path.joinpath(f"{comparison_alias}_{n_all_common}.csv")
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degss_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
        sort_categories_by=None,
    ).plot(fig=fig)
    plt.suptitle(
        "Intersecting DEGSs \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th})",
    )
    plt.savefig(save_path.joinpath(f"{comparison_alias}_{n_all_common}_upsetplot.pdf"))
    plt.close()


def intersect_degss_degs(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    msigdb_cat: str,
    comparison_alias: str = "",
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 0.0,
) -> None:
    """
    Compute all possible intersecting sets of DEGSs (differentially enriched gene sets)
        between a given list of contrasts. Then, for each DEGSs, intersect the DEGs of
        each contrast with the gene set contents.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
    """
    # 0. Setup
    diff_path = root_path.joinpath("diff_gsva").joinpath(msigdb_cat)
    deseq_path = root_path.joinpath("deseq2")
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_degs")
        .joinpath(msigdb_cat)
        .joinpath(
            f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}+"
            f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
        )
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}"
    ).replace(" ", "_")

    # 1. Get DEGSs names sets for each comparison
    degss_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degss_dfs[contrast] = pd.read_csv(
                diff_path.joinpath(
                    f"{contrast_prefix}_top_table_"
                    f"{degss_p_col}_{degss_p_th_str}_"
                    f"{degss_lfc_level}_{degss_lfc_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dataframe", e)
            degss_dfs[contrast] = pd.DataFrame(
                columns=["gs_description", "entrez_gene", "gene_symbol"]
            )

    if all([df.empty for df in degss_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] "
            "All differential enrichment results files were empty. "
            "No intersection possible."
        )
        return

    degss_intersections = from_contents(
        {contrast: set(degss_df.index) for contrast, degss_df in degss_dfs.items()}
    ).sort_index(ascending=False)

    # 1.1. Annotate DEGSs intersection dataframe
    degss_intersections = (
        degss_intersections.reset_index()
        .set_index("id")
        .join(
            pd.concat(degss_dfs.values())[
                ["gs_description", "entrez_gene", "gene_symbol"]
            ].drop_duplicates()
        )
        .sort_values(degss_intersections.index.names, ascending=False)
    )

    # 2. Load DEGs for each contrast
    degs_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_dfs[contrast] = pd.read_csv(
                deseq_path.joinpath(
                    f"{contrast_prefix}_"
                    f"{degs_p_col}_{degs_p_th_str}_"
                    f"{degs_lfc_level}_{degs_lfc_th_str}_deseq_results_unique.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dataframe", e)
            degs_dfs[contrast] = pd.DataFrame(columns=["ENTREZID", "SYMBOL"])

    # 3. Calculate intersection between each gene set genes and contrasts DEGs
    for contrast in contrast_prefixes.keys():
        degss_intersections[f"{contrast}_degs_entrez"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        set(degs_dfs[contrast]["ENTREZID"].astype(str))
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["entrez_gene"]
        ]
        degss_intersections[f"{contrast}_degs_symbol"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        set(degs_dfs[contrast]["SYMBOL"])
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["gene_symbol"]
        ]

    # 4. Save final dataframe to disk
    degss_intersections.to_csv(save_path.joinpath(f"{comparison_alias}.csv"))


def intersect_degss_degs_summary_helper(
    results_root: Path,
    results_key: str,
    degs_count: int = 2,
    msigdb_cats_summary: Iterable[str] = ("H", "C2", "C5", "C6"),
) -> None:
    """Helper function to summarize intersections between DEGSs.

    Args:
        results_root: Path to save results to.
        results_key: String ID to uniquely identify summary results.
        degs_count: Minimum count of DEGs to consider the gene sets.
        msigdb_cats_summary: MSigDB categories to include for the summary.
    """
    results_dict = defaultdict(lambda: defaultdict(dict))
    for file_path in results_root.glob(f"*/{results_key}/*.csv"):
        msigdb_cat = file_path.parents[1].stem

        if msigdb_cat not in msigdb_cats_summary:
            continue

        results_df = pd.read_csv(file_path, index_col=0)

        # 1. Get contrast columns
        contrasts_cols = results_df.select_dtypes(bool).columns

        # 2. Get gene counts for each contrast
        for contrast_col in contrasts_cols:
            results_df[f"{contrast_col}_degs_counts"] = results_df[
                f"{contrast_col}_degs_symbol"
            ].apply(lambda x: len(x.split("/")) if isinstance(x, str) else x)

        # 2. Loop over all possible combinations
        for contrasts_flags in product([False, True], repeat=len(contrasts_cols)):
            intersect_comp_id = "".join(
                [
                    f"+++{contrast_col}" if flag else f"---{contrast_col}"
                    for contrast_col, flag in zip(contrasts_cols, contrasts_flags)
                ]
            )

            # 2.1. Get gene sets matching contrasts flags
            mask = np.logical_and.reduce(
                [
                    results_df[contrast_col] == flag
                    for contrast_col, flag in zip(contrasts_cols, contrasts_flags)
                ]
                + [
                    results_df[f"{contrast_col}_degs_counts"] > degs_count
                    for contrast_col, flag in zip(contrasts_cols, contrasts_flags)
                    if flag
                ]
            )
            gene_sets_df = results_df[mask]

            # 2.2. Extract DEGs information for each relevant contrast column
            results_dict[msigdb_cat][intersect_comp_id] = (
                gene_sets_df[
                    [
                        f"{contrast_col}_degs_symbol"
                        for contrast_col, flag in zip(contrasts_cols, contrasts_flags)
                        if flag
                    ]
                ]
                .transpose()
                .to_dict()
            )

    with results_root.joinpath(f"{results_key}_results_summary.json").open("w") as fp:
        json.dump(results_dict, fp, indent=4)


def intersect_degss_degs_summary(
    root_path: Path,
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 0.0,
    degs_count: int = 2,
    msigdb_cats_summary: Iterable[str] = ("H", "C2", "C5", "C6"),
) -> None:
    """
    Summarize the results of `intersect_degss_degs`.

    Args:
        root_path: Root path of the RNASeq analysis.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
        degs_count: Minimum number of DEGs inside a gene set to be included in the
            summary.
        msigdb_cats_summary: Which MSigDB categories to summarize.
    """
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    results_root = root_path.joinpath("integrative_analysis").joinpath(
        "intersecting_degss_degs"
    )
    results_key = (
        f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}+"
        f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
    )

    intersect_degss_degs_summary_helper(
        results_root, results_key, degs_count, msigdb_cats_summary
    )


def intersect_degss_degs_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    msigdb_cat: str,
    comparison_alias: str = "",
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 0.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
    extra_genes: Iterable[Tuple[str, str]] = (("2346", "FOLH1"),),
) -> None:
    """
    Compute all possible intersecting sets of DEGSs (differentially enriched gene sets)
        between a given list of contrasts. Then, for each DEGSs, intersect the DEGs of
        each contrast with the gene set contents.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.7
        extra_genes: Extra genes to add to the intersection with the gene set genes,
            independent of SHAP values. An iterable of tuples, where the first item of
            each tuple is the ENTREZID and the second the SYMBOL ID.
    """
    # 0. Setup
    diff_path = root_path.joinpath("diff_gsva").joinpath(msigdb_cat)
    ml_path = root_path.joinpath("ml_classifiers")
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_degs_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
        .joinpath(msigdb_cat)
        .joinpath(
            f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}+"
            f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}+"
            f"shap_{shap_th_str}"
        )
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}"
    ).replace(" ", "_")

    # 1. Get DEGSs names sets for each comparison
    degss_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degss_dfs[contrast] = pd.read_csv(
                diff_path.joinpath(
                    f"{contrast_prefix}_top_table_"
                    f"{degss_p_col}_{degss_p_th_str}_"
                    f"{degss_lfc_level}_{degss_lfc_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dataframe", e)
            degss_dfs[contrast] = pd.DataFrame(
                columns=["gs_description", "entrez_gene", "gene_symbol"]
            )

    if all([df.empty for df in degss_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] "
            "All differential enrichment results files were empty. "
            "No intersection possible."
        )
        return

    degss_intersections = from_contents(
        {contrast: set(degss_df.index) for contrast, degss_df in degss_dfs.items()}
    ).sort_index(ascending=False)

    # 1.1. Annotate DEGSs intersection dataframe
    degss_intersections = (
        degss_intersections.reset_index()
        .set_index("id")
        .join(
            pd.concat(degss_dfs.values())[
                ["gs_description", "entrez_gene", "gene_symbol"]
            ].drop_duplicates()
        )
        .sort_values(degss_intersections.index.names, ascending=False)
    )

    # 2. Load DEGs for each contrast
    degs_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_"
                    f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                dtype={"ENTREZID": str},
            ).dropna(subset=["ENTREZID", "SYMBOL"])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dataframe", e)
            degs_shap_dfs[contrast] = pd.DataFrame(columns=["ENTREZID", "SYMBOL"])

    # 3. Calculate intersection between each gene set genes and contrasts DEGs
    for contrast in contrast_prefixes.keys():
        degss_intersections[f"{contrast}_degs_entrez"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        {
                            *set(degs_shap_dfs[contrast]["ENTREZID"].astype(str)),
                            *{gene[0] for gene in extra_genes},
                        }
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["entrez_gene"]
        ]
        degss_intersections[f"{contrast}_degs_symbol"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        {
                            *set(degs_shap_dfs[contrast]["SYMBOL"]),
                            *{gene[1] for gene in extra_genes},
                        }
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["gene_symbol"]
        ]

    # 4. Save final dataframe to disk
    degss_intersections.to_csv(save_path.joinpath(f"{comparison_alias}.csv"))


def intersect_degss_degs_shap_summary(
    root_path: Path,
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 1.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
    degs_count: int = 2,
    msigdb_cats_summary: Iterable[str] = ("H", "C2", "C5", "C6"),
) -> None:
    """
    Summarize the results of `intersect_degss_degs`.

    Args:
        root_path: Root path of the RNASeq analysis.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
        classifier_name: Name of classifier model used to obtain SHAP values.
        bootstrap_iterations: Number of bootstrap iterations used to obtain SHAP values.
        shap_th: SHAP value threshold used to determine the most significant genes.
        degs_count: Minimum number of DEGs inside a gene set to be included in the
            summary.
        msigdb_cats_summary: Which MSigDB categories to summarize.
    """
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    results_root = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_degs_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
    )
    results_key = (
        f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}+"
        f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}+"
        f"shap_{shap_th_str}"
    )

    intersect_degss_degs_summary_helper(
        results_root, results_key, degs_count, msigdb_cats_summary
    )


def intersect_degss_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    msigdb_cat: str,
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 0.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
) -> None:
    """
    Compute all possible intersecting sets of DEGSs (differentially enriched gene sets)
        between a given list of contrasts, that are above a certain SHAP value.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_shap")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
        .joinpath(msigdb_cat)
        .joinpath(f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}_shap_{shap_th_str}")
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}"
    ).replace(" ", "_")

    # 1. Get DEGSs names sets for each comparison
    degss_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degss_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("gene_sets_features")
                .joinpath(msigdb_cat)
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degss_dfs[contrast] = pd.DataFrame(
                columns=["gs_description", "entrez_gene", "gene_symbol"]
            )

    if all([df.empty for df in degss_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] "
            "All differential enrichment results files were empty. "
            "No intersection possible."
        )
        return

    degss_intersections = from_contents(
        {contrast: set(degss_df.index) for contrast, degss_df in degss_dfs.items()}
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degss_intersections.loc[tuple([True] * degss_intersections.index.nlevels)]
        )
    except KeyError:
        n_all_common = 0

    # 1.1. Annotate DEGSs intersection dataframe and save to disk
    degss_intersections.reset_index().set_index("id").join(
        pd.concat(degss_dfs.values())[
            ["gs_description", "entrez_gene", "gene_symbol"]
        ].drop_duplicates()
    ).sort_values(degss_intersections.index.names, ascending=False).to_csv(
        save_path.joinpath(f"{comparison_alias}_{n_all_common}.csv")
    )

    # 2. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degss_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
        sort_categories_by=None,
    ).plot(fig=fig)
    plt.suptitle(
        "Intersecting DEGSs \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th})",
    )
    plt.savefig(save_path.joinpath(f"{comparison_alias}_{n_all_common}_upsetplot.pdf"))
    plt.close()


def intersect_degss_shap_degs(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    msigdb_cat: str,
    comparison_alias: str = "",
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 0.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
) -> None:
    """
    Compute all possible intersecting sets of DEGSs (differentially enriched gene sets)
        between a given list of contrasts, that are above a certain SHAP value.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    deseq_path = root_path.joinpath("deseq2")
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_shap_degs")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
        .joinpath(msigdb_cat)
        .joinpath(
            f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}"
            f"_shap_{shap_th_str}+"
            f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
        )
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}"
    ).replace(" ", "_")

    # 1. Get DEGSs names sets for each comparison
    degss_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degss_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_"
                    f"{degss_p_col}_{degss_p_th_str}_"
                    f"{degss_lfc_level}_{degss_lfc_th_str}"
                )
                .joinpath(classifier_name)
                .joinpath("gene_sets_features")
                .joinpath(msigdb_cat)
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{bootstrap_iterations}_shap_values_{shap_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degss_dfs[contrast] = pd.DataFrame(
                columns=["gs_description", "entrez_gene", "gene_symbol"]
            )

    if all([df.empty for df in degss_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] "
            "All differential enrichment results files were empty. "
            "No intersection possible."
        )
        return

    degss_intersections = from_contents(
        {contrast: set(degss_df.index) for contrast, degss_df in degss_dfs.items()}
    ).sort_index(ascending=False)

    # 1.1. Annotate DEGSs intersection dataframe
    degss_intersections = (
        degss_intersections.reset_index()
        .set_index("id")
        .join(
            pd.concat(degss_dfs.values())[
                ["gs_description", "entrez_gene", "gene_symbol"]
            ].drop_duplicates()
        )
        .sort_values(degss_intersections.index.names, ascending=False)
    )

    # 2. Load DEGs for each contrast
    degs_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_dfs[contrast] = pd.read_csv(
                deseq_path.joinpath(
                    f"{contrast_prefix}_"
                    f"{degs_p_col}_{degs_p_th_str}_"
                    f"{degs_lfc_level}_{degs_lfc_th_str}_deseq_results_unique.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dataframe", e)
            degs_dfs[contrast] = pd.DataFrame(columns=["ENTREZID", "SYMBOL"])

    # 3. Calculate intersection between each gene set genes and contrasts DEGs
    for contrast in contrast_prefixes.keys():
        degss_intersections[f"{contrast}_degs_entrez"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        set(degs_dfs[contrast]["ENTREZID"].astype(str))
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["entrez_gene"]
        ]
        degss_intersections[f"{contrast}_degs_symbol"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        set(degs_dfs[contrast]["SYMBOL"])
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["gene_symbol"]
        ]

    # 4. Save final dataframe to disk
    degss_intersections.to_csv(save_path.joinpath(f"{comparison_alias}.csv"))


def intersect_degss_shap_degs_summary(
    root_path: Path,
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 0.0,
    classifier_name: str = "random_forest",
    bootstrap_iterations: int = 10000,
    shap_th: float = 0.001,
    degs_count: int = 2,
    msigdb_cats_summary: Iterable[str] = ("H", "C2", "C5", "C6"),
) -> None:
    """
    Summarize the results of `intersect_degss_degs`.

    Args:
        root_path: Root path of the RNASeq analysis.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
        degs_count: Minimum number of DEGs inside a gene set to be included in the
            summary.
        msigdb_cats_summary: Which MSigDB categories to summarize.
    """
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    shap_th_str = str(shap_th).replace(".", "_")
    results_root = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_shap_degs")
        .joinpath(f"{classifier_name}_{bootstrap_iterations}")
    )
    results_key = (
        f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}"
        f"_shap_{shap_th_str}+"
        f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
    )

    intersect_degss_degs_summary_helper(
        results_root, results_key, degs_count, msigdb_cats_summary
    )


def intersect_degss_shap_degs_shap(
    contrast_prefixes: Dict[str, str],
    root_path: Path,
    msigdb_cat: str,
    comparison_alias: str = "",
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 1.0,
    degss_classifier_name: str = "random_forest",
    degs_classifier_name: str = "random_forest",
    degss_bootstrap_iterations: int = 10000,
    degs_bootstrap_iterations: int = 10000,
    degss_shap_th: float = 0.001,
    degs_shap_th: float = 0.001,
    extra_genes: Iterable[Tuple[str, str]] = (("2346", "FOLH1"),),
) -> None:
    """
    Compute all possible intersecting sets of DEGSs (differentially enriched gene sets)
        between a given list of contrasts, that are above a certain SHAP value.

    Args:
        contrast_prefixes: A mapping of contrast keys and file prefixes.
        root_path: Root path of the RNASeq analysis.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
        degss_classifier_name: Name of classifier model used to obtain
            DEGSs SHAP values.
        degs_classifier_name: Name of classifier model used to obtain
            DEGs SHAP values.
        degss_bootstrap_iterations: Number of bootstrap iterations used to obtain
            DEGSs SHAP values.
        degs_bootstrap_iterations: Number of bootstrap iterations used to obtain
            DEGs SHAP values.
        degss_shap_th: SHAP value threshold used to determine the most significant
            genes sets.
        degs_shap_th: SHAP value threshold used to determine the most significant
            genes.
        extra_genes: Extra genes to add to the intersection with the gene set genes,
            independent of SHAP values. An iterable of tuples, where the first item of
            each tuple is the ENTREZID and the second the SYMBOL ID.
    """
    # 0. Setup
    ml_path = root_path.joinpath("ml_classifiers")
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    degss_shap_th_str = str(degss_shap_th).replace(".", "_")
    degs_shap_th_str = str(degs_shap_th).replace(".", "_")
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_shap_degs_shap")
        .joinpath(
            f"{degss_classifier_name}_{degss_bootstrap_iterations}+"
            f"{degs_classifier_name}_{degs_bootstrap_iterations}"
        )
        .joinpath(msigdb_cat)
        .joinpath(
            f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}"
            f"_shap_{degss_shap_th_str}+"
            f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
            f"_shap_{degs_shap_th_str}"
        )
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (
        f"{comparison_alias or '+'.join(contrast_prefixes.keys())}"
    ).replace(" ", "_")

    # 1. Get DEGSs names sets for each comparison
    degss_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degss_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_"
                    f"{degss_p_col}_{degss_p_th_str}_"
                    f"{degss_lfc_level}_{degss_lfc_th_str}"
                )
                .joinpath(degss_classifier_name)
                .joinpath("gene_sets_features")
                .joinpath(msigdb_cat)
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{degss_bootstrap_iterations}_"
                    f"shap_values_{degss_shap_th_str}.csv"
                ),
                index_col=0,
            )
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dictionary", e)
            degss_dfs[contrast] = pd.DataFrame(
                columns=["gs_description", "entrez_gene", "gene_symbol"]
            )

    if all([df.empty for df in degss_dfs.values()]):
        logging.warning(
            f"[{comparison_alias}] "
            "All differential enrichment results files were empty. "
            "No intersection possible."
        )
        return

    degss_intersections = from_contents(
        {contrast: set(degss_df.index) for contrast, degss_df in degss_dfs.items()}
    ).sort_index(ascending=False)

    # 1.1. Annotate DEGSs intersection dataframe
    degss_intersections = (
        degss_intersections.reset_index()
        .set_index("id")
        .join(
            pd.concat(degss_dfs.values())[
                ["gs_description", "entrez_gene", "gene_symbol"]
            ].drop_duplicates()
        )
        .sort_values(degss_intersections.index.names, ascending=False)
    )

    # 2. Load DEGs for each contrast
    degs_shap_dfs = {}
    for contrast, contrast_prefix in contrast_prefixes.items():
        try:
            degs_shap_dfs[contrast] = pd.read_csv(
                ml_path.joinpath(
                    f"{contrast_prefix}_"
                    f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
                )
                .joinpath(degs_classifier_name)
                .joinpath("genes_features")
                .joinpath("bootstrap")
                .joinpath(
                    f"bootstrap_{degs_bootstrap_iterations}_"
                    f"shap_values_{degs_shap_th_str}.csv"
                ),
                dtype={"ENTREZID": str},
            ).dropna(subset=["ENTREZID", "SYMBOL"])
        except FileNotFoundError as e:
            logging.warning("File not found, setting empty dataframe", e)
            degs_shap_dfs[contrast] = pd.DataFrame(columns=["ENTREZID", "SYMBOL"])

    # 3. Calculate intersection between each gene set genes and contrasts DEGs
    for contrast in contrast_prefixes.keys():
        degss_intersections[f"{contrast}_degs_entrez"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        {
                            *set(degs_shap_dfs[contrast]["ENTREZID"].astype(str)),
                            *{gene[0] for gene in extra_genes},
                        }
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["entrez_gene"]
        ]
        degss_intersections[f"{contrast}_degs_symbol"] = [
            (
                "/".join(
                    set(gene_set_genes.split("/")).intersection(
                        {
                            *set(degs_shap_dfs[contrast]["SYMBOL"].astype(str)),
                            *{gene[1] for gene in extra_genes},
                        }
                    )
                )
                if isinstance(gene_set_genes, str)
                else ""
            )
            for gene_set_genes in degss_intersections["gene_symbol"]
        ]

    # 4. Save final dataframe to disk
    degss_intersections.to_csv(save_path.joinpath(f"{comparison_alias}.csv"))


def intersect_degss_shap_degs_shap_summary(
    root_path: Path,
    degss_p_col: str = "padj",
    degs_p_col: str = "padj",
    degss_p_th: float = 0.05,
    degs_p_th: float = 0.05,
    degss_lfc_level: str = "all",
    degs_lfc_level: str = "all",
    degss_lfc_th: float = 0.0,
    degs_lfc_th: float = 1.0,
    degss_classifier_name: str = "random_forest",
    degs_classifier_name: str = "random_forest",
    degss_bootstrap_iterations: int = 10000,
    degs_bootstrap_iterations: int = 10000,
    degss_shap_th: float = 0.001,
    degs_shap_th: float = 0.001,
    degs_count: int = 2,
    msigdb_cats_summary: Iterable[str] = ("H", "C2", "C5", "C6"),
) -> None:
    """
    Summarize the results of `intersect_degss_degs`.

    Args:
        contrast_control: Contrast whose genes sets we don't want to intersect.
        contrast_test_0: Contrast whose genes sets we want.
        contrast_test_1: Contrast whose genes sets we want.
        root_path: Root path of the RNASeq analysis.
        degss_p_col: P-value column name to be used (DEGSs).
        degs_p_col: P-value column name to be used (DEGs).
        degss_p_th: P-value threshold (DEGSs).
        degs_p_th: P-value threshold (DEGs).
        degss_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGSs).
        degs_lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes)
            (DEGs).
        degss_lfc_th: Log2 Fold Chance threshold (DEGSs).
        degs_lfc_th: Log2 Fold Chance threshold (DEGs).
        degss_classifier_name: Name of classifier model used to obtain
            DEGSs SHAP values.
        degs_classifier_name: Name of classifier model used to obtain
            DEGs SHAP values.
        degss_bootstrap_iterations: Number of bootstrap iterations used to obtain
            DEGSs SHAP values.
        degs_bootstrap_iterations: Number of bootstrap iterations used to obtain
            DEGs SHAP values.
        degss_shap_th: SHAP value threshold used to determine the most significant
            genes sets.
        degs_shap_th: SHAP value threshold used to determine the most significant
            genes.
        degs_count: Minimum number of DEGs inside a gene set to be included in the
            summary.
        msigdb_cats_summary: Which MSigDB categories to summarize.
    """
    degss_p_th_str = str(degss_p_th).replace(".", "_")
    degs_p_th_str = str(degs_p_th).replace(".", "_")
    degss_lfc_th_str = str(degss_lfc_th).replace(".", "_")
    degs_lfc_th_str = str(degs_lfc_th).replace(".", "_")
    degss_shap_th_str = str(degss_shap_th).replace(".", "_")
    degs_shap_th_str = str(degs_shap_th).replace(".", "_")
    results_root = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_shap_degs_shap")
        .joinpath(
            f"{degss_classifier_name}_{degss_bootstrap_iterations}+"
            f"{degs_classifier_name}_{degs_bootstrap_iterations}"
        )
    )
    results_key = (
        f"{degss_p_col}_{degss_p_th_str}_{degss_lfc_level}_{degss_lfc_th_str}"
        f"_shap_{degss_shap_th_str}+"
        f"{degs_p_col}_{degs_p_th_str}_{degs_lfc_level}_{degs_lfc_th_str}"
        f"_shap_{degs_shap_th_str}"
    )

    intersect_degss_degs_summary_helper(
        results_root, results_key, degs_count, msigdb_cats_summary
    )


def intersect_degss_gsea(
    contrast_name_prefix: Tuple[str, str],
    root_path: Path,
    msigdb_cat: str,
    gsea_files: Dict[str, Iterable[Path]],
    comparison_alias: str = "",
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 0.0,
) -> None:
    """
    Compute intersections between DEGSs (differentially enriched gene sets) and GSEA
        results for each given contrast.

    Args:
        contrast_name_prefix: A contrast key and its file prefix.
        root_path: Root path of the RNASeq analysis.
        msigdb_cat: Category of MSigDB to extract the gene sets from.
        comparison_alias: Optionally include a comparison alias for naming figures and
            files.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
    """
    # 0. Setup
    diff_path = root_path.joinpath("diff_gsva").joinpath(msigdb_cat)
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    contrast, contrast_prefix = contrast_name_prefix
    degss_all_path = diff_path.joinpath(f"{contrast_prefix}_top_table.csv")
    degss_path = diff_path.joinpath(
        f"{contrast_prefix}_top_table_{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}.csv"
    )
    save_path = (
        root_path.joinpath("integrative_analysis")
        .joinpath("intersecting_degss_gsea")
        .joinpath(msigdb_cat)
        .joinpath(f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}")
    )
    save_path.mkdir(exist_ok=True, parents=True)
    comparison_alias = (f"{comparison_alias or contrast_prefix[0]}").replace(" ", "_")

    # 0.1. Save all paths for future reference
    with save_path.joinpath(f"{comparison_alias}_input_file_paths.json").open(
        "w"
    ) as fp:
        json.dump(
            {
                "degss_all_path": str(degss_all_path),
                "degss_path": str(degss_path),
                **{k: str(v) for k, v in gsea_files.items()},
            },
            fp,
            indent=4,
        )

    # 1. Get all unfiltered DEGSs for given contrast
    try:
        degss_all_df = pd.read_csv(degss_all_path, index_col=0)
    except FileNotFoundError as e:
        logging.warning("File not found, returning.", e)
        return

    # 2. Get DEGSs IDs for given contrast (according to certain filtering criteria)
    try:
        degss_df = pd.read_csv(degss_path, index_col=0)
    except FileNotFoundError as e:
        logging.error("File not found, returning.", e)
        return

    if degss_df.empty:
        logging.warning(
            f"[{comparison_alias}] "
            "The differential enrichment results file was empty. "
            "No intersection possible."
        )
        return

    # 3. Get GSEA gene sets for given contrast
    gsea_dfs = {}
    for contrast_id, gsea_file in gsea_files.items():
        try:
            gsea_df = pd.read_csv(
                gsea_file,
                index_col=0,
            ).set_index("ID")
        except FileNotFoundError as e:
            logging.warning(f"File ({gsea_file}) not found, continuing. {e}")
            continue

        if gsea_df.empty:
            logging.warning(
                f"[{comparison_alias}] The GSEA results file was empty. "
                "No intersection possible."
            )
            continue

        if lfc_level == "up":
            gsea_df = gsea_df[gsea_df["enrichmentScore"] > 0]
        elif lfc_level == "down":
            gsea_df = gsea_df[gsea_df["enrichmentScore"] < 0]

        gsea_dfs[contrast_id] = gsea_df

    if len(gsea_dfs) == 0:
        logging.warning(
            f"[{comparison_alias}] All GSEA results files were empty. "
            "No intersection possible."
        )
        return

    # 4. Compute intersection between DEGSs and GSEA results
    degss_gsea_ids_intersections = from_contents(
        {
            f"{contrast}_{gene_sets_src}": set(gene_sets.index)
            for gene_sets_src, gene_sets in zip(
                ("gsva", *gsea_dfs.keys()), (degss_df, *gsea_dfs.values())
            )
        }
    ).sort_index(ascending=False)

    try:
        n_all_common = len(
            degss_gsea_ids_intersections.loc[
                tuple([True] * degss_gsea_ids_intersections.index.nlevels)
            ]
        )
    except KeyError:
        n_all_common = 0

    # 5. Generate UpSet plot
    fig = plt.figure(figsize=(15, 5), dpi=300)
    UpSet(
        degss_gsea_ids_intersections,
        subset_size="count",
        element_size=None,
        show_counts=True,
        show_percentages=True,
        sort_categories_by=None,
    ).plot(fig=fig)
    plt.suptitle(
        "Intersecting gene sets from GSVA and GSEA \n("
        f"{'de' if lfc_level == 'all' else lfc_level}-regulated, "
        f"{p_col} < {p_th}, LFC > {lfc_th})",
    )
    plt.savefig(save_path.joinpath(f"{comparison_alias}_{n_all_common}_upsetplot.pdf"))
    plt.close()

    # 6. Annotate DEGSs intersection dataframe and save to disk
    gsea_all_df = deepcopy(degss_all_df)
    for gsea_key, gsea_df in gsea_dfs.items():
        gsea_all_df = gsea_all_df.join(gsea_df, how="outer", rsuffix=gsea_key)

    degss_gsea_df = (
        degss_gsea_ids_intersections.reset_index()
        .set_index("id")
        .join(gsea_all_df)
        .sort_values(degss_gsea_ids_intersections.index.names, ascending=False)
    )
    degss_gsea_df.to_csv(save_path.joinpath(f"{comparison_alias}_{n_all_common}.csv"))

    # 6.1. Save all gene sets sorted by log2FoldChange
    degss_gsea_df.drop(columns=degss_gsea_ids_intersections.index.names).sort_values(
        "log2FoldChange", ascending=False
    ).to_csv(save_path.joinpath(f"{comparison_alias}_union.csv"))
