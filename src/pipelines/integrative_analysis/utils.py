import json
import logging
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from upsetplot import UpSet, from_contents


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
