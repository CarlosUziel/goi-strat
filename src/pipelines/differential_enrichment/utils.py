"""
Utilities for differential enrichment analysis using GSVA scores.

This module provides functions for differential enrichment analysis of gene set
variation analysis (GSVA) scores, which quantify pathway activity in individual
samples. The main functionalities include:

1. Differential enrichment analysis:
   - Apply limma-based statistical testing on GSVA scores
   - Generate differential enrichment results for various contrasts
   - Filter results based on significance thresholds and fold change criteria

2. Visualization of results:
   - Generate heatmaps of differentially enriched gene sets
   - Create PCA plots to visualize sample clustering by pathway activity
   - Produce volcano plots highlighting significant gene sets

3. Support for multi-condition analysis:
   - Compare expression across different sample types or experimental conditions
   - Handle complex experimental designs with multiple factors
   - Generate consistent output formats across different contrasts

The module integrates with R libraries through rpy2 for specialized statistics
and visualization while maintaining a consistent Python interface.
"""

import re
from itertools import chain, product
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation
from r_wrappers.limma import (
    empirical_bayes,
    fit_contrasts,
    linear_model_fit,
    make_contrasts,
    top_table,
)
from r_wrappers.utils import get_design_matrix, pd_df_to_rpy2_df, rpy2_df_to_pd_df


def diff_enrich_gsva_limma(
    gsva_matrix_path: Path,
    msigdb_cat_meta_path: Path,
    annot_df_contrasts: pd.DataFrame,
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    results_path: Path,
    exp_prefix: str,
    p_cols: Iterable[str],
    p_ths: Iterable[float],
    lfc_levels: Iterable[str],
    lfc_ths: Iterable[float],
    design_factors: Optional[Iterable[str]] = None,
) -> None:
    """
    Perform differential enrichment analysis on a GSVA matrix using the limma package.

    Args:
        gsva_matrix_path (Path): Path to the GSVA matrix file.
        msigdb_cat_meta_path (Path): Path to the MSigDB category metadata file.
        annot_df_contrasts (pd.DataFrame): DataFrame containing sample annotations.
        contrast_factor (str): Column name in `annot_df_contrasts` used for contrasts.
        contrasts_levels (Iterable[Tuple[str, str]]): Pairs of contrast levels (test, control).
        results_path (Path): Directory to save the analysis results.
        exp_prefix (str): Prefix for naming output files.
        p_cols (Iterable[str]): Names of p-value columns for filtering.
        p_ths (Iterable[float]): Thresholds for p-value significance.
        lfc_levels (Iterable[str]): Levels of log fold change (e.g., "up", "down").
        lfc_ths (Iterable[float]): Thresholds for log fold change.
        design_factors (Optional[Iterable[str]]): Additional factors for the design matrix.

    Returns:
        None
    """
    # 0. Setup
    design_factors = design_factors or [contrast_factor]
    design_factors = (
        design_factors
        if contrast_factor in design_factors
        else design_factors + [contrast_factor]
    )
    gsva_matrix = pd.read_csv(gsva_matrix_path, index_col=0)
    msigdb_cat_meta = pd.read_csv(msigdb_cat_meta_path, index_col=0)

    common_idxs = gsva_matrix.columns.intersection(annot_df_contrasts.index)
    gsva_matrix = gsva_matrix.loc[:, common_idxs]
    annot_df_contrasts = annot_df_contrasts.loc[common_idxs, :]

    # Replace invalid characters with underscores
    def sanitize_factor(factor: str):
        return re.sub(r"\W|^(?=\d)", "_", factor)

    rename_map = {f: sanitize_factor(f) for f in design_factors}
    design_factors_safe = list(rename_map.values())
    annot_df_contrasts_safe = annot_df_contrasts.rename(columns=rename_map)

    # 1. Contrasts design matrix
    with localconverter(ro.default_converter):
        design_matrix = get_design_matrix(
            annot_df_contrasts_safe[design_factors_safe], design_factors_safe
        )

    # 2. Fit linear model
    with localconverter(ro.default_converter):
        lm_fitted = linear_model_fit(pd_df_to_rpy2_df(gsva_matrix), design_matrix)

    # 3. Get differential analysis statistics
    with localconverter(ro.default_converter):
        contrast_matrix = make_contrasts(
            contrasts=ro.StrVector(
                [f"{test}-{control}" for test, control in contrasts_levels]
            ),
            levels=design_matrix,
        )

    with localconverter(ro.default_converter):
        contrasts_fit = empirical_bayes(
            fit=fit_contrasts(fit=lm_fitted, contrasts=contrast_matrix)
        )

    # 4. Get differentially enriched gene sets for each comparison
    for i, (test, control) in enumerate(contrasts_levels, 1):
        # 4.1. Unfiltered results
        with localconverter(ro.default_converter):
            pd_df = rpy2_df_to_pd_df(top_table(contrasts_fit, coef=i, num=ro.r("Inf")))

        diff_gene_sets = (
            pd_df.rename(
                columns={
                    "logFC": "log2FoldChange",
                    "P.Val": "pvalue",
                    "adj.P.Val": "padj",
                }
            ).sort_values("log2FoldChange", key=abs, ascending=False)
        ).join(msigdb_cat_meta)
        diff_gene_sets.to_csv(
            results_path.joinpath(f"{exp_prefix}_{test}_vs_{control}_top_table.csv")
        )

        # 4.2. Filter by different P-value and LFC criteria
        for p_col, p_th, lfc_level, lfc_th in product(
            p_cols, p_ths, lfc_levels, lfc_ths
        ):
            # Filter by P-value column
            diff_gene_sets_sig = diff_gene_sets[diff_gene_sets[p_col] < p_th]

            # Filer by log2FoldChange
            if lfc_level == "up":
                diff_gene_sets_sig = diff_gene_sets_sig[
                    diff_gene_sets_sig["log2FoldChange"] > 0
                ]
            elif lfc_level == "down":
                diff_gene_sets_sig = diff_gene_sets_sig[
                    diff_gene_sets_sig["log2FoldChange"] < 0
                ]

            diff_gene_sets_sig = diff_gene_sets_sig[
                abs(diff_gene_sets_sig["log2FoldChange"]) > lfc_th
            ]

            # Save result to disk
            p_col_str = p_col.replace(".", "_")
            p_thr_str = str(p_th).replace(".", "_")
            lfc_thr_str = str(lfc_th).replace(".", "_")
            diff_gene_sets_sig.sort_values(
                "log2FoldChange", key=abs, ascending=False
            ).to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_{test}_vs_{control}_top_table_"
                    f"{p_col_str}_{p_thr_str}_{lfc_level}_{lfc_thr_str}.csv"
                )
            )


def diff_enrich_gsva_heatmaps(
    gsva_matrix_path: Path,
    annot_df_contrasts: pd.DataFrame,
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    contrast_levels_colors: Dict[str, str],
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    p_cols: Iterable[str],
    p_ths: Iterable[float],
    lfc_levels: Iterable[str],
    lfc_ths: Iterable[float],
    heatmap_top_n: int = 1000,
) -> None:
    """
    Generate heatmaps for differential enrichment results from a GSVA matrix.

    Args:
        gsva_matrix_path (Path): Path to the GSVA matrix file.
        annot_df_contrasts (pd.DataFrame): DataFrame containing sample annotations.
        contrast_factor (str): Column name in `annot_df_contrasts` used for contrasts.
        contrasts_levels (Iterable[Tuple[str, str]]): Pairs of contrast levels (test, control).
        contrast_levels_colors (Dict[str, str]): Mapping of contrast levels to colors.
        results_path (Path): Directory to save the analysis results.
        plots_path (Path): Directory to save the generated plots.
        exp_prefix (str): Prefix for naming output files.
        p_cols (Iterable[str]): Names of p-value columns for filtering.
        p_ths (Iterable[float]): Thresholds for p-value significance.
        lfc_levels (Iterable[str]): Levels of log fold change (e.g., "up", "down").
        lfc_ths (Iterable[float]): Thresholds for log fold change.
        heatmap_top_n (int): Number of top rows to include in the heatmap. Defaults to 1000.

    Returns:
        None
    """
    # 0. Setup
    gsva_matrix = pd.read_csv(gsva_matrix_path, index_col=0)
    annot_df_contrasts = annot_df_contrasts[
        annot_df_contrasts[contrast_factor].isin(set(chain(*contrasts_levels)))
    ]
    common_samples = annot_df_contrasts.index.intersection(gsva_matrix.columns)
    annot_df_contrasts = annot_df_contrasts.loc[common_samples, :]
    gsva_matrix = gsva_matrix.loc[:, common_samples]

    # 1. Unsupervised heatmap
    top_n = min(heatmap_top_n, len(gsva_matrix))
    top_var_genes = gsva_matrix.var(axis=1).sort_values(ascending=False)[:top_n]
    counts_matrix = gsva_matrix.loc[top_var_genes.index, :]
    # substract row means for better visualization
    counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

    save_path = plots_path.joinpath(
        f"{exp_prefix}_unsupervised_gene_sets_clustering.pdf"
    )
    ha_column = heatmap_annotation(
        df=annot_df_contrasts[[contrast_factor]],
        col={contrast_factor: contrast_levels_colors},
        show_annotation_name=False,
    )

    with localconverter(ro.default_converter):
        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            width=10,
            height=8,
            heatmap_legend_side="top",
            name=f"Top {top_n} variable gene sets",
            column_title=contrast_factor,
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=False,
            cluster_columns=False,
            cluster_rows=False,
            column_split=ro.r.factor(
                ro.StrVector(
                    annot_df_contrasts[contrast_factor]
                    .loc[counts_matrix.columns]
                    .tolist()
                ),
                levels=ro.StrVector(list(contrast_levels_colors.keys())),
            ),
            cluster_column_slices=False,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

    # 2. Supervised heatmaps
    for (test, control), p_col, p_th, lfc_level, lfc_th in product(
        contrasts_levels, p_cols, p_ths, lfc_levels, lfc_ths
    ):
        # 2.1. Filter annotation and counts dataframe
        annot_df_test_control = annot_df_contrasts[
            annot_df_contrasts[contrast_factor].isin([test, control])
        ].sort_values(contrast_factor)
        common_samples = annot_df_test_control.index.intersection(gsva_matrix.columns)
        annot_df_test_control = annot_df_test_control.loc[common_samples, :]
        counts_matrix = gsva_matrix.loc[:, common_samples]

        # 2.2. Load differential results, filtered by multiple criteria
        p_col_str = p_col.replace(".", "_")
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_th).replace(".", "_")
        diff_result = pd.read_csv(
            results_path.joinpath(
                f"{exp_prefix}_{test}_vs_{control}_top_table_"
                f"{p_col_str}_{p_thr_str}_{lfc_level}_{lfc_thr_str}.csv"
            ),
            index_col=0,
        )

        # 2.3. Select top differentially expressed genes based on log2FC
        top_n = min(heatmap_top_n, len(diff_result))

        if top_n == 0:
            continue

        filtered_gene_sets = diff_result.sort_values(
            ["log2FoldChange", p_col], ascending=[False, True], key=abs
        ).iloc[:top_n]

        counts_matrix = counts_matrix.loc[filtered_gene_sets.index, :]
        # substract row means for better visualization
        counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

        # 2.4. Define heatmap annotation
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_th).replace(".", "_")
        save_path = plots_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_"
            f"{lfc_level}_{lfc_thr_str}_supervised_gene_sets_clustering.pdf"
        )
        ha_column = heatmap_annotation(
            df=annot_df_test_control[[contrast_factor]],
            col={contrast_factor: contrast_levels_colors},
            show_annotation_name=False,
        )

        # 2.5. Plot heatmap
        with localconverter(ro.default_converter):
            complex_heatmap(
                counts_matrix,
                save_path=save_path,
                width=10,
                height=int(0.3 * len(counts_matrix)),
                name=f"Top {top_n} DEGSs (LFC > {lfc_th}, {p_col} < {p_th})",
                column_title=contrast_factor,
                top_annotation=ha_column,
                show_row_names=True,
                show_column_names=False,
                cluster_columns=False,
                cluster_rows=False,
                column_split=ro.r.factor(
                    ro.StrVector(
                        annot_df_test_control[contrast_factor]
                        .loc[counts_matrix.columns]
                        .tolist()
                    ),
                    levels=ro.StrVector(list(contrast_levels_colors.keys())),
                ),
                cluster_column_slices=False,
                heatmap_legend_param=ro.r(
                    'list(title_position = "topcenter", color_bar = "continuous",'
                    ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
                ),
            )


def diff_enrich_gsva(
    gsva_matrix_path: Path,
    msigdb_cat_meta_path: Path,
    annot_df_contrasts: pd.DataFrame,
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    contrast_levels_colors: Dict[str, str],
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    p_cols: Iterable[str],
    p_ths: Iterable[float],
    lfc_levels: Iterable[str],
    lfc_ths: Iterable[float],
    design_factors: Optional[Iterable[str]] = None,
    heatmap_top_n: int = 1000,
) -> None:
    """
    Perform differential enrichment analysis and generate heatmaps for a GSVA matrix.

    Args:
        gsva_matrix_path (Path): Path to the GSVA matrix file.
        msigdb_cat_meta_path (Path): Path to the MSigDB category metadata file.
        annot_df_contrasts (pd.DataFrame): DataFrame containing sample annotations.
        contrast_factor (str): Column name in `annot_df_contrasts` used for contrasts.
        contrasts_levels (Iterable[Tuple[str, str]]): Pairs of contrast levels (test, control).
        contrast_levels_colors (Dict[str, str]): Mapping of contrast levels to colors.
        results_path (Path): Directory to save the analysis results.
        plots_path (Path): Directory to save the generated plots.
        exp_prefix (str): Prefix for naming output files.
        p_cols (Iterable[str]): Names of p-value columns for filtering.
        p_ths (Iterable[float]): Thresholds for p-value significance.
        lfc_levels (Iterable[str]): Levels of log fold change (e.g., "up", "down").
        lfc_ths (Iterable[float]): Thresholds for log fold change.
        design_factors (Optional[Iterable[str]]): Additional factors for the design matrix.
        heatmap_top_n (int): Number of top rows to include in the heatmap. Defaults to 1000.

    Returns:
        None
    """
    # 1. Run differential analysis
    diff_enrich_gsva_limma(
        gsva_matrix_path=gsva_matrix_path,
        msigdb_cat_meta_path=msigdb_cat_meta_path,
        annot_df_contrasts=annot_df_contrasts,
        contrast_factor=contrast_factor,
        contrasts_levels=contrasts_levels,
        p_cols=p_cols,
        p_ths=p_ths,
        lfc_levels=lfc_levels,
        lfc_ths=lfc_ths,
        exp_prefix=exp_prefix,
        results_path=results_path,
        design_factors=design_factors,
    )

    # 2. Plot heatmaps
    diff_enrich_gsva_heatmaps(
        gsva_matrix_path=gsva_matrix_path,
        annot_df_contrasts=annot_df_contrasts,
        contrast_factor=contrast_factor,
        exp_prefix=exp_prefix,
        results_path=results_path,
        plots_path=plots_path,
        contrasts_levels=contrasts_levels,
        contrast_levels_colors=contrast_levels_colors,
        p_cols=p_cols,
        p_ths=p_ths,
        lfc_levels=lfc_levels,
        lfc_ths=lfc_ths,
        heatmap_top_n=heatmap_top_n,
    )
