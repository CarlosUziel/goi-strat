"""
Utilities for differential gene expression analysis using RNA-seq data.

This module provides functions to analyze differential gene expression between sample
groups using DESeq2 and associated visualizations. It supports:

1. Running full differential expression analysis pipelines, including:
   - Creating DESeq2 datasets from raw count data
   - Performing variance stabilizing transformations
   - Running differential expression tests between sample groups
   - Filtering and annotating significant differentially expressed genes (DEGs)

2. Generating visualizations for expression data:
   - PCA plots to visualize sample clustering
   - Heatmaps of differentially expressed genes
   - Sample distance matrices
   - Gene expression distribution plots

3. Processing and managing differential expression results:
   - Filtering results by significance thresholds and fold change
   - Annotating genes with additional identifiers
   - Saving results and intermediate data for downstream analyses

The module integrates with R libraries (through rpy2) to leverage established
bioinformatics tools while providing a Python interface for workflow management.
"""

import logging
import re
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
import plotly.express as px
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from sklearn.decomposition import PCA

from components.functional_analysis.orgdb import OrgDB
from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation
from r_wrappers.deseq2 import (
    filter_dds,
    get_deseq_dataset_htseq,
    get_deseq_dataset_matrix,
    lfc_shrink,
    norm_transform,
    rlog_transform,
    run_dseq2,
    vst_transform,
)
from r_wrappers.utils import (
    annotate_deseq_result,
    assay_to_df,
    filter_deseq_results,
    map_gene_id,
    pd_df_to_rpy2_df,
    rpy2_df_to_pd_df,
    sample_distance,
    save_rds,
)
from r_wrappers.visualization import (
    heatmap_sample_distance,
    ma_plot,
    mds_plot,
    mean_sd_plot,
    venn_diagram,
    volcano_plot,
)


def proc_diff_expr_dataset_plots(
    dataset: Any,
    dataset_label: str,
    annot_df_contrasts: pd.DataFrame,
    plots_path: Path,
    exp_prefix: str,
    org_db: OrgDB,
    contrast_factor: str,
    contrast_levels_colors: Dict[str, str],
    heatmap_top_n: int = 1000,
) -> None:
    """
    Generate multiple visualization plots for differential expression datasets.

    This function creates a comprehensive set of visualizations for exploring differential
    expression data. It produces sample clustering heatmaps, gene expression heatmaps,
    mean-SD plots, PCA plots, and MDS plots to help visualize relationships between samples
    and identify patterns of gene expression.

    Args:
        dataset: A DESeq2 dataset object (or transformed version like VST or rlog).
        dataset_label: String identifier for the dataset type (e.g., "dds", "VST", "RLD")
            that will be used in output file names.
        annot_df_contrasts: Sample annotation dataframe with metadata for each sample,
            indexed by sample ID. Must contain the contrast_factor column.
        plots_path: Path to directory where generated plots will be stored.
        exp_prefix: String prefix to be used for all output filenames.
        org_db: Organism annotation database object for gene ID conversion.
        contrast_factor: Column name in annot_df_contrasts used to group samples for
            differential analysis and visualization (e.g., "sample_type").
        contrast_levels_colors: Dictionary mapping factor levels to colors for plots
            (e.g., {"tumor": "red", "normal": "blue"}).
        heatmap_top_n: Maximum number of top variable genes to include in heatmaps.
            Default is 1000.

    Returns:
        None. Multiple plot files are written to the specified plots_path directory:
        - Sample clustering heatmaps (PDF)
        - Unsupervised gene clustering heatmaps (PDF)
        - Mean-SD plots (PDF)
        - PCA plots (PDF and HTML)
        - MDS plots (PDF)

    Examples:
        >>> proc_diff_expr_dataset_plots(
        ...     dataset=vst,
        ...     dataset_label="VST",
        ...     annot_df_contrasts=sample_annotations,
        ...     plots_path=Path("/results/plots"),
        ...     exp_prefix="experiment1",
        ...     org_db=org_db_object,
        ...     contrast_factor="treatment",
        ...     contrast_levels_colors={"control": "blue", "treated": "red"},
        ... )
    """
    # 1. Samples clustering
    sample_dist = sample_distance(dataset)

    save_path = plots_path.joinpath(
        f"{exp_prefix}_samples_clustering_{dataset_label}.pdf"
    )
    with localconverter(ro.default_converter):
        heatmap_sample_distance(sample_dist, save_path)

    # 2. Genes clustering (unsupervised)
    with localconverter(ro.default_converter):
        counts_df = assay_to_df(
            norm_transform(dataset) if dataset_label == "dds" else dataset
        )
        try:
            counts_df.index = map_gene_id(counts_df.index, org_db, "ENSEMBL", "SYMBOL")
        except Exception as e:
            logging.warn(e)

    # get rid of non-uniquely mapped transcripts
    counts_df = counts_df.loc[~counts_df.index.str.contains("/", na=False)]
    # remove all transcripts that share SYMBOL IDs
    counts_df = counts_df.loc[counts_df.index.dropna().drop_duplicates(keep=False)]

    top_n = min(heatmap_top_n, len(counts_df))
    top_var_genes = counts_df.var(axis=1).sort_values(ascending=False)[:top_n]
    counts_matrix = counts_df.loc[top_var_genes.index, :]
    counts_matrix = counts_matrix.sub(
        counts_matrix.mean(axis=1), axis=0
    )  # substract row means for better visualization

    save_path = plots_path.joinpath(
        f"{exp_prefix}_unsupervised_genes_clustering_{dataset_label}.pdf"
    )
    with localconverter(ro.default_converter):
        ha_column = heatmap_annotation(
            df=annot_df_contrasts[[contrast_factor]],
            col={contrast_factor: contrast_levels_colors},
            show_annotation_name=False,
        )

        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            width=10,
            height=8,
            column_title=f"Top {top_n} variable genes ({dataset_label})",
            name=f"Gene Expression ({dataset_label})",
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=True,
            cluster_columns=True,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

        # for publishing
        save_path = plots_path.joinpath(
            f"{exp_prefix}_unsupervised_genes_clustering_{dataset_label}_pub.pdf"
        )
        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            width=10,
            height=8,
            column_title=f"Top {top_n} variable genes ({dataset_label})",
            name=f"Gene Expression ({dataset_label})",
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=False,
            cluster_columns=False,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

    # 3. Mean SD plot
    with localconverter(ro.default_converter):
        save_path = plots_path.joinpath(
            f"{exp_prefix}_mean_sd_plot_{dataset_label}.pdf"
        )
        mean_sd_plot(
            norm_transform(dataset) if dataset_label == "dds" else dataset, save_path
        )

    # 4. Principal Component Analysis (PCA)
    with localconverter(ro.default_converter):
        dataset_df = assay_to_df(dataset)

    pca = PCA(n_components=2, random_state=8080)
    components = pca.fit_transform(dataset_df.transpose())
    ratios = pca.explained_variance_ratio_ * 100

    labels = annot_df_contrasts.loc[dataset_df.columns, contrast_factor]
    fig = px.scatter(
        components,
        x=0,
        y=1,
        labels={"0": f"PC 1 ({ratios[0]:.2f}%)", "1": f"PC 2 ({ratios[1]:.2f}%)"},
        color=labels,
        color_discrete_map=contrast_levels_colors,
        hover_name=dataset_df.columns,
        title=f"All samples ({dataset_label})",
    )

    fig.write_image(plots_path.joinpath(f"{exp_prefix}_pca_{dataset_label}.pdf"))
    fig.write_html(plots_path.joinpath(f"{exp_prefix}_pca_{dataset_label}.html"))

    # 5. Multidimensional scaling (MDS)
    with localconverter(ro.default_converter):
        sample_dist = sample_distance(dataset)
        save_path = plots_path.joinpath(f"{exp_prefix}_MDS_{dataset_label}.pdf")
        mds_plot(
            sample_dist,
            dataset,
            color=re.sub(r"\W|^(?=\d)", "_", contrast_factor),
            save_path=save_path,
        )


def proc_diff_expr_dataset(
    annot_df_contrasts: pd.DataFrame,
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    org_db: OrgDB,
    factors: Iterable[str],
    contrast_factor: str,
    contrast_levels_colors: Dict[str, str],
    design_factors: Iterable[str] = None,
    heatmap_top_n: int = 1000,
    counts_matrix: Optional[pd.DataFrame] = None,
    counts_path: Optional[Path] = None,
    counts_files_pattern: str = "*.tsv",
    compute_vst: bool = True,
    compute_rlog: bool = False,
) -> Tuple[Any, Any, Any]:
    """
    Generate DESeq2 dataset and normalized counts with visualizations.

    This function creates a DESeq2 dataset from either count files or a pre-computed matrix,
    optionally computes variance-stabilizing transformations (VST) and regularized
    log transformations (rlog), and generates visualization plots for each dataset.
    Results are saved to disk and comprehensive quality control plots are generated.

    Args:
        annot_df_contrasts: Annotation dataframe of samples to be analyzed, indexed by
            sample IDs that match either column names in counts_matrix or file names
            in counts_path.
        results_path: Directory path where intermediate and final results will be saved.
        plots_path: Directory path where plots will be saved.
        exp_prefix: A string prefix for all generated file names.
        org_db: Organism annotation database object for gene ID mapping.
        factors: Columns from annot_df_contrasts to be included in the DESeq2 dataset.
        contrast_factor: Column name in annot_df_contrasts used for grouping samples and
            performing differential expression analysis.
        contrast_levels_colors: Dictionary mapping factor levels to colors for plots
            (e.g., {"tumor": "red", "normal": "blue"}).
        design_factors: Columns to include in the design formula for differential analysis.
            If None, defaults to [contrast_factor].
        heatmap_top_n: Maximum number of top variable genes to include in heatmaps.
        counts_matrix: Optional pre-computed counts matrix with genes in rows and
            samples in columns. If provided, counts_path is ignored.
        counts_path: Optional directory path containing count files (one per sample).
            Used only if counts_matrix is None.
        counts_files_pattern: File pattern for count files. Used only with counts_path.
        compute_vst: Whether to compute variance-stabilizing transformation. Default is True.
        compute_rlog: Whether to compute regularized log transformation. More accurate
            than VST for small sample sizes but slower to compute. Default is False.

    Raises:
        ValueError: If neither counts_matrix nor counts_path with counts_files_pattern
            is provided.

    Returns:
        Tuple[Any, Any, Any]: A tuple containing:
            - dds: DESeq2 dataset object
            - vst: Variance stabilized transformed dataset (or None if compute_vst=False)
            - rld: Regularized log transformed dataset (or None if compute_rlog=False)

    Examples:
        >>> # Using count files
        >>> dds, vst, rld = proc_diff_expr_dataset(
        ...     annot_df_contrasts=sample_annotations,
        ...     results_path=Path("/results/deseq2"),
        ...     plots_path=Path("/results/plots"),
        ...     exp_prefix="experiment1",
        ...     org_db=org_db_object,
        ...     factors=["sample_type", "batch"],
        ...     contrast_factor="sample_type",
        ...     contrast_levels_colors={"tumor": "red", "normal": "blue"},
        ...     counts_path=Path("/data/counts"),
        ...     compute_vst=True,
        ...     compute_rlog=False
        ... )
        >>>
        >>> # Using a pre-computed counts matrix
        >>> dds, vst, _ = proc_diff_expr_dataset(
        ...     annot_df_contrasts=sample_annotations,
        ...     results_path=Path("/results/deseq2"),
        ...     plots_path=Path("/results/plots"),
        ...     exp_prefix="experiment1",
        ...     org_db=org_db_object,
        ...     factors=["sample_type"],
        ...     contrast_factor="sample_type",
        ...     contrast_levels_colors={"tumor": "red", "normal": "blue"},
        ...     counts_matrix=raw_counts_df,
        ...     compute_vst=True,
        ...     compute_rlog=False
        ... )
    """
    # 1. DESeq2 dataset
    design_factors = design_factors if design_factors is not None else [contrast_factor]
    design_factors = (
        design_factors
        if contrast_factor in design_factors
        else design_factors + [contrast_factor]
    )
    for factor in design_factors:
        if factor not in factors:
            factors.append(factor)

    # 1.1. Get dataset
    if counts_path and counts_files_pattern:
        with localconverter(ro.default_converter):
            dds = get_deseq_dataset_htseq(
                annot_df=annot_df_contrasts,
                counts_path=counts_path,
                factors=factors,
                design_factors=design_factors,
                counts_files_pattern=counts_files_pattern,
            )
    elif counts_matrix is not None:
        with localconverter(ro.default_converter):
            dds = get_deseq_dataset_matrix(
                counts_matrix=counts_matrix,
                annot_df=annot_df_contrasts,
                factors=factors,
                design_factors=design_factors,
            )
    else:
        raise ValueError(
            "Either plots_path and counts_files_pattern or counts_matrix must be used."
        )

    # 1.2. Counts filter of DESeq2 datasets
    dds = filter_dds(dds, 10)

    # 1.3. Save to disk
    with localconverter(ro.default_converter):
        save_path = results_path.joinpath(f"{exp_prefix}_dds")
        rpy2_df_to_pd_df(ro.r("counts")(dds)).to_csv(save_path.with_suffix(".csv"))
        save_rds(dds, save_path.with_suffix(".RDS"))

    # 1.4. Dataset visualizations
    with localconverter(ro.default_converter):
        proc_diff_expr_dataset_plots(
            dataset=dds,
            dataset_label="DDS",
            annot_df_contrasts=deepcopy(annot_df_contrasts),
            plots_path=plots_path,
            exp_prefix=exp_prefix,
            org_db=org_db,
            contrast_factor=contrast_factor,
            contrast_levels_colors=contrast_levels_colors,
            heatmap_top_n=heatmap_top_n,
        )

    # 2. Variance Stabilizing Transform (VST)
    if compute_vst:
        with localconverter(ro.default_converter):
            # 2.1. Get dataset
            vst = vst_transform(dds)

            # 2.2. Save to disk
            with localconverter(ro.default_converter):
                save_path = results_path.joinpath(f"{exp_prefix}_vst.csv")
                rpy2_df_to_pd_df(ro.r("assay")(vst)).to_csv(save_path)

            # 2.3. Dataset visualizations
            proc_diff_expr_dataset_plots(
                dataset=vst,
                dataset_label="VST",
                annot_df_contrasts=deepcopy(annot_df_contrasts),
                plots_path=plots_path,
                exp_prefix=exp_prefix,
                org_db=org_db,
                contrast_factor=contrast_factor,
                contrast_levels_colors=contrast_levels_colors,
                heatmap_top_n=heatmap_top_n,
            )
    else:
        vst = None

    # 3. RLOG transform
    if compute_rlog:
        with localconverter(ro.default_converter):
            # 3.1. Get dataset
            rld = rlog_transform(dds)

            # 3.2. Save to disk
            with localconverter(ro.default_converter):
                save_path = results_path.joinpath(f"{exp_prefix}_rld.csv")
                rpy2_df_to_pd_df(ro.r("assay")(rld)).to_csv(save_path)

            # 3.3. Dataset visualizations
            proc_diff_expr_dataset_plots(
                dataset=rld,
                dataset_label="RLD",
                annot_df_contrasts=deepcopy(annot_df_contrasts),
                plots_path=plots_path,
                exp_prefix=exp_prefix,
                org_db=org_db,
                contrast_factor=contrast_factor,
                contrast_levels_colors=contrast_levels_colors,
                heatmap_top_n=heatmap_top_n,
            )
    else:
        rld = None

    return dds, vst, rld


def proc_diff_expr_results(
    dds: Any,
    annot_df_contrasts: pd.DataFrame,
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    org_db: OrgDB,
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    contrast_levels_colors: Dict[str, str],
    p_cols: Iterable[str],
    p_ths: Iterable[float],
    lfc_levels: Iterable[str],
    lfc_ths: Iterable[float],
    vst: Optional[Any] = None,
    heatmap_top_n: int = 1000,
) -> None:
    """
    Process differential expression results and generate visualizations.

    This function performs differential expression testing on a DESeq2 dataset across multiple
    contrasts, filters the results using various significance thresholds, generates visualizations
    (MA plots, volcano plots, heatmaps), and saves the results to disk. The function handles:

    1. Running DESeq2 differential expression testing
    2. Computing log fold-change shrinkage for better estimates
    3. Annotating genes with identifiers (ENTREZ, SYMBOL)
    4. Creating MA plots and volcano plots to visualize differential expression
    5. Filtering results based on p-values and log fold-change thresholds
    6. Generating heatmaps of differentially expressed genes
    7. Creating Venn diagrams for comparing gene lists across contrasts
    8. Summarizing results for all contrasts and threshold combinations

    Args:
        dds: DESeq2 dataset object created from proc_diff_expr_dataset.
        annot_df_contrasts: Sample annotation dataframe with metadata for each sample,
            indexed by sample ID. Must contain the contrast_factor column.
        results_path: Directory path where differential expression results will be saved.
        plots_path: Directory path where plots will be saved.
        exp_prefix: String prefix for all output files.
        org_db: Organism annotation database object for gene ID mapping.
        contrast_factor: Column name in annot_df_contrasts used to group samples for
            differential expression comparisons (e.g., "sample_type").
        contrasts_levels: List of tuples (test, control) specifying the contrast factor
            levels to compare. Each tuple represents one contrast with test group vs.
            control group.
        contrast_levels_colors: Dictionary mapping contrast factor levels to colors for plots
            (e.g., {"tumor": "red", "normal": "blue"}).
        p_cols: List of p-value columns to use for filtering (e.g., ["pvalue", "padj"]).
        p_ths: List of p-value thresholds to apply for filtering (e.g., [0.05, 0.01]).
        lfc_levels: List of log fold-change filtering criteria to apply ("up", "down", "all").
        lfc_ths: List of log fold-change threshold values to apply (e.g., [0, 1, 2]).
        vst: Optional variance-stabilized transformed dataset for visualization.
            If None, the raw dds will be used.
        heatmap_top_n: Maximum number of top DEGs to include in each heatmap.

    Returns:
        None. All results are written to disk as CSV files and visualizations as PDF/HTML:
        - Unfiltered DESeq2 results for each contrast
        - Filtered results for each combination of p-value column, threshold, and LFC criteria
        - MA plots showing log fold-change vs. mean expression
        - Volcano plots showing significance vs. log fold-change
        - Venn diagrams comparing DEGs across contrasts (when multiple contrasts provided)
        - Heatmaps of top differentially expressed genes
        - Summary statistics of DEG counts

    Examples:
        >>> proc_diff_expr_results(
        ...     dds=dds_object,
        ...     annot_df_contrasts=sample_annotations,
        ...     results_path=Path("/results/deseq2"),
        ...     plots_path=Path("/results/plots"),
        ...     exp_prefix="experiment1",
        ...     org_db=org_db_object,
        ...     contrast_factor="sample_type",
        ...     contrasts_levels=[("tumor", "normal"), ("metastasis", "tumor")],
        ...     contrast_levels_colors={"normal": "blue", "tumor": "red", "metastasis": "purple"},
        ...     p_cols=["padj"],
        ...     p_ths=[0.05],
        ...     lfc_levels=["all"],
        ...     lfc_ths=[1.0],
        ...     vst=vst_object,
        ...     heatmap_top_n=100
        ... )
    """
    # 1. Run DESeq2
    with localconverter(ro.default_converter):
        deseq = run_dseq2(dds)

    # 2. Get results for each contrast
    results = {}
    for test, control in contrasts_levels:
        # [factor, active (numerator), baseline (denominator)]
        contrast = ro.StrVector(
            [re.sub(r"\W|^(?=\d)", "_", contrast_factor), test, control]
        )
        with localconverter(ro.default_converter):
            results[(test, control)] = lfc_shrink(
                dds=deseq, contrast=contrast, type="ashr"
            )

    # 3. MA Plot
    for (test, control), result in results.items():
        # [factor, active (numerator), baseline (denominator)]
        contrast = [re.sub(r"\W|^(?=\d)", "_", contrast_factor), test, control]

        save_path = plots_path.joinpath(f"{exp_prefix}_{test}_vs_{control}_ma_plot.pdf")
        with localconverter(ro.default_converter):
            ma_plot(result, save_path=save_path, contrast=ro.StrVector(contrast))

    # 4. Annotate results and save to disk
    with localconverter(ro.default_converter):
        results_anno = {
            k: annotate_deseq_result(deseq_result, org_db)
            for k, deseq_result in results.items()
        }
    for (test, control), deseq_result in results_anno.items():
        save_path = results_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_deseq_results.csv"
        )
        deseq_result.to_csv(save_path)

    # 4.1. Keep only transcripts uniquely mapped and without shared ENTREZ IDs
    results_anno_unique = {
        k: (
            deseq_result[
                ~deseq_result["ENTREZID"].str.contains("/", na=False)
                | ~deseq_result["SYMBOL"].str.contains("/", na=False)
            ]
            .dropna(subset=["ENTREZID", "SYMBOL"])
            .drop_duplicates(subset=["ENTREZID", "SYMBOL"], keep=False)
        )
        for k, deseq_result in results_anno.items()
    }
    for (test, control), deseq_result in results_anno_unique.items():
        save_path = results_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_deseq_results_unique.csv"
        )
        deseq_result.to_csv(save_path)

    # 5. Volcano plot
    for (test, control), result in results_anno_unique.items():
        labels = result["SYMBOL"].to_list()
        save_path = plots_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_volcano_plot.pdf"
        )
        with localconverter(ro.default_converter):
            volcano_plot(
                data=pd_df_to_rpy2_df(result[["log2FoldChange", "padj"]]),
                lab=ro.StrVector(labels),
                x="log2FoldChange",
                y="padj",
                save_path=save_path,
                title=f"{test} vs {control}",
            )

    # 6. Filter results
    results_filtered = {}
    for ((test, control), result), p_col, p_th, lfc_level, lfc_thr in product(
        results_anno.items(), p_cols, p_ths, lfc_levels, lfc_ths
    ):
        results_filtered[(test, control, p_col, p_th, lfc_level, lfc_thr)] = (
            filter_deseq_results(result, p_col, p_th, lfc_level, lfc_thr)
        )
    for (
        test,
        control,
        p_col,
        p_th,
        lfc_level,
        lfc_thr,
    ), result in results_filtered.items():
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_thr).replace(".", "_")
        save_path = results_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_"
            f"{lfc_level}_{lfc_thr_str}_deseq_results.csv"
        )
        result.sort_values("log2FoldChange").to_csv(save_path)

    # 6.1. Keep only transcripts uniquely mapped and without shared ENTREZIDs
    results_filtered_unique = {}
    for ((test, control), result), p_col, p_th, lfc_level, lfc_thr in product(
        results_anno_unique.items(), p_cols, p_ths, lfc_levels, lfc_ths
    ):
        results_filtered_unique[(test, control, p_col, p_th, lfc_level, lfc_thr)] = (
            filter_deseq_results(result, p_col, p_th, lfc_level, lfc_thr)
        )
    for (
        test,
        control,
        p_col,
        p_th,
        lfc_level,
        lfc_thr,
    ), result in results_filtered_unique.items():
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_thr).replace(".", "_")
        save_path = results_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_"
            f"{lfc_level}_{lfc_thr_str}_deseq_results_unique.csv"
        )
        result.sort_values("log2FoldChange").to_csv(save_path)

    # 7. VENN diagram
    if 1 < len(contrasts_levels) < 6:
        for p_col, p_th, lfc_level, lfc_thr in product(
            p_cols, p_ths, lfc_levels, lfc_ths
        ):
            contrast_degs = {}
            for test, control in contrasts_levels:
                result = results_filtered_unique[
                    (test, control, p_col, p_th, lfc_level, lfc_thr)
                ]
                contrast_degs[f"{test} vs {control}"] = result["ENTREZID"].tolist()

            contrast_degs = {k: v for k, v in contrast_degs.items() if len(v) > 0}

            if len(contrast_degs) <= 1:
                continue

            p_thr_str = str(p_th).replace(".", "_")
            lfc_thr_str = str(lfc_thr).replace(".", "_")
            save_path = plots_path.joinpath(
                f"{exp_prefix}_{p_col}_{p_thr_str}_{lfc_level}_{lfc_thr_str}_venn.pdf"
            )

            with localconverter(ro.default_converter):
                venn_diagram(
                    contrast_degs,
                    save_path,
                    main=f"{exp_prefix}",
                    sub=f"{p_col} < {p_th}, LFC={lfc_level}, >{lfc_thr}",
                    cex=2,
                    fontface="bold",
                    fill=(
                        ro.StrVector(["red", "blue", "green"])
                        if len(contrasts_levels) == 3
                        else ro.NULL
                    ),
                )

    # 8. Gene clustering (supervised heatmap)
    for (
        test,
        control,
        p_col,
        p_th,
        lfc_level,
        lfc_thr,
    ), result in results_filtered_unique.items():
        if result.empty:
            continue

        # 8.1. Load and annotate VST matrix
        with localconverter(ro.default_converter):
            counts_df = assay_to_df(vst if vst else dds)
        # get SYMBOL IDs of common transcripts
        counts_df = counts_df.loc[counts_df.index.intersection(result.index)]
        counts_df.index = result.loc[counts_df.index, "SYMBOL"].values

        # 8.2. Get samples belonging to the comparison (test, control), sort by
        # factor category
        annot_df_test_control = (
            annot_df_contrasts[
                annot_df_contrasts[contrast_factor].isin([test, control])
            ][contrast_factor]
            .sort_values()
            .to_frame()
        )

        # 8.3. Filter counts_matrix to include only relevant samples
        common_samples = annot_df_test_control.index.intersection(
            counts_df.columns
        )  # ensure we use only available samples (annotated could be more)
        annot_df_test_control = annot_df_test_control.loc[common_samples, :]
        counts_df_contrast = counts_df.loc[:, common_samples]

        # 8.4. Sort genes by LFC
        top_n = min(heatmap_top_n, len(result))

        # filter result to include only SYMBOL IDs with VST expression values
        result = result[result["SYMBOL"].isin(counts_df.index)]
        filtered_genes = result.sort_values(
            ["log2FoldChange", p_col], ascending=[False, True], key=abs
        ).dropna(subset=["SYMBOL"])[:top_n]

        counts_matrix = counts_df_contrast.loc[filtered_genes["SYMBOL"].tolist(), :]

        # substract row means fot better visualization
        counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

        # 8.5. Define heatmap annotation
        p_thr_str = str(p_th).replace(".", "_")
        lfc_thr_str = str(lfc_thr).replace(".", "_")
        save_path = plots_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_"
            f"{lfc_level}_{lfc_thr_str}_supervised_genes_clustering_vst.pdf"
        )
        with localconverter(ro.default_converter):
            ha_column = heatmap_annotation(
                df=annot_df_test_control[[contrast_factor]],
                col={contrast_factor: contrast_levels_colors},
                show_annotation_name=False,
            )

        # 8.6. Plot heatmap
        with localconverter(ro.default_converter):
            complex_heatmap(
                counts_matrix,
                save_path=save_path,
                width=10,
                column_title=(
                    f"Top {top_n} DEGs in {contrast_factor} (LFC > {lfc_thr}, {p_col} <"
                    f" {p_th}) (VST)"
                ),
                name="Gene Expression (VST)",
                top_annotation=ha_column,
                heatmap_legend_param=ro.r(
                    'list(title_position = "topcenter", color_bar = "continuous",'
                    ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
                ),
            )

        # 8.7. Plot heatmap (for publishing)
        save_path = plots_path.joinpath(
            f"{exp_prefix}_{test}_vs_{control}_{p_col}_{p_thr_str}_"
            f"{lfc_level}_{lfc_thr_str}_supervised_genes_clustering_vst_pub.pdf"
        )
        with localconverter(ro.default_converter):
            complex_heatmap(
                counts_matrix,
                save_path=save_path,
                width=10,
                height=8,
                column_title=(
                    f"Top {top_n} DEGs in {contrast_factor} (LFC > {lfc_thr}, {p_col} <"
                    f" {p_th}) (VST)"
                ),
                name="Gene Expression (VST)",
                top_annotation=ha_column,
                show_row_names=False,
                show_column_names=False,
                cluster_columns=False,
                heatmap_legend_param=ro.r(
                    'list(title_position = "topcenter", color_bar = "continuous",'
                    ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
                ),
            )

    # 9. Summary statistics
    summary_degs = defaultdict(dict)
    for (
        test,
        control,
        p_col,
        p_th,
        lfc_level,
        lfc_thr,
    ), result in results_filtered.items():
        summary_degs[(test, control, p_col, p_th, lfc_thr)][lfc_level] = len(result)
    summary_degs_df = pd.DataFrame(summary_degs).transpose()
    summary_degs_df.to_csv(results_path.joinpath(f"{exp_prefix}_degs_summary.csv"))


def differential_expression(
    annot_df_contrasts: pd.DataFrame,
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    org_db: OrgDB,
    factors: Iterable[str],
    contrast_factor: str,
    contrasts_levels: Iterable[Tuple[str, str]],
    contrast_levels_colors: Dict[str, str],
    p_cols: Iterable[str],
    p_ths: Iterable[float],
    lfc_levels: Iterable[str],
    lfc_ths: Iterable[float],
    design_factors: Iterable[str] = None,
    heatmap_top_n: int = 1000,
    counts_matrix: Optional[pd.DataFrame] = None,
    counts_path: Optional[Path] = None,
    counts_files_pattern: str = "*.tsv",
    compute_vst: bool = True,
    compute_rlog: bool = False,
) -> None:
    """
    Run a complete differential expression analysis workflow.

    This is a high-level function that orchestrates the entire differential expression
    analysis workflow, from dataset creation to result generation. It combines the
    functionality of proc_diff_expr_dataset and proc_diff_expr_results to create a
    streamlined pipeline for analyzing RNA-seq data with DESeq2.

    In a DESeq2 differential expression analysis, there are mainly two steps:
        1. Generate the DESeq2 datasets and its transformations (e.g., VST, RLD) and
           plot various visualizations. Here we only need to know which samples (with
           existing counts files) need to be taken into account. For example, only
           samples for the intended comparisons should be included.
           Only has to be run once per experiment (for the same samples).

        2. Calculate differentially expressed genes (DESeq2 results) for interesting
           contrasts and generate multiple gene lists according to different criteria.
           Has to be run once per contrast comparison, hence can be parallelized.

    A full differential expression run considers only one contrast factor (e.g., comparisons
    between levels of a single annotation field, such as sample type) but can include
    multiple pairwise comparisons within that factor.

    Args:
        annot_df_contrasts: Sample annotation dataframe indexed by sample IDs, containing
            metadata columns including the contrast_factor column for grouping samples.
        results_path: Directory path where all analysis results will be saved.
        plots_path: Directory path where all visualization plots will be saved.
        exp_prefix: String prefix for all generated files and plot titles.
        org_db: Organism annotation database object for gene ID conversion.
        factors: List of columns from annot_df_contrasts to be included in the DESeq2 dataset.
        contrast_factor: Column name in annot_df_contrasts used for grouping samples and
            defining contrasts for differential expression analysis.
        contrasts_levels: List of tuples (test_group, reference_group) specifying the
            contrasts to analyze, where each group is a level of the contrast_factor.
            Example: [("tumor", "normal"), ("metastasis", "tumor")]
        contrast_levels_colors: Dictionary mapping factor levels to colors for visualization.
            Example: {"tumor": "red", "normal": "blue", "metastasis": "purple"}
        p_cols: List of p-value columns to use for filtering results (e.g., ["pvalue", "padj"]).
        p_ths: List of p-value thresholds to apply when filtering (e.g., [0.05, 0.01]).
        lfc_levels: List of log fold-change filtering categories ("up", "down", "all").
        lfc_ths: List of log fold-change threshold values (e.g., [0, 1, 2]).
        design_factors: Optional list of factors to include in the design formula for
            differential analysis. If None, defaults to [contrast_factor].
        heatmap_top_n: Maximum number of top differentially expressed genes to include
            in heatmaps. Default is 1000.
        counts_matrix: Optional pre-computed counts matrix with genes in rows and
            samples in columns. If provided, counts_path is ignored.
        counts_path: Optional path to directory containing count files (one per sample).
            Used only if counts_matrix is None.
        counts_files_pattern: File pattern for count files. Used only with counts_path.
            Default is "*.tsv".
        compute_vst: Whether to compute variance-stabilizing transformation. Default is True.
        compute_rlog: Whether to compute regularized log transformation. More accurate
            than VST for small sample sizes but slower to compute. Default is False.

    Returns:
        None. All results are written to the specified output directories as CSV files,
        and visualizations are saved as PDF/HTML files.

    Examples:
        >>> # Basic differential expression analysis
        >>> differential_expression(
        ...     annot_df_contrasts=sample_annotations,
        ...     results_path=Path("/results/deseq2"),
        ...     plots_path=Path("/results/plots"),
        ...     exp_prefix="experiment1",
        ...     org_db=org_db_object,
        ...     factors=["sample_type", "batch"],
        ...     contrast_factor="sample_type",
        ...     contrasts_levels=[("tumor", "normal")],
        ...     contrast_levels_colors={"tumor": "red", "normal": "blue"},
        ...     p_cols=["padj"],
        ...     p_ths=[0.05],
        ...     lfc_levels=["all", "up", "down"],
        ...     lfc_ths=[1.0],
        ...     counts_path=Path("/data/counts")
        ... )
        >>>
        >>> # More complex analysis with multiple contrasts and thresholds
        >>> differential_expression(
        ...     annot_df_contrasts=sample_annotations,
        ...     results_path=Path("/results/deseq2"),
        ...     plots_path=Path("/results/plots"),
        ...     exp_prefix="experiment2",
        ...     org_db=org_db_object,
        ...     factors=["sample_type", "patient_id", "batch"],
        ...     contrast_factor="sample_type",
        ...     contrasts_levels=[("tumor", "normal"), ("metastasis", "tumor")],
        ...     contrast_levels_colors={
        ...         "normal": "blue", "tumor": "red", "metastasis": "purple"
        ...     },
        ...     p_cols=["pvalue", "padj"],
        ...     p_ths=[0.05, 0.01],
        ...     lfc_levels=["all", "up", "down"],
        ...     lfc_ths=[0, 1, 2],
        ...     design_factors=["sample_type", "batch"],
        ...     counts_matrix=counts_df,
        ...     compute_vst=True,
        ...     compute_rlog=True
        ... )
    """
    # 1. Process DESeq2 dataset
    dds, vst, _ = proc_diff_expr_dataset(
        annot_df_contrasts=deepcopy(annot_df_contrasts),
        counts_matrix=counts_matrix,
        counts_path=counts_path,
        results_path=results_path,
        plots_path=plots_path,
        exp_prefix=exp_prefix,
        org_db=org_db,
        factors=factors,
        contrast_factor=contrast_factor,
        contrast_levels_colors=contrast_levels_colors,
        design_factors=design_factors,
        heatmap_top_n=heatmap_top_n,
        counts_files_pattern=counts_files_pattern,
        compute_vst=compute_vst,
        compute_rlog=compute_rlog,
    )

    # 2. Process DESeq2 results
    proc_diff_expr_results(
        dds=dds,
        annot_df_contrasts=deepcopy(annot_df_contrasts),
        results_path=results_path,
        plots_path=plots_path,
        exp_prefix=exp_prefix,
        org_db=org_db,
        contrast_factor=contrast_factor,
        contrasts_levels=contrasts_levels,
        p_cols=p_cols,
        p_ths=p_ths,
        lfc_levels=lfc_levels,
        lfc_ths=lfc_ths,
        contrast_levels_colors=contrast_levels_colors,
        vst=vst,
        heatmap_top_n=heatmap_top_n,
    )
