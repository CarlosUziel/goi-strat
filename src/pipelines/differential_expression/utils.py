import logging
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
import plotly.express as px
import rpy2.robjects as ro
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
    """Process differential expression dataset plots.

    Args:
        dataset: Dataset object.
        dataset_label: Dataset label.
        annot_df_contrasts: Samples annotation dataframe.
        plots_path: Path to store all generated plots.
        exp_prefix: A string prefixing generated files.
        org_db: Organism annotation database.
        contrast_factor: Annotation field used for differential analysis.
        contrast_levels_colors: Colors used to plot contrast factor levels.
        heatmap_top_n: Top number of rows to show in the heatmap.
    """
    # 1. Samples clustering
    sample_dist = sample_distance(dataset)

    save_path = plots_path.joinpath(
        f"{exp_prefix}_samples_clustering_{dataset_label}.pdf"
    )
    heatmap_sample_distance(sample_dist, save_path)

    # 2. Genes clustering (unsupervised)
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
    save_path = plots_path.joinpath(f"{exp_prefix}_mean_sd_plot_{dataset_label}.pdf")
    mean_sd_plot(
        norm_transform(dataset) if dataset_label == "dds" else dataset, save_path
    )

    # 4. Principal Component Analysis (PCA)
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
    sample_dist = sample_distance(dataset)
    save_path = plots_path.joinpath(f"{exp_prefix}_MDS_{dataset_label}.pdf")
    mds_plot(sample_dist, dataset, color=contrast_factor, save_path=save_path)


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
    Generate DESeq2 dataset plus multiple plots. Optionally, compute normalized datasets
    and their corresponding plots.

    Args:
        annot_df_contrasts: Annotation file of samples to be analysed.
        results_path: Path to store intermediate and final results to.
        plots_path: Path to store intermediate and final plots to.
        exp_prefix: A string prefixing generated files.
        org_db: Organism annotation database object.
        factors: Columns to be added to the deseq dataset.
        contrast_factor: Annotation field used for differential analysis.
        contrast_levels_colors: Colors used to plot contrast factor levels.
        design_factors: Factors to include in the design formula for differential
            analysis.
        heatmap_top_n: Top number of genes to include in heatmaps.
        counts_matrix: Optionally, include an already-computed counts matrix.
        counts_path: Optionally, include path to load gene raw counts from.
        counts_files_pattern: Optionally, include pattern of gene raw counts files.
        compute_vst: Whether to compute VST-normalized gene counts.
        compute_rlog: Whether to compute RLOG-normalized gene counts.

    Raises:
        ValueError: _description_

    Returns:
        Tuple[Any, Any, Any]: _description_
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
        dds = get_deseq_dataset_htseq(
            annot_df=annot_df_contrasts,
            counts_path=counts_path,
            factors=factors,
            design_factors=design_factors,
            counts_files_pattern=counts_files_pattern,
        )
    elif counts_matrix is not None:
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
    save_path = results_path.joinpath(f"{exp_prefix}_dds")
    rpy2_df_to_pd_df(ro.r("counts")(dds)).to_csv(save_path.with_suffix(".csv"))
    save_rds(dds, save_path.with_suffix(".RDS"))

    # 1.4. Dataset visualizations
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
        # 2.1. Get dataset
        vst = vst_transform(dds)

        # 2.2. Save to disk
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
        # 3.1. Get dataset
        rld = rlog_transform(dds)

        # 3.2. Save to disk
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
    Get differentially expressed genes, filter by different criteria, save results to
    disk and do some plots.

    Args:
        dds: DDS dataset.
        annot_df_contrasts: Annotation file of samples to be analysed.
        results_path: Path to store intermediate and final results to.
        plots_path: Path to store intermediate and final plots to.
        exp_prefix: A string prefixing generated files.
        org_db: Organism annotation database object.
        contrast_factor: Annotation field used for differential analysis.
        contrasts_levels: Contrast factor levels to compare in differential analysis.
        contrast_levels_colors: Colors used to plot contrast factor levels.
        p_cols: P-value field columns to filter by.
        p_ths: P-value thresholds to filter by.
        lfc_levels: LFC levels (up, down, all) to filter by.
        lfc_ths: LFC thresholds to filter by.
        vst: VST dataset, optional.
        heatmap_top_n: Top number of genes to include in heatmaps.
    """

    # 1. Run DESeq2
    deseq = run_dseq2(dds)

    # 2. Get results for each contrast
    results = {}
    for test, control in contrasts_levels:
        # [factor, active (numerator), baseline (denominator)]
        contrast = ro.StrVector([contrast_factor, test, control])

        results[(test, control)] = lfc_shrink(dds=deseq, contrast=contrast, type="ashr")

    # 3. MA Plot
    for (test, control), result in results.items():
        # [factor, active (numerator), baseline (denominator)]
        contrast = [contrast_factor, test, control]

        save_path = plots_path.joinpath(f"{exp_prefix}_{test}_vs_{control}_ma_plot.pdf")
        ma_plot(result, save_path=save_path, contrast=ro.StrVector(contrast))

    # 4. Annotate results and save to disk
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
        ha_column = heatmap_annotation(
            df=annot_df_test_control[[contrast_factor]],
            col={contrast_factor: contrast_levels_colors},
            show_annotation_name=False,
        )

        # 8.6. Plot heatmap
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
    counts_matrix: Optional[Path] = None,
    counts_path: Optional[Path] = None,
    counts_files_pattern: str = ".tsv",
    compute_vst: bool = True,
    compute_rlog: bool = False,
) -> None:
    """
    Run a differential expression experiment.

    In a DESeq2 differential expression analysis, there are mainly two steps:
        1. Generate the DESeq2 datasets and its transformations (e.g., VST, RLD) and
            plot various visualizations. Here we only need to know which samples (with
            existing counts files) need to be taken into account. For example, only
            samples for the intended comparisons should be included.
            Only has to be run once per experiment (for the same samples).

        2. Calculate differentially expressed genes (DESeq2 results) for interesting
            contrasts and generate multiple gene lists according to different criteria.
            Has to be run once per contrast comparison, hence can be parallelized.

    A full differential expression run considers only one contrast (e.g. comparisons
        between levels of a single annotation field, such as sample type). Each GOI run
        is essentially a full independent run.

    Args:
        annot_df_contrasts: Annotation file of samples to be analysed.
        results_path: Path to store intermediate and final results to.
        plots_path: Path to store intermediate and final plots to.
        exp_prefix: A string prefixing generated files.
        org_db: Organism annotation database object.
        factors: Columns to be added to the deseq dataset.
        contrast_factor: Annotation field used for differential analysis.
        contrasts_levels: Contrast factor levels to compare in differential analysis.
        contrast_levels_colors: Colors used to plot contrast factor levels.
        p_cols: P-value field columns to filter by.
        p_ths: P-value thresholds to filter by.
        lfc_levels: LFC levels (up, down, all) to filter by.
        lfc_ths: LFC thresholds to filter by.
        design_factors: Factors to include in the design formula for differential
            analysis.
        heatmap_top_n: Top number of genes to include in heatmaps.
        counts_matrix: Optionally, include an already-computed counts matrix.
        counts_path: Optionally, include path to load gene raw counts from.
        counts_files_pattern: Optionally, include pattern of gene raw counts files.
        compute_vst: Whether to compute VST-normalized gene counts.
        compute_rlog: Whether to compute RLOG-normalized gene counts.
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
