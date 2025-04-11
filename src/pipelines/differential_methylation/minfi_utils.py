# Reference workflow:
#   https://dockflow.org/workflow/methylation-array-analysis/#content

import functools
import logging
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

from data.utils import parallelize_map
from r_wrappers.annotatr import (
    annotate_regions,
    build_annotations,
    plot_annotation,
    summarize_annotations,
)
from r_wrappers.complex_heatmaps import complex_heatmap, heatmap_annotation
from r_wrappers.dmr_cate import cpg_annotate, dmr_cate, dmr_plot
from r_wrappers.limma import (
    decide_tests,
    empirical_bayes,
    fit_contrasts,
    linear_model_fit,
    make_contrasts,
    plot_md,
    top_table,
    venn_diagram,
    volcano_plot,
)
from r_wrappers.maxprobes import drop_xreactive_loci, xreactive_probes
from r_wrappers.minfi import (
    density_bean_plot,
    density_plot,
    density_plot_minfi_pair,
    detection_p,
    detection_p_barplot,
    drop_loci_with_snps,
    get_qc,
    map_to_genome,
    mds_plot,
    pca_scatterplot3d,
    plot_cpgs,
    plot_qc,
    preprocess_funnorm,
    preprocess_illumina,
    preprocess_noob,
    preprocess_quantile,
    preprocess_raw,
    preprocess_swan,
    qc_report,
    read_metharray_exp,
)
from r_wrappers.utils import (
    df_to_file,
    get_design_matrix,
    make_granges_from_dataframe,
    pd_df_to_rpy2_df,
    rpy2_df_to_pd_df,
    save_rds,
)
from utils import run_func_dict


def quality_control(
    rg_set: Any,
    plots_path: Path,
    exp_prefix: str,
    targets: pd.DataFrame,
    contrast_factor: str,
    id_col: str,
) -> Tuple[Any, pd.DataFrame, Any]:
    """
    Perform quality control on an RG Set.

    Args:
        rg_set: A user-provided RG set.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        targets: Samples annotation file.
        contrast_factor: Contrast factor (column of targets) for differential
            methylation.
        contrasts_levels: List of contrasts to test for differential methylation.
        contrasts_levels_colors: A dictionary where each contrast level
            has a color assigned.
        id_col: Column name to uniquely identify samples.

    Returns:
        RG set and targets dataframe with poor quality samples removed plus detection
            p values.
    """
    # 0. Setup
    sample_groups = targets[contrast_factor].tolist()
    sample_names = targets[id_col].tolist()

    # 1. Detection p-values
    det_p = detection_p(rg_set)

    save_path = plots_path.joinpath(f"{exp_prefix}_detection_p_barplot.pdf")
    detection_p_barplot(
        det_p, sample_groups=ro.StrVector(sample_groups), save_path=save_path
    )

    # 2. Sample-specific QC (remove bad samples manually)
    qc_values = get_qc(preprocess_raw(rg_set))

    save_path = plots_path.joinpath(f"{exp_prefix}_samples_qc.pdf")
    plot_qc(qc_values, save_path)

    # 3. QC Report
    save_path = plots_path.joinpath(f"{exp_prefix}_qc_report.pdf")
    qc_report(
        rg_set,
        sampNames=ro.StrVector(sample_names),
        sampGroups=ro.StrVector(sample_groups),
        pdf=str(save_path),
    )

    # 4. Density plot of raw data
    save_path = plots_path.joinpath(f"{exp_prefix}_raw_data_density_plot.pdf")
    density_plot(
        obj=rg_set,
        save_path=save_path,
        sampGroups=ro.StrVector(sample_groups),
        legendPos="top",
        main="Density plot of raw data",
    )

    # 5. Violin plot of raw data
    save_path = plots_path.joinpath(f"{exp_prefix}_raw_violin_plot.pdf")
    density_bean_plot(
        rg_set,
        save_path=save_path,
        sampGroups=ro.StrVector(sample_groups),
        sampNames=ro.StrVector(sample_names),
        main="Violin plot of raw data",
    )

    # 6. Remove poor quality samples
    samples_keep = [i for i, x in enumerate(ro.r("colMeans")(det_p)) if x < 0.05]
    det_p = det_p.rx(True, ro.IntVector([x + 1 for x in samples_keep]))
    rg_set = ro.r("f <- function(x, y){ return(x[,y]) }")(
        rg_set, ro.IntVector([x + 1 for x in samples_keep])
    )
    targets = targets.iloc[samples_keep, :]

    return rg_set, targets, det_p


def normalization(
    rg_set: Any,
    norm_type: str,
    results_path: Path,
    plots_path: Path,
    sample_names: Iterable[str],
    sample_groups: Iterable[str],
    exp_prefix: str,
) -> Tuple[str, Any]:
    """
    Normalize RG set objects.

    Args:
        rg_set: A user-provided RG set.
        norm_type: Normalization type.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        sample_names: Name of each sample.
        sample_groups: Name of the group each sample belongs to.

    Returns:
        Normalization type, normalized RG set object.
    """

    # 1. Apply normalization type to RG set
    if norm_type == "raw":
        norm_set = preprocess_raw(rg_set)
    elif norm_type == "funnorm":
        norm_set = preprocess_funnorm(rg_set)
    elif norm_type == "illumina":
        norm_set = preprocess_illumina(rg_set, normalize="controls")
    elif norm_type == "noob_reference":
        norm_set = preprocess_noob(rg_set, dyeMethod=ro.StrVector(["reference"]))
    elif norm_type == "noob_single":
        norm_set = preprocess_noob(rg_set, dyeMethod=ro.StrVector(["single"]))
    elif norm_type == "swan":
        norm_set = preprocess_swan(rg_set)
    elif norm_type == "quantile":
        norm_set = preprocess_quantile(rg_set)
    elif norm_type == "noob_quantile":
        norm_set = preprocess_quantile(preprocess_noob(rg_set))
    else:
        raise ValueError(f"Normalization type not supported ({norm_type})")

    # 2. Save Methylated/Unmethylated/M/B values
    save_rds(norm_set, results_path.joinpath(f"{exp_prefix}_{norm_type}.RDS"))
    for func, data_type in [
        (ro.r("getMeth"), "meth_matrix"),
        (ro.r("getUnmeth"), "unmeth_matrix"),
        (ro.r("getM"), "m_values"),
        (ro.r("getBeta"), "b_values"),
    ]:
        try:
            values = func(norm_set)
        except Exception:
            continue

        save_path = results_path.joinpath(
            f"{exp_prefix}_{norm_type}_{data_type}.csv.gz"
        )
        if not save_path.exists():
            rpy2_df_to_pd_df(values).to_csv(save_path)

    # 3. MDS plot
    # 3.1. With sample names
    save_path = plots_path.joinpath(f"{exp_prefix}_mds_plot_{norm_type}.pdf")
    mds_plot(
        obj=ro.r("getBeta")(norm_set),
        save_path=save_path,
        sampNames=ro.StrVector(sample_names),
        sampGroups=ro.StrVector(sample_groups),
        legendPos="top",
        main=f"MDS plot ({norm_type})",
        pch=19,
    )

    # 3.2. Without sample names
    save_path = plots_path.joinpath(f"{exp_prefix}_mds_plot_{norm_type}_no_names.pdf")
    mds_plot(
        obj=ro.r("getBeta")(norm_set),
        save_path=save_path,
        sampGroups=ro.StrVector(sample_groups),
        legendPos="top",
        main=f"MDS plot ({norm_type})",
        pch=19,
    )

    # 4. Density plot
    save_path = plots_path.joinpath(f"{exp_prefix}_density_plot_{norm_type}.pdf")
    density_plot(
        obj=ro.r("getBeta")(norm_set),
        save_path=save_path,
        sampGroups=ro.StrVector(sample_groups),
        legendPos="top",
        main=f"Density plot ({norm_type})",
    )

    # 5. Raw vs. Normalized density plot
    save_path = plots_path.joinpath(
        f"{exp_prefix}_density_plot_{norm_type}_raw_vs_norm.pdf"
    )
    density_plot_minfi_pair(
        obj_0=rg_set,
        obj_1=ro.r("getBeta")(norm_set),
        sample_groups=ro.StrVector(sample_groups),
        save_path=save_path,
        title_0="Raw",
        title_1=f"Normalized using {norm_type}",
        x_lab_0="B-values",
        x_lab_1="B-values",
    )

    return norm_type, norm_set


def filtering(
    mset: Any,
    norm_type: str,
    det_p: Any,
    genome_anno: pd.DataFrame,
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    sample_names: Iterable[str],
    sample_groups: Iterable[str],
    contrasts_levels_colors: Dict[str, str],
    array_type: str = "450K",
) -> Tuple[str, Any]:
    """
    Filter probes in methylation set.

    Args:
        mset: Normalized methylation set object.
        norm_type: Normalized type.
        det_p: Quality control detection P-values.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        sample_names: Name of each sample.
        sample_groups: Name of the group each sample belongs to.
        contrasts_levels_colors: A dictionary where each contrast level
            has a color assigned.
        array_type: Methylation array type.

    Returns:
        Normalization type, filtered RG set object.
    """
    # 1. Ensure probes are in the same order in the norm_mset and det_p objects
    det_p = det_p.rx(mset.names, True)

    # 2. Filter out probes that have failed in one or more samples based on detection
    # p-value
    det_p_df = rpy2_df_to_pd_df(det_p)
    mset = ro.r("f <- function(x, y){ return(x[y, ]) }")(
        mset, ro.StrVector(det_p_df.loc[(det_p_df < 0.05).all(axis=1)].index)
    )

    # 3. If data includes males and females, remove probes on the sex chromosomes
    mset = ro.r("f <- function(x, y){ return(x[y, ]) }")(
        mset,
        ro.StrVector(
            genome_anno[
                ~genome_anno["CpG_chrm"].isin(("chrX", "chrY"))
            ].index.intersection(set(mset.names))
        ),
    )

    # 4. Remove probes with SNPs at CpG sites
    mset = drop_loci_with_snps(map_to_genome(mset))

    # 5. Exclude cross reactive probes
    xreactive_probes(array_type=array_type)
    mset = drop_xreactive_loci(mset)

    # 6. Save results after filtering
    save_rds(
        mset,
        results_path.joinpath(f"{exp_prefix}_{norm_type}_filt.RDS"),
    )

    # 6.1. M-values
    rpy2_df_to_pd_df(ro.r("getM")(mset)).to_csv(
        results_path.joinpath(f"{exp_prefix}_{norm_type}_filt_m_values.csv.gz")
    )

    # 6.2. Beta-values
    rpy2_df_to_pd_df(ro.r("getBeta")(mset)).to_csv(
        results_path.joinpath(f"{exp_prefix}_{norm_type}_filt_beta_values.csv.gz")
    )

    # 7. Plot data after filtering
    save_path = plots_path.joinpath(f"{exp_prefix}_mds_plot_{norm_type}_filtered.pdf")
    mds_plot(
        obj=ro.r("getM")(mset),
        save_path=save_path,
        sampNames=ro.StrVector(sample_names),
        sampGroups=ro.StrVector(sample_groups),
        legendPos="top",
        main=f"MDS plot using {norm_type}",
        pch=19,
    )

    # 8. 3D PCA Scatter plot of M-values (log2(M/U))
    save_path = plots_path.joinpath(
        f"{exp_prefix}_PCA_scatterplot3d_M_values_{norm_type}_filtered.pdf"
    )
    pca_scatterplot3d(
        data=ro.r("getM")(mset),
        sample_colors=ro.StrVector([contrasts_levels_colors[x] for x in sample_groups]),
        legend_groups=ro.StrVector(set(sample_groups)),
        legend_colors=ro.StrVector(
            [contrasts_levels_colors[x] for x in set(sample_groups)]
        ),
        save_path=save_path,
        pch=19,
        cex_symbols=1.5,
        main=f"3D M-values PCA - {norm_type} (filtered)",
        grid=True,
        box=False,
        col_grid="grey",
        lty_grid=ro.r("par")("lty"),
        xlab="PC1",
        ylab="PC2",
        zlab="PC3",
    )

    # 9. Density plot comparing filter B-values and M-values
    save_path = plots_path.joinpath(
        f"{exp_prefix}_density_plot_b_values_vs_m_values_{norm_type}.pdf"
    )
    density_plot_minfi_pair(
        obj_0=ro.r("getM")(mset),
        obj_1=ro.r("getBeta")(mset),
        sample_groups=ro.StrVector(sample_groups),
        save_path=save_path,
        title_0="B-values",
        title_1="M-values",
        x_lab_0="B-values",
        x_lab_1="M-values",
    )

    return norm_type, mset


def get_diff_meth_probes(
    m_values: Any,
    norm_type: str,
    targets: pd.DataFrame,
    id_col: str,
    contrast_factor: str,
    contrasts_levels: List[Tuple[str, str]],
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
) -> Tuple[Any, Any, Any]:
    """
    Get differentially methylated probes.

    Args:
        m_values: Methylation values.
        norm_type: Normalization type.
        targets: Samples annotation file.
        id_col: Column name to uniquely identify samples.
        contrast_factor: Contrast factor (column of targets) for differential
            methylation.
        contrasts_levels: List of contrasts to test for differential methylation.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.

    Returns:
        Contrasts fit, design matrix, contrast matrix
    """
    # 1. Get differentially methylated probes (DMPs)
    design_matrix = get_design_matrix(targets, [contrast_factor], id_col=id_col)

    lm_fitted = linear_model_fit(m_values, design_matrix)

    contrast_matrix = make_contrasts(
        contrasts=ro.StrVector(
            [f"{test}-{control}" for test, control in contrasts_levels]
        ),
        levels=design_matrix,
    )

    contrasts_fit = empirical_bayes(
        fit=fit_contrasts(fit=lm_fitted, contrasts=contrast_matrix)
    )

    save_path = results_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_coefficients.csv"
    )
    df_to_file(contrasts_fit.rx2("coefficients"), save_path)

    save_path = results_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_pvalue.csv"
    )
    df_to_file(contrasts_fit.rx2("p.value"), save_path)

    diff_meth_results = decide_tests(contrasts_fit)

    save_path = results_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_results.csv"
    )
    df_to_file(diff_meth_results, save_path)

    save_path = results_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_results_summary.csv"
    )
    df_to_file(ro.r("summary")(diff_meth_results), save_path)

    # 2. Plot results
    # 2.1. VENN diagram
    save_path = plots_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_results_venn_diagram.pdf"
    )
    venn_diagram(
        diff_meth_results,
        save_path,
        include=ro.StrVector(["up", "down"]),
        counts_col=ro.StrVector(["red", "blue"]),
        cex=ro.IntVector([1]),
        circle_col=ro.StrVector(["red", "blue", "green3"]),
    )

    # 2.2. Volcano plot
    save_path = plots_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_contrast_fit_volcano_plot.pdf"
    )
    volcano_plot(contrasts_fit, save_path, coef=1, highlight=30)

    # 2.3. Mean difference (MD) plot
    save_path = plots_path.joinpath(
        f"{exp_prefix}_diff_meth_probes_{norm_type}_contrast_fit_md_plot.pdf"
    )
    plot_md(contrasts_fit, save_path, column=1)

    return contrasts_fit, design_matrix, contrast_matrix


def diff_meth_probes(
    mset: Any,
    norm_type: str,
    targets: pd.DataFrame,
    id_col: str,
    contrast_factor: str,
    contrasts_levels: List[Tuple[str, str]],
    contrasts_levels_colors: Dict[str, str],
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    genome_anno: pd.DataFrame,
    sample_groups: Iterable[str],
    p_cols: Iterable[str] = ("P.Value", "adj.P.Val"),
    p_ths: Iterable[float] = (0.05, 0.01),
    lfc_levels: Iterable[str] = ("hyper", "hypo", "all"),
    lfc_ths: Iterable[float] = (0.0, 1.0, 2.0),
    mean_meth_diff_ths: Iterable[float] = (0.1, 0.2, 0.3),
    genome: str = "hg38",
    heatmap_top_n: int = 1000,
) -> Tuple[str, Tuple[Any, Any]]:
    """
    Get differentially methylated probes, extract significant ones using multiple
    criteria and annotate them.

    Args:
        mset: Normalized methylation set object.
        norm_type: Normalization type.
        targets: Samples annotation file.
        id_col: Column name to uniquely identify samples.
        contrast_factor: Contrast factor (column of targets) for differential
            methylation.
        contrasts_levels: List of contrasts to test for differential methylation.
        contrasts_levels_colors: A dictionary where each contrast level
            has a color assigned.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        genome_anno: Genome annotation dataframe.
        sample_groups: Name of the group each sample belongs to.
        p_cols: P-value column names used for filtering.
        p_ths: P-value thresholds to determine significance.
        lfc_levels: logFC levels to subset results by.
        lfc_ths: logFC thresholds to subset results by.
        mean_meth_diff_ths: Mean methylation thresholds to subset results by.
        genome: Genome version.
        heatmap_top_n: Number of top differentially methylated probes to include in
            heatmap plots.

    Returns:
        Normalization type, design matrix, contrast matrix.
    """
    # 0. Setup
    m_values = ro.r("getM")(mset)
    b_values = ro.r("getBeta")(mset)
    assert all([lfc_level in ("hyper", "hypo", "all") for lfc_level in lfc_levels]), (
        'only "hyper", "hypo" and "all" are allowed lfc_level values'
    )
    assert all([p_col in ("P.Value", "adj.P.Val") for p_col in p_cols]), (
        'only "P.Value" and "adj.P.Val" are allowed p_col values'
    )
    annots = {
        "cpg": build_annotations(genome=genome, annotations=f"{genome}_cpgs"),
        "gene": build_annotations(genome=genome, annotations=f"{genome}_basicgenes"),
    }
    annots_order = ro.StrVector(
        [
            f"{genome}_genes_1to5kb",
            f"{genome}_genes_promoters",
            f"{genome}_genes_5UTRs",
            f"{genome}_genes_exons",
            f"{genome}_genes_intronexonboundaries",
            f"{genome}_genes_introns",
            f"{genome}_genes_3UTRs",
            f"{genome}_genes_intergenic",
            f"{genome}_cpg_inter",
            f"{genome}_cpg_islands",
            f"{genome}_cpg_shelves",
            f"{genome}_cpg_shores",
        ]
    )

    # 1. Get differentially methylated probes (DMPs)
    contrasts_fit, design_matrix, contrast_matrix = get_diff_meth_probes(
        m_values=m_values,
        targets=targets,
        id_col=id_col,
        contrast_factor=contrast_factor,
        contrasts_levels=contrasts_levels,
        results_path=results_path,
        plots_path=plots_path,
        exp_prefix=exp_prefix,
        norm_type=norm_type,
    )

    # 2. Get table of results for each contrast
    m_values_ann = pd_df_to_rpy2_df(rpy2_df_to_pd_df(m_values).join(genome_anno))

    dmps_set = {}
    for i, (test, control) in enumerate(contrasts_levels, 1):
        diff_meth_probes = top_table(
            contrasts_fit, coef=i, sort_by="P", num=ro.r("Inf"), genelist=m_values_ann
        )
        dmps_set[(test, control)] = rpy2_df_to_pd_df(diff_meth_probes)

        # 2.1. Save DMPs to disk
        save_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"top_table_{test}_vs_{control}.csv"
        )
        df_to_file(diff_meth_probes, save_path)

        # 2.2. Plot the top 5 most significantly differentially methylated CpGs
        for cpg in list(diff_meth_probes.rownames)[:5]:
            save_path = plots_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_{cpg}_"
                f"cpg_plot_{test}_vs_{control}.pdf"
            )
            plot_cpgs(
                dat=b_values,
                cpg=cpg,
                pheno=ro.StrVector(sample_groups),
                save_path=save_path,
                ylab="Beta values",
                type="categorical",
            )

    # 3. Significant differentially methylated probes
    dmps_set_sig = {}
    for (test, control), dmps_df in dmps_set.items():
        for p_col, p_th, lfc_level, lfc_th in product(
            p_cols, p_ths, lfc_levels, lfc_ths
        ):
            # 3.1. Filter by P-value column
            dmps_sig = dmps_df[dmps_df[p_col] < p_th]

            # 3.2. Filer by logFC
            if lfc_level == "hyper":
                dmps_sig = dmps_sig[dmps_sig["logFC"] > 0]
            elif lfc_level == "hypo":
                dmps_sig = dmps_sig[dmps_sig["logFC"] < 0]

            dmps_sig = dmps_sig[abs(dmps_sig["logFC"]) > lfc_th]

            # 3.3. Save result to disk
            p_col_str = p_col.replace(".", "_")
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")
            dmps_sig.to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                    f"top_table_{test}_vs_{control}_sig_"
                    f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}.csv"
                )
            )

            dmps_set_sig[(test, control, p_col, p_th, lfc_level, lfc_th)] = dmps_sig

    # 5. Differential mean methylation of probes
    m_values_df = deepcopy(rpy2_df_to_pd_df(m_values).join(genome_anno))
    b_values_df = deepcopy(rpy2_df_to_pd_df(b_values).join(genome_anno))

    for test, control in contrasts_levels:
        test_samples = targets[id_col][targets[contrast_factor] == test].tolist()
        control_samples = targets[id_col][targets[contrast_factor] == control].tolist()

        b_values_df[f"mean_diff_{test}_vs_{control}"] = b_values_df[test_samples].mean(
            axis=1
        ) - b_values_df[control_samples].mean(axis=1)
        m_values_df[f"mean_diff_{test}_vs_{control}"] = m_values_df[test_samples].mean(
            axis=1
        ) - m_values_df[control_samples].mean(axis=1)

        b_values_df[f"meth_status_{test}_vs_{control}"] = np.where(
            b_values_df[f"mean_diff_{test}_vs_{control}"] > 0, "hyper", "hypo"
        )
        m_values_df[f"meth_status_{test}_vs_{control}"] = np.where(
            m_values_df[f"mean_diff_{test}_vs_{control}"] > 0, "hyper", "hypo"
        )

    b_values_df.to_csv(
        results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_b_values_meth_metadata.csv.gz"
        )
    )
    m_values_df.to_csv(
        results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_m_values_meth_metadata.csv.gz"
        )
    )

    # 5.1. Keep only probes with a methylation difference above threshold
    for (test, control), mean_meth_diff_th in product(
        contrasts_levels, mean_meth_diff_ths
    ):
        b_values_df_filtered = deepcopy(
            b_values_df[
                abs(b_values_df[f"mean_diff_{test}_vs_{control}"]) > mean_meth_diff_th
            ]
        )

        # 5.1.1. Save filtered M-Values
        m_values_filtered = m_values.rx(
            ro.StrVector(b_values_df_filtered.index.tolist()), True
        )
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        save_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"{test}_vs_{control}_"
            f"m_values_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
        )
        df_to_file(m_values_filtered, save_path)

        # 5.1.2. Save filtered B-Values
        b_values_filtered = b_values.rx(
            ro.StrVector(b_values_df_filtered.index.tolist()), True
        )
        save_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"{test}_vs_{control}_"
            f"b_values_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
        )
        df_to_file(b_values_filtered, save_path)

    # 5.2. Unsupervised heatmaps
    ha_column = heatmap_annotation(
        df=pd.DataFrame(targets[contrast_factor]),
        col={contrast_factor: contrasts_levels_colors},
    )

    for (test, control), values_type, mean_meth_diff_th in product(
        contrasts_levels, ("m_values", "b_values"), mean_meth_diff_ths
    ):
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        counts_matrix = pd.read_csv(
            results_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"{test}_vs_{control}_"
                f"{values_type}_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
            ),
            index_col=0,
        )

        if counts_matrix.empty:
            continue

        inds = (
            counts_matrix.var(axis=1)
            .sort_values(ascending=False)
            .iloc[:heatmap_top_n]
            .index
        )
        counts_matrix = counts_matrix.loc[inds]
        # substract row means for better visualization
        counts_matrix = counts_matrix.sub(counts_matrix.mean(axis=1), axis=0)

        save_path = plots_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"{test}_vs_{control}_"
            f"{values_type}_mean_diff_filtered_{mean_meth_diff_th_str}_heatmap_pub.pdf"
        )

        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            width=10,
            height=20,
            name=f"Probes with methylation difference > {mean_meth_diff_th}",
            column_title=f"Top {heatmap_top_n} most variable probes ({values_type})",
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=False,
            cluster_columns=False,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

        save_path = plots_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"{test}_vs_{control}_"
            f"{values_type}_mean_diff_filtered_{mean_meth_diff_th_str}_heatmap.pdf"
        )
        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            name=f"Probes with methylation difference > {mean_meth_diff_th}",
            column_title=f"Top {heatmap_top_n} most variable probes ({values_type})",
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=True,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

    # 5.3. Extract differentially methylated probes (DMPs) according to the indices
    # (CpGs) filtered by mean methylation
    dmps_set_wrt_mean_diff = {}
    for (test, control), mean_meth_diff_th in product(
        contrasts_levels, mean_meth_diff_ths
    ):
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        indices_filtered = pd.read_csv(
            results_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"{test}_vs_{control}_"
                f"b_values_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
            ),
            index_col=0,
        ).index

        # load dmps
        load_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_top_table_"
            f"{test}_vs_{control}.csv"
        )
        dmps_df = pd.read_csv(load_path, index_col=0)

        # corresponding DMPs associated with the cpgs filtered by mean methylation
        save_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_top_table_"
            f"{test}_vs_{control}_wrt_mean_diff_{mean_meth_diff_th_str}.csv"
        )
        dmps_df.loc[indices_filtered].to_csv(save_path)

        dmps_set_wrt_mean_diff[(test, control, mean_meth_diff_th)] = dmps_df.loc[
            indices_filtered
        ]

    # 5.4. Extract significant differentially methylated probes (DMPs) according to the
    # indices (CpGs) filtered by mean methylation
    dmps_set_sig_wrt_mean_diff = {}
    for (
        (
            test,
            control,
            p_col,
            p_th,
            lfc_level,
            lfc_th,
        ),
        dmps_df_sig,
    ), mean_meth_diff_th in product(dmps_set_sig.items(), mean_meth_diff_ths):
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        indices_filtered = pd.read_csv(
            results_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"{test}_vs_{control}_"
                f"b_values_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
            ),
            index_col=0,
        ).index

        # intersection between significant probes under rank and lfc considerations
        # with those significant under mean meth considerations
        dmps_df_sig_wrt_mean_diff = dmps_df_sig.loc[
            indices_filtered.intersection(dmps_df_sig.index)
        ]

        # save to disk
        p_col_str = p_col.replace(".", "_")
        p_th_str = str(p_th).replace(".", "_")
        lfc_th_str = str(lfc_th).replace(".", "_")
        dmps_df_sig_wrt_mean_diff.to_csv(
            results_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"top_table_{test}_vs_{control}_sig_"
                f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                f"wrt_mean_diff_{mean_meth_diff_th_str}.csv"
            )
        )

        dmps_set_sig_wrt_mean_diff[
            (test, control, p_col, p_th, lfc_level, lfc_th, mean_meth_diff_th)
        ] = dmps_df_sig_wrt_mean_diff

    # 6. Supervised Heatmaps
    for (
        (
            test,
            control,
            p_col,
            p_th,
            lfc_level,
            lfc_th,
            mean_meth_diff_th,
        ),
        sig_probes,
    ), values_type in product(
        dmps_set_sig_wrt_mean_diff.items(), ("m_values", "b_values")
    ):
        if sig_probes.empty:
            logging.info("Empty sig_probes. Continuing.")
            continue

        sig_probes = sig_probes.sort_values("logFC", key=abs).iloc[:heatmap_top_n]

        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        counts_matrix = pd.read_csv(
            results_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"{test}_vs_{control}_"
                f"{values_type}_mean_diff_filtered_{mean_meth_diff_th_str}.csv"
            ),
            index_col=0,
        )

        if counts_matrix.empty:
            continue

        counts_matrix = counts_matrix.loc[sig_probes.index]
        counts_matrix = counts_matrix.sub(
            counts_matrix.mean(axis=1), axis=0
        )  # substract row means for better visualization

        p_col_str = p_col.replace(".", "_")
        p_th_str = str(p_th).replace(".", "_")
        lfc_th_str = str(lfc_th).replace(".", "_")
        save_path = plots_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"top_table_{test}_vs_{control}_sig_"
            f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
            f"wrt_mean_diff_{mean_meth_diff_th_str}_heatma_pub.pdf"
        )

        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            width=10,
            height=20,
            name=f"Probes with methylation difference > {mean_meth_diff_th}",
            column_title=(
                f"Most significant probes ({test} vs. {control}, "
                f"{p_col} < {p_th}, {lfc_level}, "
                f"abs(logFC) > {lfc_th}, "
                f"diff. mean meth. < {mean_meth_diff_th}, {values_type})"
            ),
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=False,
            cluster_columns=False,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

        save_path = plots_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            f"top_table_{test}_vs_{control}_sig_"
            f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
            f"wrt_mean_diff_{mean_meth_diff_th_str}_heatmap.pdf"
        )
        complex_heatmap(
            counts_matrix,
            save_path=save_path,
            name=f"Probes with methylation difference > {mean_meth_diff_th}",
            column_title=(
                f"Most significant probes ({test} vs. {control}, "
                f"{p_col} < {p_th}, {lfc_level}, "
                f"abs(logFC) > {lfc_th}, "
                f"diff. mean meth. < {mean_meth_diff_th}, {values_type})"
            ),
            top_annotation=ha_column,
            show_row_names=False,
            show_column_names=True,
            heatmap_legend_param=ro.r(
                'list(title_position = "topcenter", color_bar = "continuous",'
                ' legend_height = unit(5, "cm"), legend_direction = "horizontal")'
            ),
        )

    # 7. Annotation of probes
    # TODO: encapsule and extract to avoid repetition
    # 7.1. Probes filtered by mean methylation
    for (
        test,
        control,
        mean_meth_diff_th,
    ), dmps_df_wrt_mean_diff in dmps_set_wrt_mean_diff.items():
        sig_probes = dmps_df_wrt_mean_diff[
            [
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "logFC",
                "P.Value",
                "adj.P.Val",
            ]
        ].dropna()
        sig_probes.rename(
            columns={"CpG_chrm": "chr", "CpG_beg": "start", "CpG_end": "end"},
            inplace=True,
        )

        if sig_probes.empty:
            logging.info("Empty result. Continuing.")
            continue

        # create genomic ranges
        seqinfo = ro.r("Seqinfo")(
            seqnames=ro.StrVector(sig_probes["chr"].astype(str).unique()), genome=genome
        )
        regions = make_granges_from_dataframe(
            sig_probes, keep_extra_columns=True, seqinfo=seqinfo
        )

        # annotate regions
        for anno_type, annot in annots.items():
            try:
                sig_probes_annot = annotate_regions(
                    regions=regions, annotations=annot, ignore_strand=True, quiet=False
                )
            except RRuntimeError as e:
                logging.error(e)
                continue

            # remove duplicate rows
            sig_probes_annot = ro.r("unique")(sig_probes_annot)

            # save to disk
            mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
            rpy2_df_to_pd_df(sig_probes_annot).replace("NA_character_", np.nan).to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                    f"top_table_{test}_vs_{control}_"
                    f"wrt_mean_diff_{mean_meth_diff_th_str}_{anno_type}s.csv"
                )
            )

            # annotation summary
            sig_probes_annot_summary = summarize_annotations(
                annotated_regions=sig_probes_annot, quiet=False
            )
            rpy2_df_to_pd_df(sig_probes_annot_summary).to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                    f"top_table_{test}_vs_{control}_"
                    f"wrt_mean_diff_{mean_meth_diff_th_str}_{anno_type}s_summary.csv"
                )
            )

            # plot annotations
            save_path = plots_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"top_table_{test}_vs_{control}_"
                f"wrt_mean_diff_{mean_meth_diff_th_str}_{anno_type}s_plot.pdf"
            )
            plot_annotation(
                annotated_regions=sig_probes_annot,
                save_path=save_path,
                annotation_order=annots_order,
                plot_title=(
                    f"# of significant sites ({test} vs. {control}, "
                    f"diff. mean meth. < {mean_meth_diff_th_str})"
                ),
                x_label=f"Known {anno_type} Annotations",
                y_label="Count",
            )

            # Extract and save only one annotation type at a time
            # (separately for cpgs and genes)
            df = rpy2_df_to_pd_df(sig_probes_annot)
            for ann_to_extract in set(df["annot.type"]):
                df_annot = df[df["annot.type"] == ann_to_extract]

                mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
                df_annot.to_csv(
                    results_path.joinpath(
                        f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                        f"top_table_{test}_vs_{control}_"
                        f"wrt_mean_diff_{mean_meth_diff_th_str}_"
                        f"{ann_to_extract}.csv"
                    )
                )

    # 7.2. Significant probes filtered by mean methylation
    sig_probes_summary_stats = defaultdict(dict)
    sig_unique_genes_summary_stats = defaultdict(dict)
    for (
        test,
        control,
        p_col,
        p_th,
        lfc_level,
        lfc_th,
        mean_meth_diff_th,
    ), dmps_df_sig_wrt_mean_diff in dmps_set_sig_wrt_mean_diff.items():
        sig_probes = dmps_df_sig_wrt_mean_diff[
            [
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "logFC",
                "P.Value",
                "adj.P.Val",
            ]
        ].dropna()
        sig_probes.rename(
            columns={"CpG_chrm": "chr", "CpG_beg": "start", "CpG_end": "end"},
            inplace=True,
        )

        if sig_probes.empty:
            logging.info("Empty result. Continuing.")
            continue

        # create genomic ranges
        seqinfo = ro.r("Seqinfo")(
            seqnames=ro.StrVector(sig_probes["chr"].astype(str).unique()), genome=genome
        )
        regions = make_granges_from_dataframe(
            sig_probes, keep_extra_columns=True, seqinfo=seqinfo
        )

        # annotate regions
        for anno_type, annot in annots.items():
            try:
                sig_probes_annot = annotate_regions(
                    regions=regions, annotations=annot, ignore_strand=True, quiet=False
                )
            except RRuntimeError as e:
                logging.error(e)
                continue

            # remove duplicate rows
            sig_probes_annot = ro.r("unique")(sig_probes_annot)

            # save to disk
            p_col_str = p_col.replace(".", "_")
            p_th_str = str(p_th).replace(".", "_")
            lfc_th_str = str(lfc_th).replace(".", "_")
            mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
            rpy2_df_to_pd_df(sig_probes_annot).replace("NA_character_", np.nan).to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                    f"top_table_{test}_vs_{control}_sig_"
                    f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                    f"wrt_mean_diff_{mean_meth_diff_th_str}_{anno_type}s.csv"
                )
            )

            # annotation summary
            sig_probes_annot_summary = summarize_annotations(
                annotated_regions=sig_probes_annot, quiet=False
            )
            rpy2_df_to_pd_df(sig_probes_annot_summary).to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                    f"top_table_{test}_vs_{control}_sig_"
                    f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                    f"wrt_mean_diff_{mean_meth_diff_th_str}_{anno_type}s_summary.csv"
                )
            )

            # plot annotations
            save_path = plots_path.joinpath(
                f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                f"top_table_{test}_vs_{control}_sig_"
                f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                f"wrt_mean_diff_{mean_meth_diff_th_str}_{anno_type}s_plot.pdf"
            )
            plot_annotation(
                annotated_regions=sig_probes_annot,
                save_path=save_path,
                annotation_order=annots_order,
                plot_title=(
                    f"# of significant sites ({test} vs. {control}, "
                    f"{p_col} < {p_th}, {lfc_level}, "
                    f"abs(logFC) > {lfc_th}, "
                    f"diff. mean meth. < {mean_meth_diff_th})"
                ),
                x_label=f"Known {anno_type} Annotations",
                y_label="Count",
            )

            # Extract and save only one annotation type at a time
            # (separately for cpgs and genes)
            df = rpy2_df_to_pd_df(sig_probes_annot)
            for ann_to_extract in set(df["annot.type"]):
                df_annot = df[df["annot.type"] == ann_to_extract]

                sig_probes_summary_stats[
                    test, control, p_col, p_th, lfc_level, lfc_th, mean_meth_diff_th
                ][ann_to_extract] = len(df_annot)

                if anno_type == "gene":
                    sig_unique_genes_summary_stats[
                        test, control, p_col, p_th, lfc_level, lfc_th, mean_meth_diff_th
                    ][ann_to_extract] = len(set(df_annot["annot.gene_id"]))

                df_annot.to_csv(
                    results_path.joinpath(
                        f"{exp_prefix}_diff_meth_probes_{norm_type}_"
                        f"top_table_{test}_vs_{control}_sig_"
                        f"{p_col_str}_{p_th_str}_{lfc_level}_{lfc_th_str}_"
                        f"wrt_mean_diff_{mean_meth_diff_th_str}_"
                        f"{ann_to_extract}.csv"
                    )
                )

    # 8. Save summary stats
    sig_probes_summary_stats_df = (
        pd.DataFrame(sig_probes_summary_stats).transpose().fillna(0)
    )
    sig_probes_summary_stats_df.to_csv(
        results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            "top_table_sig_probes_summary_stats.csv"
        )
    )

    sig_unique_genes_summary_stats_df = (
        pd.DataFrame(sig_unique_genes_summary_stats).transpose().fillna(0)
    )
    sig_unique_genes_summary_stats_df.to_csv(
        results_path.joinpath(
            f"{exp_prefix}_diff_meth_probes_{norm_type}_"
            "top_table_sig_unique_genes_summary_stats.csv"
        )
    )

    return norm_type, (design_matrix, contrast_matrix)


def diff_meth_regions(
    mset: Any,
    norm_type: str,
    design_matrix: Any,
    contrast_matrix: Any,
    sample_groups: Iterable[str],
    contrasts_levels: List[Tuple[str, str]],
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    p_ths: Iterable[float] = (0.05, 0.01),
    lfc_levels: Iterable[str] = ("hyper", "hypo", "all"),
    mean_meth_diff_ths: Iterable[float] = (0.1, 0.2, 0.3),
    genome: str = "hg38",
    array_type: str = "450K",
    dmrs_top_n: int = 10,
) -> None:
    """
    Get differentially methylated regions.

    Args:
        mset: Normalized methylation set object.
        norm_type: Normalization type.
        design_matrix: Design matrix (limma).
        contrast_matrix: Design matrix (limma).
        sample_groups: Name of the group each sample belongs to.
        contrasts_levels: List of contrasts to test for differential methylation.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        p_ths: P-value thresholds to determine significance.
        lfc_levels: logFC levels to subset results by.
        mean_meth_diff_ths: Mean methylation thresholds to subset results by.
        genome: Genome version.
        array_type: Methylation array type.
        dmrs_top_n: Number of individual top DMRs to plot.
    """
    # 0. Setup
    m_values = ro.r("getM")(mset)
    b_values = ro.r("getBeta")(mset)

    annots = {
        "cpg": build_annotations(genome=genome, annotations=f"{genome}_cpgs"),
        "gene": build_annotations(genome=genome, annotations=f"{genome}_basicgenes"),
    }
    annots_order = ro.StrVector(
        [
            f"{genome}_genes_1to5kb",
            f"{genome}_genes_promoters",
            f"{genome}_genes_5UTRs",
            f"{genome}_genes_exons",
            f"{genome}_genes_intronexonboundaries",
            f"{genome}_genes_introns",
            f"{genome}_genes_3UTRs",
            f"{genome}_genes_intergenic",
            f"{genome}_cpg_inter",
            f"{genome}_cpg_islands",
            f"{genome}_cpg_shelves",
            f"{genome}_cpg_shores",
        ]
    )

    assert all([lfc_level in ("hyper", "hypo", "all") for lfc_level in lfc_levels]), (
        'only "hyper", "hypo" and "all" are allowed lfc_level values'
    )

    # 1. Get differentially methylated regions and annotate cpgs and genes.
    for (test, control), p_th, mean_diff_level, mean_meth_diff_th in product(
        contrasts_levels, p_ths, lfc_levels, mean_meth_diff_ths
    ):
        p_col_str = "fdr"
        # 1.1. Get cpg annotations
        cpg_annotation = cpg_annotate(
            obj=m_values,
            design=design_matrix,
            datatype="array",
            what="M",
            analysis_type="differential",
            contrasts=True,
            cont_matrix=contrast_matrix,
            coef=f"{test}-{control}",
            arraytype=array_type,
            fdr=p_th,
        )

        # 1.2. Find differentially methylated regions
        try:
            dmrs = dmr_cate(cpg_annotation)
        except RRuntimeError as e:
            logging.error(f"Error executing dmr_cate: \n{e}")
            continue

        dmrs_df = pd.DataFrame({k: list(dmrs.do_slot(k)) for k in dmrs.slotnames()})

        # 1.3. Filter by mean methylation difference, add new fields, sort...
        if mean_diff_level == "hyper":
            dmrs_df = dmrs_df[dmrs_df["meandiff"] > 0]
        elif mean_diff_level == "hypo":
            dmrs_df = dmrs_df[dmrs_df["meandiff"] < 0]

        dmrs_df = dmrs_df[abs(dmrs_df["meandiff"]) > mean_meth_diff_th].dropna()

        if dmrs_df.empty:
            logging.warning("Empty dmrs_df after filtering. Continuing.")
            continue

        dmrs_df["chr"] = dmrs_df["coord"].apply(lambda x: x.split(":")[0])
        dmrs_df["start"] = dmrs_df["coord"].apply(
            lambda x: x.split(":")[1].split("-")[0]
        )
        dmrs_df["end"] = dmrs_df["coord"].apply(lambda x: x.split(":")[1].split("-")[1])
        dmrs_df = dmrs_df.sort_values(
            by="meandiff", key=abs, ascending=False
        ).set_index("coord")

        # 1.4. Save
        p_th_str = str(p_th).replace(".", "_")
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        save_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_regions_{norm_type}_"
            f"{test}_vs_{control}_{p_col_str}_{p_th_str}_"
            f"{mean_diff_level}_wrt_mean_diff_{mean_meth_diff_th_str}_results.csv"
        )
        dmrs_df.to_csv(save_path)

        # 1.5. Extract region ranges
        dmrs_df = dmrs_df.sort_values("chr")
        seqinfo = ro.r("Seqinfo")(
            seqnames=ro.StrVector(dmrs_df["chr"].astype(str).unique()), genome=genome
        )
        regions = make_granges_from_dataframe(
            dmrs_df, keep_extra_columns=True, seqinfo=seqinfo
        )

        # 1.6. Annotate regions
        for anno_type, annot in annots.items():
            # 1.6.1. Annotate cpgs
            dmrs_annot = annotate_regions(
                regions=regions, annotations=annot, ignore_strand=True, quiet=False
            )
            # 1.6.2. Remove duplicate rows
            dmrs_annot = ro.r("unique")(dmrs_annot)

            save_path = results_path.joinpath(
                f"{exp_prefix}_diff_meth_regions_{norm_type}_"
                f"{test}_vs_{control}_{p_col_str}_{p_th_str}_"
                f"{mean_diff_level}_wrt_mean_diff_{mean_meth_diff_th_str}_"
                f"{anno_type}s.csv"
            )
            rpy2_df_to_pd_df(ro.r("data.frame")(dmrs_annot)).replace(
                "NA_character_", np.nan
            ).to_csv(save_path)

            # 1.6.3. CPGs annotation summary
            dmrs_annot_summary = summarize_annotations(
                annotated_regions=dmrs_annot, quiet=False
            )
            save_path = results_path.joinpath(
                f"{exp_prefix}_diff_meth_regions_{norm_type}_"
                f"{test}_vs_{control}_{p_col_str}_{p_th_str}_"
                f"{mean_diff_level}_wrt_mean_diff_{mean_meth_diff_th_str}_"
                f"{anno_type}s_summary.csv"
            )
            rpy2_df_to_pd_df(ro.r("data.frame")(dmrs_annot_summary)).to_csv(save_path)

            # 1.6.4. Plot annotations
            save_path = plots_path.joinpath(
                f"{exp_prefix}_diff_meth_regions_{norm_type}_"
                f"{test}_vs_{control}_{p_col_str}_{p_th_str}_"
                f"{mean_diff_level}_wrt_mean_diff_{mean_meth_diff_th_str}_"
                f"{anno_type}s_plot.pdf"
            )
            plot_annotation(
                annotated_regions=dmrs_annot,
                save_path=save_path,
                annotation_order=annots_order,
                plot_title=(
                    "# of Regions Tested for DM "
                    f"(FDR<{p_th}, {mean_diff_level}, "
                    f"abs(mean_diff) > {mean_meth_diff_th})"
                ),
                x_label=f"Known {anno_type} Annotations",
                y_label="Count",
            )

    # 2. Plot top DMRs
    for (test, control), p_th, mean_diff_level, mean_meth_diff_th in product(
        contrasts_levels, p_ths, lfc_levels, mean_meth_diff_ths
    ):
        # 2.1. Load previously computed DMRs
        p_th_str = str(p_th).replace(".", "_")
        mean_meth_diff_th_str = str(mean_meth_diff_th).replace(".", "_")
        load_path = results_path.joinpath(
            f"{exp_prefix}_diff_meth_regions_{norm_type}_"
            f"{test}_vs_{control}_{p_col_str}_{p_th_str}_"
            f"{mean_diff_level}_wrt_mean_diff_{mean_meth_diff_th_str}_results.csv"
        )

        if not load_path.exists():
            logging.info("DMRs file does not exist. Continuing.")
            continue

        dmrs_df = pd.read_csv(load_path, index_col=0)[
            : min(dmrs_top_n, len(dmrs_df))
        ].dropna()

        if dmrs_df.empty:
            logging.info("Loaded DMRs are empty. Continuing.")
            continue

        # 2.2. Extract region ranges
        dmrs_df = dmrs_df.sort_values("chr")
        seqinfo = ro.r("Seqinfo")(
            seqnames=ro.StrVector(dmrs_df["chr"].astype(str).unique()), genome=genome
        )
        regions = make_granges_from_dataframe(
            dmrs_df, keep_extra_columns=True, seqinfo=seqinfo
        )

        # 2.3. Plot DMRs
        for i, coord in enumerate(dmrs_df.index):
            logging.info(f"Plotting DMR [{i}] - {coord}")
            coord = coord.replace(":", "_").replace("-", "_")
            save_path = plots_path.joinpath(
                f"{exp_prefix}_diff_meth_regions_{norm_type}_"
                f"{test}_vs_{control}_{p_col_str}_{p_th_str}_"
                f"{mean_diff_level}_wrt_mean_diff_{mean_meth_diff_th_str}_"
                f"dmr_plot_{i}_{coord}.pdf"
            )

            dmr_plot(
                ranges=regions,
                dmr=i + 1,
                cpgs=b_values,
                sample_groups=ro.StrVector(sample_groups),
                save_path=save_path,
                what="Beta",
                arraytype=array_type,
                pch=16,
                toscale=True,
                plotmedians=True,
                genome=genome,
            )


def differential_methylation_array(
    genome_anno: pd.DataFrame,
    idat_path: Path,
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    targets: pd.DataFrame,
    contrast_factor: str,
    contrasts_levels: List[Tuple[str, str]],
    contrasts_levels_colors: Dict[str, str],
    rg_set: Any = None,
    id_col: str = "sample_id",
    genome: str = "hg38",
    array_type: str = "450K",
    norm_types: Iterable[str] = (
        "raw",
        "funnorm",
        "illumina",
        "noob_reference",
        "noob_single",
        "swan",
        "quantile",
        "noob_quantile",
    ),
    n_threads: int = 8,
    p_cols: Iterable[str] = ("P.Value", "adj.P.Val"),
    p_ths: Iterable[float] = (0.05, 0.01),
    lfc_levels: Iterable[str] = ("hyper", "hypo", "all"),
    lfc_ths: Iterable[float] = (0.0, 1.0, 2.0),
    mean_meth_diff_ths: Iterable[float] = (0.1, 0.2, 0.3),
    heatmap_top_n: int = 1000,
    dmrs_top_n: int = 10,
    do_diff_analysis: bool = True,
):
    """Differential methylation analysis.

    Args:
        genome_anno: Genome annotation dataframe.
        idat_path: Path pointing to directory containing .idat files.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        targets: Samples annotation file.
        contrast_factor: Contrast factor (column of targets) for differential
            methylation.
        contrasts_levels: List of contrasts to test for differential methylation.
        contrasts_levels_colors: A dictionary where each contrast level
            has a color assigned.
        rg_set: A user-provided RG set.
        id_col: Column name to uniquely identify samples.
        genome: Genome version.
        array_type: Methylation array type.
        norm_types: Data normalization tyes to test. Each is run independently.
        n_threads: Number of threads to use internally.
        p_cols: P-value column names used for filtering.
        p_ths: P-value thresholds to determine significance.
        lfc_levels: logFC levels to subset results by.
        lfc_ths: logFC thresholds to subset results by.
        mean_meth_diff_ths: Mean methylation thresholds to subset results by.
        heatmap_top_n: Number of top differentially methylated probes to include in
            heatmap plots.
        dmrs_top_n: Number of individual top DMRs to plot.
    """
    # 1. Get RGChannelSet object
    logging.info("Get RGChannelSet object...")
    rg_set = rg_set or read_metharray_exp(
        data_dir=idat_path,
        targets=pd_df_to_rpy2_df(targets),
        id_col=id_col,
        force=True,
    )

    # 1.1. Save RGSet to disk
    save_rds(rg_set, results_path.joinpath(f"{exp_prefix}_rg_set.RDS"))

    # 2. Quality control
    logging.info("Quality control...")
    rg_set, targets, det_p = quality_control(
        rg_set=rg_set,
        targets=targets,
        id_col=id_col,
        contrast_factor=contrast_factor,
        plots_path=plots_path,
        exp_prefix=exp_prefix,
    )
    sample_groups = targets[contrast_factor].tolist()
    sample_names = targets[id_col].tolist()

    # 3. Normalization
    logging.info("Normalization...")
    func = functools.partial(
        normalization,
        rg_set=rg_set,
        results_path=results_path,
        plots_path=plots_path,
        sample_names=sample_names,
        sample_groups=ro.StrVector(sample_groups),
        exp_prefix=exp_prefix,
    )
    norm_inputs = [dict(norm_type=norm_type) for norm_type in norm_types]

    if len(norm_types) == 1:
        normalization_results = [func(**norm_inputs[0])]
    else:
        normalization_results = parallelize_map(
            functools.partial(run_func_dict, func=func),
            norm_inputs,
            threads=n_threads,
        )

    norm_msets = {
        norm_type: norm_mset
        for norm_type, norm_mset in [x for x in normalization_results if x is not None]
    }

    # 4. Probe filtering
    logging.info("Probe filtering...")
    func = functools.partial(
        filtering,
        det_p=det_p,
        genome_anno=genome_anno,
        plots_path=plots_path,
        results_path=results_path,
        sample_names=sample_names,
        sample_groups=sample_groups,
        contrasts_levels_colors=contrasts_levels_colors,
        array_type=array_type,
        exp_prefix=exp_prefix,
    )
    filtering_inputs = [
        dict(mset=norm_mset, norm_type=norm_type)
        for norm_type, norm_mset in norm_msets.items()
    ]

    if len(norm_types) == 1:
        filtering_results = [func(**filtering_inputs[0])]
    else:
        filtering_results = parallelize_map(
            functools.partial(
                run_func_dict,
                func=func,
            ),
            filtering_inputs,
            threads=n_threads,
        )

    norm_msets_filtered = {
        norm_type: norm_mset
        for norm_type, norm_mset in [x for x in filtering_results if x is not None]
    }

    # If no differential analysis is required, return.
    if not do_diff_analysis:
        return

    # 5. Differential methylation analysis of probes (DMPs)
    logging.info("Differential methylation analysis of probes (DMPs)...")
    func = functools.partial(
        diff_meth_probes,
        targets=targets,
        id_col=id_col,
        contrast_factor=contrast_factor,
        contrasts_levels=contrasts_levels,
        results_path=results_path,
        plots_path=plots_path,
        exp_prefix=exp_prefix,
        genome_anno=genome_anno,
        sample_groups=sample_groups,
        contrasts_levels_colors=contrasts_levels_colors,
        p_cols=p_cols,
        p_ths=p_ths,
        lfc_levels=lfc_levels,
        lfc_ths=lfc_ths,
        mean_meth_diff_ths=mean_meth_diff_ths,
        genome=genome,
        heatmap_top_n=heatmap_top_n,
    )
    diff_meth_probes_inputs = [
        dict(mset=norm_mset, norm_type=norm_type)
        for norm_type, norm_mset in norm_msets_filtered.items()
    ]
    if len(norm_types) == 1:
        contrast_intermediate_results = [func(**diff_meth_probes_inputs[0])]
    else:
        contrast_intermediate_results = parallelize_map(
            functools.partial(run_func_dict, func=func),
            diff_meth_probes_inputs,
            threads=n_threads,
        )

    contrast_intermediate_results = {
        norm_type: intermediate_results
        for norm_type, intermediate_results in [
            x for x in contrast_intermediate_results if x is not None
        ]
    }

    # 6. Differential methylation analysis of regions (DMRs)
    logging.info("Differential methylation analysis of regions (DMRs)")
    func = functools.partial(
        diff_meth_regions,
        contrast_factor=contrast_factor,
        sample_groups=sample_groups,
        contrasts_levels=contrasts_levels,
        results_path=results_path,
        plots_path=plots_path,
        exp_prefix=exp_prefix,
        p_ths=p_ths,
        lfc_levels=lfc_levels,
        mean_meth_diff_ths=mean_meth_diff_ths,
        genome=genome,
        array_type=array_type,
        dmrs_top_n=dmrs_top_n,
    )
    diff_meth_regions_inputs = [
        dict(
            mset=norm_msets_filtered[norm_type],
            norm_type=norm_type,
            design_matrix=design_matrix,
            contrast_matrix=contrast_matrix,
        )
        for norm_type, (
            design_matrix,
            contrast_matrix,
        ) in contrast_intermediate_results.items()
    ]
    if len(norm_types) == 1:
        func(**diff_meth_regions_inputs[0])
    else:
        parallelize_map(
            functools.partial(run_func_dict, func=func),
            diff_meth_regions_inputs,
            threads=n_threads,
        )
