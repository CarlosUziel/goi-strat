"""
Utilities for differential methylation analysis using BiSeq.

This module provides functions for analyzing DNA methylation data specifically
focusing on identifying differentially methylated regions (DMRs) rather than
individual CpG sites. BiSeq is particularly suited for RRBS (Reduced Representation
Bisulfite Sequencing) data. Key features include:

1. Data processing and clustering:
   - Reading and processing methylation data from Bismark coverage files
   - Clustering CpG sites into regions for more robust statistical analysis
   - Computing and visualizing coverage statistics across samples

2. Differential methylation analysis:
   - Identifying differentially methylated loci (DMLs) between conditions
   - Identifying differentially methylated regions (DMRs) between conditions
   - Filtering results based on FDR thresholds and methylation difference levels

3. Annotation and visualization:
   - Annotating DMRs with genomic features (CpG islands, genes, enhancers, etc.)
   - Creating summary statistics and plots for interpretation
   - Comparing methylation changes to randomly selected regions for significance assessment

The module integrates with multiple R libraries (BiSeq, bsseq, DSS) through rpy2
for specialized methylation analysis while maintaining a consistent Python interface
for workflow orchestration.
"""

# Reference workflow:
#   https://dockflow.org/workflow/methylation-array-analysis/#content

import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

from r_wrappers.annotatr import (
    annotate_regions,
    build_annotations,
    plot_annotation,
    randomize_regions,
    summarize_annotations,
)
from r_wrappers.biseq import (
    bs_raw,
    cluster_sites,
    cluster_sites_to_gr,
    cov_boxplot,
    cov_statistics,
)
from r_wrappers.bsseq import get_coverage, read_bismark
from r_wrappers.dds import call_dml, call_dmr, dml_test
from r_wrappers.utils import (
    homogeinize_seqlevels_style,
    make_granges_from_dataframe,
    pd_df_to_rpy2_df,
    rpy2_df_to_pd_df_manual,
    save_csv,
)


def differential_methylation_rrbs_regions(
    exp_prefix: str,
    annot_df: pd.DataFrame,
    bismark_path: Path,
    contrast_factor: str,
    results_path: Path,
    plots_path: Path,
    contrast_levels: Tuple[str, str],
    genome: str = "hg38",
    n_processes: int = 16,
    fdr_ths: Iterable[float] = (0.05, 0.01),
    mean_diff_levels: Iterable[str] = ("hyper", "hypo", "all"),
    mean_diff_ths: Iterable[float] = (10, 20, 30),
) -> None:
    """
    Perform differential methylation analysis focusing on regions in RRBS data.

    This function processes Reduced Representation Bisulfite Sequencing (RRBS) data
    to identify differentially methylated regions (DMRs) and differentially
    methylated loci (DMLs) between two sample groups. The workflow includes:

    1. Loading and processing Bismark coverage files
    2. Clustering CpG sites into regions
    3. Testing for differential methylation
    4. Filtering results by FDR and methylation difference thresholds
    5. Annotating results with genomic features
    6. Generating summary statistics and visualization plots

    Args:
        exp_prefix (str): Prefix string for all generated files and plots.
        annot_df (pd.DataFrame): Sample annotation dataframe containing sample metadata indexed by
            sample ID. Must contain the contrast_factor column for grouping samples.
        bismark_path (Path): Path to directory containing Bismark coverage files (*.bismark.cov.gz).
            Files should be named with the sample ID as prefix.
        contrast_factor (str): Column name in annot_df used to define sample groups for
            differential methylation analysis (e.g., "condition", "treatment").
        results_path (Path): Directory path where all analysis results files will be saved.
        plots_path (Path): Directory path where all visualization plots will be saved.
        contrast_levels (Tuple[str, str]): Tuple of (test, control) specifying the two levels of the
            contrast_factor to compare. The first element is the test condition,
            the second is the reference/control condition.
        genome (str): Genome assembly version used for annotations (e.g., "hg38", "mm10").
            Default is "hg38".
        n_processes (int): Number of CPU processes to use for parallel processing in the
            differential methylation testing step. Default is 16.
        fdr_ths (Iterable[float]): Iterable of FDR threshold values for filtering significant results.
            Default is (0.05, 0.01).
        mean_diff_levels (Iterable[str]): Iterable of methylation difference types to filter results.
            "hyper" selects hypermethylated regions, "hypo" selects hypomethylated regions,
            and "all" selects both. Default is ("hyper", "hypo", "all").
        mean_diff_ths (Iterable[float]): Iterable of mean methylation difference threshold values (percentage points)
            for filtering significant results. Default is (10, 20, 30).

    Returns:
        None. Results are written to disk in the specified output directories:
        - Clustering results and coverage statistics
        - Differentially methylated loci (DML) and regions (DMR)
        - Filtered results for various significance thresholds and methylation differences
        - Annotation files with genomic context for differential methylation
        - Visualization plots for methylation patterns and annotations

    Examples:
        >>> differential_methylation_rrbs_regions(
        ...     exp_prefix="experiment1",
        ...     annot_df=sample_annotations,
        ...     bismark_path=Path("/data/bismark_coverage"),
        ...     contrast_factor="treatment",
        ...     results_path=Path("/results/methylation"),
        ...     plots_path=Path("/results/plots"),
        ...     contrast_levels=("treated", "control"),
        ...     genome="hg38",
        ...     n_processes=8,
        ...     fdr_ths=[0.05],
        ...     mean_diff_levels=["hyper", "hypo", "all"],
        ...     mean_diff_ths=[10, 20]
        ... )
    """
    # 0. Setup
    annotations = build_annotations(
        genome=genome,
        annotations=ro.StrVector(
            [
                f"{genome}_cpgs",
                f"{genome}_basicgenes",
                f"{genome}_genes_intergenic",
                f"{genome}_genes_intronexonboundaries",
                f"{genome}_enhancers_fantom",
            ]
        ),
    )
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
    test, control = contrast_levels
    annot_df_contrasts = deepcopy(
        annot_df[annot_df[contrast_factor].isin((test, control))]
    )

    # 1. Get methylation coverage files
    samples_cov_files = {
        sample_id: cov_file
        for cov_file in sorted(bismark_path.glob("**/*.bismark.cov.gz"))
        if (sample_id := cov_file.name.split(".")[0]) in annot_df_contrasts.index
    }
    assert len(samples_cov_files) > 0, "No coverage files were found."
    sample_ids = list(samples_cov_files.keys())

    # 1.1. Filter annotation dataframe by available samples
    annot_df_contrasts = annot_df_contrasts.loc[sample_ids]
    sample_groups = [
        1 if sample_condition == test else 0
        for sample_condition in annot_df_contrasts[contrast_factor]
    ]

    col_data = pd_df_to_rpy2_df(
        pd.DataFrame(
            {
                "sample_ids": sample_ids,
                "group": sample_groups,
            }
        ).set_index("sample_ids")
    )

    # 2. Get BSseq object from bismark coverage files
    bsseq_obj = read_bismark(
        list(samples_cov_files.values()),
        colData=col_data,
    )
    logging.info("Finished loading methylation calls data.")

    # 3. Get and annotate methylation cluster sites
    # 3.1. Get BS raw object
    bb = bs_raw(
        metadata={},
        row_ranges=ro.r("granges")(bsseq_obj),
        col_data=col_data,
        total_reads=get_coverage(bsseq_obj),
        meth_reads=ro.r("getBSseq")(bsseq_obj, type="M"),
    )

    # 3.2. Coverage statistics
    save_csv(cov_statistics(bb), results_path.joinpath(f"{exp_prefix}_cov_stats.csv"))
    cov_boxplot(
        bb,
        plots_path.joinpath(f"{exp_prefix}_cov_stats_boxplot.pdf"),
        col="cornflowerblue",
        las=2,
        ylim=ro.IntVector([0, 500000]),
    )
    logging.info("Finished statistics and plots for methylation calls data.")

    # 3.3. Compute sites clusters
    regions = cluster_sites_to_gr(
        cluster_sites(
            bb,
            groups=ro.r("factor")(ro.r("data.frame")(ro.r("colData")(bb)).rx2("group")),
            perc_samples=10 / 12,
            min_sites=20,
            max_dist=100,
        )
    )
    logging.info("Finished sites clustering.")

    # 3.4. Annotate clusters
    regions = homogeinize_seqlevels_style(regions, annotations)
    regions = annotate_regions(
        regions=regions,
        annotations=annotations,
        ignore_strand=True,
        quiet=False,
    )
    rpy2_df_to_pd_df_manual(regions).replace("NA_character_", np.nan).to_csv(
        results_path.joinpath(f"{exp_prefix}_clusters_ann.csv"),
    )
    save_csv(
        ro.r("data.frame")(
            summarize_annotations(annotated_regions=regions, quiet=False)
        ),
        results_path.joinpath(f"{exp_prefix}_clusters_ann_summary.csv"),
    )
    logging.info("Finished annotating sites clusters.")

    # 4. Get differential methylation loci (DML)
    test_samples = annot_df_contrasts[
        annot_df_contrasts[contrast_factor] == test
    ].index.tolist()
    control_samples = annot_df_contrasts[
        annot_df_contrasts[contrast_factor] == control
    ].index.tolist()

    # 4.1. DML testing
    dml_test_res = dml_test(
        bsseq_obj,
        group1=ro.StrVector(control_samples),
        group2=ro.StrVector(test_samples),
        ncores=n_processes,
    )
    logging.info("Finished DML testing.")

    # 4.2. Call DMLs and DMRs
    diff_meth_results = dict()
    for dm_type in ("dml", "dmr"):
        # 4.2.1. Call
        dm_df = rpy2_df_to_pd_df_manual(
            call_dml(dml_test_res, p_threshold=1)
            if dm_type == "dml"
            else call_dmr(dml_test_res, p_threshold=1)
        ).sort_values("chr")
        dm_df.to_csv(
            results_path.joinpath(f"{exp_prefix}_{dm_type}_{test}_vs_{control}.csv")
        )

        diff_meth_results[(None, None, None, dm_type)] = dm_df

        # 4.2.2. Filter results
        for fdr_th, mean_diff, mean_diff_level in product(
            fdr_ths, mean_diff_ths, mean_diff_levels
        ):
            if mean_diff_level == "hyper":
                dm_df_filt = deepcopy(dm_df[dm_df["diff"] > mean_diff])
            elif mean_diff_level == "hypo":
                dm_df_filt = deepcopy(dm_df[dm_df["diff"] < mean_diff])
            else:
                dm_df_filt = deepcopy(dm_df[abs(dm_df["diff"]) > mean_diff])

            dm_df_filt = dm_df_filt[dm_df_filt["fdr"] < fdr_th]

            diff_meth_results[(fdr_th, mean_diff, mean_diff_level, dm_type)] = (
                dm_df_filt
            )

            fdr_th_str = str(fdr_th).replace(".", "_")
            mean_diff_str = str(mean_diff).replace(".", "_")
            dm_df_filt.to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_{dm_type}_{test}_vs_{control}_"
                    f"{mean_diff_level}_fdr_{fdr_th_str}_diff_{mean_diff_str}.csv"
                )
            )
    logging.info("Finished differential methylation (DML/DMR testing).")

    # 5. Annotate differential methylation results (DMLs and DMRs)
    diff_meth_results_ann = dict()
    for (
        fdr_th,
        mean_diff,
        mean_diff_level,
        dm_type,
    ), diff_meth_res in diff_meth_results.items():
        # 5.1. Convert result to GRanges object
        diff_meth_res_granges = homogeinize_seqlevels_style(
            make_granges_from_dataframe(
                pd_df_to_rpy2_df(diff_meth_res),
                start_field="pos",
                end_field="pos",
                keep_extra_columns=True,
            ),
            annotations,
        )

        # 5.2. Annotate regions
        diff_meth_res_ann = annotate_regions(
            regions=diff_meth_res_granges,
            annotations=annotations,
            ignore_strand=True,
            quiet=False,
        )
        diff_meth_results_ann[(fdr_th, mean_diff, mean_diff_level, dm_type)] = (
            diff_meth_res_ann
        )

        # 5.3. Save to disk
        diff_meth_res_ann_df = rpy2_df_to_pd_df_manual(diff_meth_res_ann).replace(
            "NA_character_", np.nan
        )

        # 5.3.1. All annotation types
        if all(v is None for v in (fdr_th, mean_diff, mean_diff_level)):
            save_path_prefix = f"{exp_prefix}_{dm_type}_{test}_vs_{control}_ann"
        else:
            fdr_th_str = str(fdr_th).replace(".", "_")
            mean_diff_str = str(mean_diff).replace(".", "_")
            save_path_prefix = (
                f"{exp_prefix}_{dm_type}_{test}_vs_{control}_"
                f"{mean_diff_level}_fdr_{fdr_th_str}_diff_{mean_diff_str}_ann"
            )
        diff_meth_res_ann_df.to_csv(results_path.joinpath(save_path_prefix + ".csv"))

        # 5.3.2. Each annotation type separately
        for annot_subtype, df in diff_meth_res_ann_df.groupby("annot.type"):
            df.to_csv(results_path.joinpath(save_path_prefix + f"_{annot_subtype}.csv"))

        # 5.4. Save annotations summary
        if all(v is None for v in (fdr_th, mean_diff, mean_diff_level)):
            save_path = results_path.joinpath(
                f"{exp_prefix}_{dm_type}_{test}_vs_{control}_ann_summary.csv"
            )
        else:
            fdr_th_str = str(fdr_th).replace(".", "_")
            mean_diff_str = str(mean_diff).replace(".", "_")
            save_path = results_path.joinpath(
                f"{exp_prefix}_{dm_type}_{test}_vs_{control}_"
                f"{mean_diff_level}_fdr_{fdr_th_str}_diff_{mean_diff_str}_ann_summary"
                ".csv"
            )
        save_csv(
            ro.r("data.frame")(
                summarize_annotations(annotated_regions=diff_meth_res_ann, quiet=False)
            ),
            save_path,
        )

        # 5.5. Plot annotations
        if all(v is None for v in (fdr_th, mean_diff, mean_diff_level)):
            save_path = plots_path.joinpath(
                f"{exp_prefix}_{dm_type}_{test}_vs_{control}_ann_plot.pdf"
            )
        else:
            fdr_th_str = str(fdr_th).replace(".", "_")
            mean_diff_str = str(mean_diff).replace(".", "_")
            save_path = plots_path.joinpath(
                f"{exp_prefix}_{dm_type}_{test}_vs_{control}_"
                f"{mean_diff_level}_fdr_{fdr_th_str}_diff_{mean_diff_str}_"
                "ann_plot.pdf"
            )
        plot_annotation(
            annotated_regions=diff_meth_res_ann,
            save_path=save_path,
            annotation_order=annots_order,
            plot_title="# of Sites Tested for DM",
            x_label="Known Annotations",
            y_label="Count",
        )
    logging.info("Finished annotation of differential methylation filtered results.")

    # 6. Plot annottions with random regions
    for (
        fdr_th,
        mean_diff,
        mean_diff_level,
        dm_type,
    ), diff_meth_res_ann in diff_meth_results_ann.items():
        if all(v is None for v in (fdr_th, mean_diff, mean_diff_level)):
            continue

        try:
            regions = homogeinize_seqlevels_style(
                make_granges_from_dataframe(
                    ro.r("data.frame")(diff_meth_res_ann),
                    keep_extra_columns=True,
                ),
                annotations,
            )
            annotated_random = annotate_regions(
                regions=randomize_regions(
                    regions, allow_overlaps=True, per_chromosome=True
                ),
                annotations=annotations,
                ignore_strand=True,
                quiet=False,
            )
            plot_annotation(
                annotated_regions=diff_meth_res_ann,
                annotated_random=annotated_random,
                save_path=plots_path.joinpath(
                    f"{exp_prefix}_{dm_type}_{test}_vs_{control}_"
                    f"{mean_diff_level}_fdr_{fdr_th_str}_diff_{mean_diff_str}_"
                    "ann_rnd_regions_plot.pdf"
                ),
                annotation_order=[annots_order],
                plot_title="# of Regions Tested for DM wrt. random regions",
                x_label="Known Gene & CpGs Annotations",
                y_label="Count",
            )
        except RRuntimeError as e:
            print(e)
            print("Continuing.")

    logging.info("Finished plotting of annotations against random regions.")
