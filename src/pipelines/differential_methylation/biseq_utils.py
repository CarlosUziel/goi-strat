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
from rpy2.rinterface_lib.embedded import RRuntimeError


def differential_methylation_rrbs_regions(
    exp_prefix: str,
    annot_df: pd.DataFrame,
    bismark_path: Path,
    contrast_factor: str,
    results_path: Path,
    plots_path: Path,
    contrast_levels: Tuple[str, str],
    genome: str = "hg38",
    n_threads: int = 16,
    fdr_ths: Iterable[float] = (0.05, 0.01),
    mean_diff_levels: Iterable[str] = ("hyper", "hypo", "all"),
    mean_diff_ths: Iterable[float] = (10, 20, 30),
) -> None:
    """Differential methylation analysis of RRBS samples focusing on regions.

    Args:
        exp_prefix: Prefix string for all generated files.
        annot_df: Genome annotation dataframe.
        bismark_path: Path to bismark directory with coverage files.
        contrast_factor: Field name defining contrast levels.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        contrast_levels: Contrast to test for differential methylation.
        genome: Genome version.
        n_threads: Number of threads to use to calculate differential methylation.
        fdr_ths: FDR thresholds to use to filter the results.
        mean_diff_levels: Mean methylation difference levels to filter the results by.
        mean_diff_ths: Mean methylation difference thresholds to filter the results by.
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
        ncores=n_threads,
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
