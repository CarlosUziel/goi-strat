# Reference workflow:
#   https://dockflow.org/workflow/methylation-array-analysis/#content

import logging
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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
from r_wrappers.methylkit import (
    calculate_diff_meth,
    cluster_samples,
    filter_by_coverage,
    get_coverage_stats,
    get_methyl_diff,
    get_methylation_stats,
    global_strand_specific_scatter,
    meth_changes_anno_boxplot,
    meth_changes_anno_scatter3d,
    meth_levels_anno_boxplot,
    meth_read,
    methylation_change_wrt_condition,
    pca_samples,
    reorganize,
    triplet_analysis,
    unite,
    violin_plot,
)
from r_wrappers.utils import (
    homogeinize_seqlevels_style,
    make_granges_from_dataframe,
    rpy2_df_to_pd_df_manual,
    save_csv,
    save_rds,
)
from rpy2.rinterface_lib.embedded import RRuntimeError


def plot_save_stats(
    methyl_obj: Any,
    plots_path: Path,
    results_path: Path,
    exp_prefix: str,
) -> None:
    """Compute, plot and save methylation and coverage statistics.

    Args:
        methyl_obj: Methylation object.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
    """
    # 0. Setup subdirectories
    meth_results_root = results_path.joinpath("methylation_statistics")
    meth_results_root.mkdir(exist_ok=True, parents=True)
    meth_plots_root = plots_path.joinpath("methylation_statistics")
    meth_plots_root.mkdir(exist_ok=True, parents=True)

    cov_results_root = results_path.joinpath("coverage_statistics")
    cov_results_root.mkdir(exist_ok=True, parents=True)
    cov_plots_root = plots_path.joinpath("coverage_statistics")
    cov_plots_root.mkdir(exist_ok=True, parents=True)

    # 1. Compute statistics
    for sample_id in methyl_obj.names:
        # 1.1. Methylation statistics
        save_csv(
            get_methylation_stats(
                methyl_obj.rx2(sample_id),
                meth_plots_root.joinpath(f"{exp_prefix}_{sample_id}.pdf"),
                plot=True,
                both_strands=False,
            ),
            meth_results_root.joinpath(f"{exp_prefix}_{sample_id}.csv"),
        )

        # 1.2. Coverage statistics
        save_csv(
            get_coverage_stats(
                methyl_obj.rx2(sample_id),
                cov_plots_root.joinpath(f"{exp_prefix}_{sample_id}.pdf"),
                plot=True,
                both_strands=False,
            ),
            cov_results_root.joinpath(f"{exp_prefix}_{sample_id}.csv"),
        )


def make_meth_plots(
    methyl_obj: Any,
    condition_samples: Dict[str, Iterable[str]],
    plots_path: Path,
    results_path: Path,
    exp_prefix: str,
) -> None:
    """Compute various plots from the united methylation object.

    Args:
        methyl_obj: Methylation object.
        condition_samples: Samples per condition.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
    """

    # 1. Methylation correlation
    # save_csv(
    #     get_correlation(
    #         methyl_obj,
    #         plots_path.joinpath(f"{exp_prefix}_correlation.pdf"),
    #         plot=True,
    #     ),
    #     results_path.joinpath(f"{exp_prefix}_correlation.csv"),
    # )

    # 2. Samples clustering
    _ = cluster_samples(
        methyl_obj,
        plots_path.joinpath(f"{exp_prefix}_samples_clustering.pdf"),
        plot=True,
    )

    # 3. PCA on samples
    save_csv(
        pca_samples(
            methyl_obj,
            plots_path.joinpath(f"{exp_prefix}_samples_pca.pdf"),
        ),
        results_path.joinpath(f"{exp_prefix}_samples_pca.csv"),
    )

    # 4. Violin plot
    violin_plot(
        methyl_obj,
        condition_samples,
        plots_path.joinpath(f"{exp_prefix}_violin_plot.pdf"),
    )

    # 5. Global strand-specific effects (smooth scatter)
    global_strand_specific_scatter(
        methyl_obj,
        plots_path.joinpath(f"{exp_prefix}_strand_effects_scatter.pdf"),
    )

    # 6. Triplets analysis
    triplet_analysis(
        methyl_obj,
        plots_path.joinpath(f"{exp_prefix}_strand_triplet_analysis.pdf"),
    )


def make_diff_meth_plots(
    methyl_obj: Any,
    methyl_diffs: Dict[Tuple[str, str], Any],
    condition_samples: Dict[str, Iterable[str]],
    cpgs_ann: Any,
    plots_path: Path,
    exp_prefix: str,
) -> None:
    """Compute various plots from differential methylation results objects.

    Args:
        methyl_obj: Methylation object.
        methyl_diffs: Differential methylation objects per contrast.
        condition_samples: Dictionary of samples per condition.
        cpgs_ann: Annotated CpGs object.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
    """
    # 0. Change format of methyl_diff
    methyl_diffs = {
        f"{test} vs {control}": diff_meth_obj
        for (test, control), diff_meth_obj in methyl_diffs.items()
    }

    # 1. Does methylation change depending on methylation level of another condition?
    for condition in condition_samples.keys():
        methylation_change_wrt_condition(
            methyl_obj=methyl_obj,
            methyl_diffs=methyl_diffs,
            condition_sample=condition_samples,
            wrt_condition=condition,
            save_path_prefix=plots_path.joinpath(
                f"{exp_prefix}_methylation_level_change_wrt_{condition}"
            ),
        )

    # 2. Methylation levels with regard to CpG Annotations
    meth_levels_anno_boxplot(
        methyl_obj=methyl_obj,
        cpgs_ann=cpgs_ann,
        condition_sample=condition_samples,
        save_path_prefix=plots_path.joinpath(
            f"{exp_prefix}_methylation_levels_wrt_cpg_ann"
        ),
    )

    # 3. Methylation changes with regard to CpG Annotations
    for condition in condition_samples.keys():
        meth_changes_anno_boxplot(
            methyl_obj=methyl_obj,
            cpgs_ann=cpgs_ann,
            wrt_condition=condition,
            methyl_diffs=methyl_diffs,
            save_path_prefix=plots_path.joinpath(
                f"{exp_prefix}_methylation_changes_wrt_cpg_ann_{condition}"
            ),
        )

    # 4. Methylation changes in a 3D scatter plot with regard to CpG Annotations
    for condition in condition_samples.keys():
        meth_changes_anno_scatter3d(
            methyl_obj=methyl_obj,
            cpgs_ann=cpgs_ann,
            condition_sample=condition_samples,
            wrt_condition=condition,
            methyl_diffs=methyl_diffs,
            save_path_prefix=plots_path.joinpath(
                f"{exp_prefix}_methylation_changes_3d_wrt_cpg_ann_{condition}"
            ),
        )


def differential_methylation_rrbs_sites(
    annot_df: pd.DataFrame,
    bismark_path: Path,
    results_path: Path,
    plots_path: Path,
    exp_prefix: str,
    contrast_factor: str,
    contrast_levels: Tuple[str, str],
    genome: str = "hg38",
    n_threads: int = 16,
    q_ths: Iterable[float] = (0.05, 0.01),
    mean_diff_levels: Iterable[str] = ("hyper", "hypo", "all"),
    mean_diff_ths: Iterable[float] = (10, 20, 30),
) -> None:
    """
    Differential methylation analysis of RRBS samples focusing on sites.

    Args:
        annot_df: Genome annotation dataframe.
        bismark_path: Path to bismark directory with coverage files.
        results_path: Path to store all generated results.
        plots_path: Path to store all generated plots.
        exp_prefix: Prefix string for all generated files.
        contrast_factor: Field name defining contrast levels.
        contrast_levels: Contrast to test for differential methylation.
        genome: Genome version.
        n_threads: Number of threads to use to calculate differential methylation.
        q_ths: FDR thresholds to use to filter the results.
        mean_diff_levels: Mean methylation difference levels to filter the results by.
        mean_diff_ths: Mean methylation difference thresholds to filter the results by.
    """
    # 0. Setup
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
    test, control = contrast_levels

    # 1. Get methylation coverage files
    samples_cov_files = {
        sample_id: cov_file
        for cov_file in sorted(bismark_path.glob("**/*.bismark.cov.gz"))
        if (sample_id := cov_file.name.split(".")[0]) in annot_df.index
    }
    assert len(samples_cov_files) > 0, "No coverage files were found."
    sample_ids = list(samples_cov_files.keys())

    # 1.1. Filter annotation dataframe by available samples
    annot_df = annot_df.loc[sample_ids]
    annot_df["sample_groups"] = [
        1 if sample_condition == test else 0
        for sample_condition in annot_df[contrast_factor]
    ]

    condition_samples = {}
    for k, v in annot_df[contrast_factor].to_dict().items():
        condition_samples.setdefault(v, []).append(k)

    # 2. Obtain methylation calls data
    methyl_calls = meth_read(
        files=[samples_cov_files[sample_id] for sample_id in sample_ids],
        sample_ids=sample_ids,
        assembly=genome,
        treatment=ro.IntVector(annot_df["sample_groups"].tolist()),
        pipeline="bismarkCoverage",
    )
    methyl_calls.names = sample_ids
    logging.info("Finished loading methylation calls data.")

    # 2.1. Compute and save stats
    plot_save_stats(
        methyl_calls,
        f"{exp_prefix}_methyl_calls_raw_{test}_vs_{control}",
        plots_path,
        results_path,
    )
    logging.info("Finished statistics and plots for methylation calls data.")

    # 3. Prepare data for differential methylation
    methyl_obj_filt = filter_by_coverage(methyl_calls, lo_count=15, hi_perc=99.9)
    del methyl_calls
    plot_save_stats(
        methyl_obj_filt,
        f"{exp_prefix}_methyl_calls_filt_{test}_vs_{control}",
        plots_path,
        results_path,
    )
    logging.info("Finished statistics and plots for filtered methylation calls data.")

    methyl_obj_filt_united = unite(
        methyl_obj_filt, destrand=False, **{"mc.cores": n_threads}
    )
    del methyl_obj_filt
    # make_meth_plots(
    #     methyl_obj_filt_united,
    #     f"{exp_prefix}_{test}_vs_{control}",
    #     condition_samples,
    #     plots_path,
    #     results_path,
    # )
    logging.info("Finished statistics and plots for joint methylation calls data.")

    # 4. Differential methylation analysis
    annot_df_contrasts = deepcopy(
        annot_df[annot_df[contrast_factor].isin((test, control))]
    )

    # 4.1. Reorganize methyl object for the goal comparison
    methyl_obj_diff = reorganize(
        methyl_obj=methyl_obj_filt_united,
        sample_ids=ro.StrVector(annot_df_contrasts.index.tolist()),
        treatment=ro.IntVector(annot_df_contrasts["sample_groups"].tolist()),
    )
    save_rds(
        methyl_obj_diff, results_path.joinpath(f"{exp_prefix}_methyl_obj_diff.RDS")
    )

    # 4.2. Calculate differential methylation
    logging.info(
        "Running differential methylation analysis for "
        f"{test} (n={len(condition_samples[test])}) vs "
        f"{control} (n={len(condition_samples[control])})"
    )
    diff_res = calculate_diff_meth(methyl_obj_diff, **{"mc.cores": n_threads})
    del methyl_obj_diff
    logging.info("Finished differential methylation call.")

    # 4.3. Save results
    save_csv(
        diff_res,
        results_path.joinpath(f"{exp_prefix}_diff_meth_{test}_vs_{control}.csv"),
    )
    save_rds(
        diff_res,
        results_path.joinpath(f"{exp_prefix}_diff_meth_{test}_vs_{control}.RDS"),
    )

    # 4.4. Differential methylation plots
    # make_diff_meth_plots(
    #     methyl_obj=methyl_obj_filt_united,
    #     methyl_diffs={(test, control): diff_res},
    #     exp_prefix=exp_prefix,
    #     plots_path=plots_path,
    #     cpgs_ann=annots["cpg"],
    #     condition_samples=condition_samples,
    # )
    # logging.info("Finished differential methylation plots.")

    # 4.5. Annotate unfiltered differential results
    for annot_type in ("cpg", "gene"):
        try:
            # 4.5.1. Annotate regions
            regions = homogeinize_seqlevels_style(
                make_granges_from_dataframe(
                    ro.r("data.frame")(diff_res), keep_extra_columns=True
                ),
                annots[annot_type],
            )
            diff_res_ann = annotate_regions(
                regions=regions,
                annotations=annots[annot_type],
                ignore_strand=True,
                quiet=False,
            )
            del regions

            # 4.5.2. Save to disk
            diff_res_ann_df = rpy2_df_to_pd_df_manual(diff_res_ann).replace(
                "NA_character_", np.nan
            )
            diff_res_ann_df.to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_ann_{annot_type}s.csv"
                ),
            )
            for annot_subtype, df in diff_res_ann_df.groupby("annot.type"):
                df.to_csv(
                    results_path.joinpath(
                        f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                        f"ann_{annot_type}s_{annot_subtype}.csv"
                    )
                )

            # 4.5.3. Save annotations summary
            save_csv(
                ro.r("data.frame")(
                    summarize_annotations(annotated_regions=diff_res_ann)
                ),
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_ann_{annot_type}s"
                    "_summary.csv"
                ),
            )

            # 4.5.4. Plot annotations
            plot_annotation(
                annotated_regions=diff_res_ann,
                save_path=plots_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_ann_{annot_type}s"
                    "_plot.pdf"
                ),
                annotation_order=annots_order,
                plot_title="# of Sites Tested for DM",
                x_label="Known Annotations",
                y_label="Count",
            )
            del diff_res_ann

        except RRuntimeError as e:
            logging.warning(e)
    logging.info("Finished annotation of unfiltered differential methylation results.")

    # 5. Filter differential methylation results
    diff_res_filts = dict()
    for q_th, mean_diff, mean_diff_level in product(
        q_ths, mean_diff_ths, mean_diff_levels
    ):
        # 5.1. Filter results
        diff_res_filt = get_methyl_diff(
            diff_res, difference=mean_diff, qvalue=q_th, type=mean_diff_level
        )

        # 5.2. Save to disk
        q_th_str = str(q_th).replace(".", "_")
        mean_diff_str = str(mean_diff).replace(".", "_")
        save_csv(
            diff_res_filt,
            results_path.joinpath(
                f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                f"{mean_diff_level}_q_value_{q_th_str}_diff_{mean_diff_str}.csv"
            ),
        )

        # 5.3. Collect stats
        diff_res_filts[(q_th, mean_diff, mean_diff_level)] = diff_res_filt

    pd.Series({k: len(v.rx2("chr")) for k, v in diff_res_filts.items()}).rename(
        "diff_cpgs"
    ).to_csv(
        results_path.joinpath(f"{exp_prefix}_diff_meth_{test}_vs_{control}_summary.csv")
    )
    logging.info("Finished differential methylation filtering.")

    # 6. Annotate differentially methylated cpgs and genes
    diff_res_filts_ann = dict()
    for (
        (
            q_th,
            mean_diff,
            mean_diff_level,
        ),
        diff_res_filt,
    ), annot_type in product(diff_res_filts.items(), ("cpg", "gene")):
        try:
            # 6.1. Annotate regions
            regions = homogeinize_seqlevels_style(
                make_granges_from_dataframe(
                    ro.r("data.frame")(diff_res_filt), keep_extra_columns=True
                ),
                annots[annot_type],
            )
            diff_res_filt_ann = annotate_regions(
                regions=regions,
                annotations=annots[annot_type],
                ignore_strand=True,
                quiet=False,
            )
            diff_res_filts_ann[
                (test, control, q_th, mean_diff, mean_diff_level, annot_type)
            ] = diff_res_filt_ann

            # 6.2. Save to disk
            q_th_str = str(q_th).replace(".", "_")
            mean_diff_str = str(mean_diff).replace(".", "_")
            diff_res_filt_ann_df = rpy2_df_to_pd_df_manual(diff_res_filt_ann).replace(
                "NA_character_", np.nan
            )
            diff_res_filt_ann_df.to_csv(
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                    f"{mean_diff_level}_q_value_{q_th_str}_diff_{mean_diff_str}_"
                    f"ann_{annot_type}s.csv"
                ),
            )
            for annot_subtype in diff_res_filt_ann_df["annot.type"].unique():
                diff_res_filt_ann_df[
                    diff_res_filt_ann_df["annot.type"] == annot_subtype
                ].to_csv(
                    results_path.joinpath(
                        f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                        f"{mean_diff_level}_q_value_{q_th_str}_diff_{mean_diff_str}_"
                        f"ann_{annot_type}s_{annot_subtype}.csv"
                    ),
                )

            # 6.3. Save annotations summary
            save_csv(
                ro.r("data.frame")(
                    summarize_annotations(annotated_regions=diff_res_filt_ann)
                ),
                results_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                    f"{mean_diff_level}_q_value_{q_th_str}_diff_{mean_diff_str}_"
                    f"ann_{annot_type}s_summary.csv"
                ),
            )

            # 6.4. Plot annotations
            plot_annotation(
                annotated_regions=diff_res_filt_ann,
                save_path=plots_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                    f"{mean_diff_level}_q_value_{q_th_str}_diff_{mean_diff_str}_"
                    f"ann_{annot_type}s_plot.pdf"
                ),
                annotation_order=annots_order,
                plot_title="# of Sites Tested for DM",
                x_label="Known Annotations",
                y_label="Count",
            )

        except RRuntimeError as e:
            logging.warning(e)
    logging.info("Finished differential annotation (I).")

    # 7. Plot annotations versus random regions
    for (
        test,
        control,
        q_th,
        mean_diff,
        mean_diff_level,
        annot_type,
    ), diff_res_filt_ann in diff_res_filts_ann.items():
        try:
            # 7.1 Get random regions annotation
            inds = random.choices(
                list(range(1, ro.r("nrow")(methyl_obj_filt_united)[0])),
                k=ro.r("length")(diff_res_filt_ann)[0],
            )
            regions = homogeinize_seqlevels_style(
                make_granges_from_dataframe(
                    ro.r("data.frame")(methyl_obj_filt_united).rx(
                        ro.IntVector(inds), True
                    ),
                    keep_extra_columns=True,
                ),
                annots[annot_type],
            )
            annotated_random = annotate_regions(
                regions=randomize_regions(
                    regions, allow_overlaps=True, per_chromosome=True
                ),
                annotations=annots[annot_type],
                ignore_strand=True,
                quiet=False,
            )

            # 7.2. Plot random regions
            q_th_str = str(q_th).replace(".", "_")
            mean_diff_str = str(mean_diff).replace(".", "_")
            plot_annotation(
                annotated_regions=diff_res_filt_ann,
                annotated_random=annotated_random,
                save_path=plots_path.joinpath(
                    f"{exp_prefix}_diff_meth_{test}_vs_{control}_"
                    f"{mean_diff_level}_q_value_{q_th_str}_diff_{mean_diff_str}_"
                    f"ann_{annot_type}s_rnd_regions_plot.pdf"
                ),
                annotation_order=[annots_order],
                plot_title=(
                    f"{mean_diff_level.title()} Methylated Sites ({annot_type}): "
                    f"{test} vs {control}"
                ),
                x_label="Annotations",
                y_label="Count",
            )
        except RRuntimeError as e:
            logging.warning(e)
    logging.info("Finished differential annotation (II).")
