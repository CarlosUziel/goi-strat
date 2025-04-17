import functools
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

from data.utils import parallelize_map
from utils import run_func_dict


def get_gene_occurrence_stats(
    gene_sets_df: pd.DataFrame, mode: str = "relative"
) -> pd.Series:
    """Compute gene occurrence statistics for a collection of gene sets.

    Args:
        gene_sets_df (pd.DataFrame): A collection of gene sets with metadata.
        mode (str): Whether to compute absolute or relative gene occurrence. Options are 'absolute' or 'relative'.

    Returns:
        pd.Series: A pandas Series with the requested statistic per gene.

    Raises:
        ValueError: If the mode is not 'absolute' or 'relative'.
    """
    genes_occurrence = gene_sets_df["gene_symbol"].str.get_dummies(sep="/")

    if mode == "relative":
        return genes_occurrence.apply(lambda x: np.sum(x) / len(x)).rename(
            "relative_occurrence"
        )
    elif mode == "absolute":
        return genes_occurrence.sum().rename("total_occurrence")
    else:
        raise ValueError(f"{mode} is not a valid option.")


def compute_degss_genes_stats(
    degss_cats_dfs: Dict[str, pd.DataFrame],
    msigdb_dfs: Dict[str, pd.DataFrame],
    save_dir: Path,
    bootstrap_iterations: int = 100,
    processes: int = 32,
) -> None:
    """Given multiple collections of gene sets, one per MSigDB category, compute gene
    occurrence statistics corrected by Z-Score normalization using bootstrapping.

    More concretely, for each MSigDB category, we randomly sample N gene sets from all
        gene sets in that category, where N is the current number of DEGSs. We do this
        M times. Then, we compute the absolute and relative gene occurrence of all
        genes within the gene sets in each M bootstrapping iteration. Next, we compute
        the mean and standard deviation of the occurence stats over the M iterations.
        Finally, we use the mean and std. dev. to compute the Z-score normalization
        of the occurence stats of the original DEGSs.

    Args:
        degss_cats_dfs (Dict[str, pd.DataFrame]): A mapping of MSigDB categories and the current DEGSs detected in that category.
        msigdb_dfs (Dict[str, pd.DataFrame]): A mapping of MSigDB categories and the metadata of all gene sets in that category.
        save_dir (Path): Directory to save results to.
        bootstrap_iterations (int): Number of bootstrap random resamplings to perform. Default is 100.
        processes (int): Number of processes used to compute statistics in parallel. Default is 32.

    Returns:
        None
    """
    # 1. Process each MSigDB category
    z_scores_dfs = {}
    z_scores_means = {}
    for msigdb_cat, degss_df in degss_cats_dfs.items():
        n_sets = len(degss_df)
        msigdb_cat_dir = save_dir.joinpath(msigdb_cat)
        msigdb_cat_dir.mkdir(exist_ok=True, parents=True)

        # 1.1. Get DEGSs gene occurrence statistics
        degss_gene_stats = get_gene_occurrence_stats(degss_df, mode="relative")

        # 1.2. Get bootstrapped gene occurrence statistics
        bs_gene_stats = pd.concat(
            parallelize_map(
                func=functools.partial(run_func_dict, func=get_gene_occurrence_stats),
                inputs=[
                    {
                        "gene_sets_df": msigdb_dfs[msigdb_cat].sample(n_sets),
                        "mode": "relative",
                    }
                    for _ in range(bootstrap_iterations)
                ],
                processes=processes,
            ),
            axis=1,
        ).fillna(0)
        bs_gene_stats.columns = list(range(bootstrap_iterations))
        bs_gene_stats.to_csv(
            msigdb_cat_dir.joinpath("gene_occurrence_stats_bootstrap.csv")
        )

        bs_gene_stats_summary = bs_gene_stats.agg(("mean", "std"), axis=1)
        bs_gene_stats_summary.sort_values("mean", ascending=False).to_csv(
            msigdb_cat_dir.joinpath("gene_occurrence_stats_bootstrap_summary.csv")
        )
        z_scores_means[msigdb_cat] = deepcopy(bs_gene_stats_summary["mean"])

        # 1.3. Perform Z-Score normalization
        z_scores_dfs[msigdb_cat] = (
            degss_gene_stats - bs_gene_stats_summary.loc[degss_gene_stats.index, "mean"]
        ) / bs_gene_stats_summary.loc[degss_gene_stats.index, "std"]
        z_scores_dfs[msigdb_cat].rename("z_score").sort_values(
            ascending=False, key=abs
        ).to_csv(msigdb_cat_dir.joinpath("gene_occurrence_stats_bootstrap_z_score.csv"))

    gene_stats_all = pd.DataFrame(z_scores_dfs).fillna(0)
    gene_stats_all.to_csv(
        save_dir.joinpath("gene_occurrence_stats_bootstrap_z_score_all.csv")
    )

    # 2. Summarize gene occurrence statistics for all categories
    # 2.1. Get mean absolute Z-Score per gene via a weighted sum of mean occurrences
    z_scores_means_df = pd.DataFrame(z_scores_means).fillna(0)
    gene_stats_all_summary = (
        (
            (gene_stats_all.abs() * z_scores_means_df).sum(axis=1)
            / z_scores_means_df.sum(axis=1)
        )
        .rename("mean_z_score")
        .to_frame()
    )

    # 2.2. Get P-value and P-value adjusted of the average Z-Score
    gene_stats_all_summary["p_value"] = gene_stats_all_summary["mean_z_score"].map(
        lambda x: ss.norm.sf(abs(x)) * 2
    )
    gene_stats_all_summary["p_value_adj"] = multipletests(
        gene_stats_all_summary["p_value"], method="fdr_bh"
    )[1]

    # 3. Sort and save final results
    gene_stats_all_summary.sort_values("p_value", ascending=True).to_csv(
        save_dir.joinpath("gene_occurrence_stats_bootstrap_z_score_summary.csv")
    )


def add_degss_genes_stats_metadata(
    save_dir: Path,
    gene_stats_path: Path,
    diff_expr_df: Optional[pd.DataFrame] = None,
    diff_meth_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Add differential expression and/or methylation metadata to DEGSs genes.

    Args:
        save_dir (Path): Directory to save results to.
        gene_stats_path (Path): Path to gene stats file.
        diff_expr_df (Optional[pd.DataFrame]): Dataframe with differential expression statistics and metrics. Default is None.
        diff_meth_df (Optional[pd.DataFrame]): Dataframe with differential methylation statistics and metrics. Default is None.

    Returns:
        None
    """
    # 0. Setup
    gene_stats_summary = pd.read_csv(gene_stats_path, index_col=0)

    # 1. Differential expression metrics
    if diff_expr_df is not None:
        diff_expr_df.set_index("SYMBOL", inplace=True)
        ann_cols = ["ENTREZID", "GENENAME", "GENETYPE"]
        diff_expr_df.columns = [
            "diff_expr_" + c if c not in ann_cols else c for c in diff_expr_df.columns
        ]
        gene_stats_summary = gene_stats_summary.join(diff_expr_df)

    # 2. Differential methylation metrics
    if diff_meth_df is not None:
        diff_meth_df.columns = ["diff_meth_" + c for c in diff_meth_df.columns]
        gene_stats_summary = gene_stats_summary.join(diff_meth_df)

    # 3. Sort and save final results
    gene_stats_summary.sort_values("p_value", ascending=True).to_csv(
        save_dir.joinpath("gene_occurrence_stats_bootstrap_z_score_summary_ann.csv")
    )
