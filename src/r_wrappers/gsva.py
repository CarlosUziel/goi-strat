"""
Wrappers for R package GSVA (Gene Set Variation Analysis)

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any, Dict, Iterable, Union

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from data.utils import supress_stdout
from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_gsva = importr("GSVA")


@supress_stdout
def gsva(
    expr_df: pd.DataFrame,
    gene_sets: Dict[str, Iterable[Union[str, int]]],
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate gene set enrichment scores per sample using GSVA.

    Gene Set Variation Analysis (GSVA) is a non-parametric, unsupervised method
    for estimating variation of gene set enrichment across samples. It transforms
    a gene expression matrix into a gene set enrichment matrix, allowing the evaluation
    of pathway activity over a sample population.

    Args:
        expr_df: A pandas DataFrame containing gene expression data, where rows
            represent genes and columns represent samples. The index should contain
            gene identifiers that match those used in gene_sets.
        gene_sets: A dictionary where keys are gene set names and values are
            iterables of gene identifiers contained in each gene set. The gene
            identifiers should match those in the index of expr_df.
        **kwargs: Additional arguments to pass to the gsva function.
            Common parameters include:
            - method: Method used to calculate the enrichment scores. Options are
              "gsva" (default), "ssgsea", "zscore", or "plage".
            - kcdf: Kernel to use for the cumulative distribution function.
              Options are "Gaussian" (default) or "Poisson".
            - abs_ranking: Whether to use absolute ranking (default: FALSE).
            - min_sz: Minimum size of a gene set (default: 1).
            - max_sz: Maximum size of a gene set (default: Inf).
            - parallel_sz: Number of processors to use for parallel computation (default: 0).
            - mx_diff: Whether to compute the max difference between any two samples
              in a gene set (default: TRUE).
            - tau: Parameter for the "gsva" method (default: 1).
            - ssgsea_norm: Whether to normalize the ssgsea enrichment scores (default: TRUE).
            - verbose: Whether to display progress messages (default: TRUE).

    Returns:
        pd.DataFrame: A pandas DataFrame containing gene set enrichment scores,
        where rows represent gene sets and columns represent samples. The row
        names correspond to the keys in gene_sets, and the column names match
        the column names in expr_df.

    References:
        https://rdrr.io/bioc/GSVA/man/gsva.html
        https://doi.org/10.1186/1471-2105-14-7

    Notes:
        This function uses the @supress_stdout decorator to hide the R console output,
        which can be verbose when running GSVA.
    """
    data_matrix = ro.r("data.matrix")(pd_df_to_rpy2_df(expr_df))
    data_matrix.rownames = ro.StrVector(expr_df.index)
    return rpy2_df_to_pd_df(
        r_gsva.gsva(data_matrix, ro.ListVector(gene_sets), **kwargs)
    )
