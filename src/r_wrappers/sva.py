"""
Wrappers for R package sva (Surrogate Variable Analysis)

All functions have pythonic inputs and outputs.

The sva package contains functions for removing batch effects and unwanted variation
in high-throughput genomic data, including surrogate variable analysis, ComBat batch
adjustment, and related methods.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_sva = importr("sva")


def combat_seq(
    counts_df: pd.DataFrame, batch: Iterable[int], **kwargs: Any
) -> pd.DataFrame:
    """Adjust RNA-Seq count data for batch effects using ComBat_seq.

    ComBat_seq is an improved model from ComBat using negative binomial regression,
    which specifically targets RNA-Seq count data. This method adjusts for known
    batch effects in high-throughput genomic data.

    Args:
        counts_df: Counts dataframe (genes x samples) where rows represent genes
            and columns represent samples.
        batch: Integer vector indicating batch labels for each sample. Must be
            the same length as the number of columns in counts_df.
        **kwargs: Additional arguments to pass to the ComBat_seq function.
            Common parameters include:
            - group: Factor specifying biological group for each sample.
            - covar_mod: Model matrix for other covariates to adjust for.
            - full_mod: Whether to use a full model (default: False).

    Returns:
        pd.DataFrame: Raw gene counts adjusted for batch effects, returned as a
        pandas DataFrame with integer values. The dimensions match the input
        counts_df with genes as rows and samples as columns.

    References:
        Zhang Y, Parmigiani G, Johnson WE. ComBat-seq: batch effect adjustment
        for RNA-seq count data. NAR Genomics and Bioinformatics. 2020.
    """
    return rpy2_df_to_pd_df(
        r_sva.ComBat_seq(
            ro.r("as.matrix")(pd_df_to_rpy2_df(counts_df)),
            ro.IntVector(batch),
            full_mod=False,
            **kwargs,
        )
    ).astype(int)


def combat(
    data_df: pd.DataFrame,
    batch: Iterable[int],
    mod: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Adjust for batch effects using ComBat.

    ComBat harmonizes data from different batches using an empirical Bayes framework.
    It adjusts for known batches using a parametric empirical Bayesian framework,
    making it particularly effective for small sample sizes.

    Args:
        data_df: Data matrix (features x samples) where rows represent features
            (genes, proteins, etc.) and columns represent samples.
        batch: Integer vector indicating batch labels for each sample. Must be
            the same length as the number of columns in data_df.
        mod: Model matrix for outcomes of interest and other covariates to adjust for.
            If None, an intercept-only model is used.
        **kwargs: Additional arguments to pass to the ComBat function.
            Common parameters include:
            - par_prior: Whether to use parametric empirical priors (default: True).
            - mean_only: Whether to adjust only the mean (default: False).
            - ref_batch: Reference batch to adjust to (default: None).
            - BPPARAM: BiocParallel parameter for parallel processing.

    Returns:
        pd.DataFrame: Data matrix adjusted for batch effects, returned as a
        pandas DataFrame. The dimensions match the input data_df with features
        as rows and samples as columns.

    Notes:
        Unlike ComBat_seq, standard ComBat is designed for continuous data
        (e.g., microarray, log-transformed RNA-seq) and not raw count data.

    References:
        Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
        expression data using empirical Bayes methods. Biostatistics. 2007.
    """
    # Convert inputs to R objects
    expr_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(data_df))
    batch_vector = ro.StrVector([str(b) for b in batch])

    # Create model matrix if not provided
    if mod is None:
        mod = ro.r("model.matrix")(
            ro.Formula("~1"), data=ro.r("data.frame")(x=ro.IntVector([1] * len(batch)))
        )
    else:
        numpy2ri.activate()
        mod = ro.r("as.matrix")(mod)
        numpy2ri.deactivate()

    # Run ComBat
    combat_data = r_sva.ComBat(dat=expr_matrix, batch=batch_vector, mod=mod, **kwargs)

    # Convert back to pandas DataFrame
    return rpy2_df_to_pd_df(combat_data)


def sva_estimate(
    data_df: pd.DataFrame,
    mod: np.ndarray,
    mod_0: Optional[np.ndarray] = None,
    n_sv: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[int, np.ndarray]:
    """Estimate the number and values of surrogate variables.

    Surrogate Variable Analysis (SVA) identifies and estimates unwanted sources
    of variation in high-throughput experiments. This function estimates both the
    number of surrogate variables needed and their values.

    Args:
        data_df: Data matrix (features x samples) where rows represent features
            (genes, proteins, etc.) and columns represent samples.
        mod: Full model matrix including covariates of interest (must include intercept).
        mod_0: Null model matrix excluding covariates of interest. If None,
            an intercept-only model is constructed.
        n_sv: Number of surrogate variables to estimate. If None, the function
            will estimate this number using the algorithm from Buja and Eyuboglu 1992.
        **kwargs: Additional arguments to pass to the sva function.
            Common parameters include:
            - method: "irw" (default) or "two-step", the method for estimating surrogate variables.
            - B: Number of permutations for estimating the number of surrogate variables.
            - numSVmethod: Method for estimating number of SVs ("be" or "leek").

    Returns:
        tuple: A tuple containing:
            - n_sv: Estimated number of surrogate variables.
            - sv: Matrix of surrogate variables (dimensions: samples x n_sv).

    Notes:
        The surrogate variables can be included as covariates in downstream analyses
        to adjust for unwanted variation.

    References:
        Leek JT and Storey JD. Capturing heterogeneity in gene expression studies
        by surrogate variable analysis. PLoS Genetics. 2007.
    """
    # Convert inputs to R objects
    expr_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(data_df))

    numpy2ri.activate()
    mod_matrix = ro.r("as.matrix")(mod)

    # Create mod0 if not provided
    if mod_0 is None:
        mod_0 = ro.r("model.matrix")(
            ro.Formula("~1"),
            data=ro.r("data.frame")(x=ro.IntVector([1] * mod.shape[0])),
        )
    else:
        mod_0 = ro.r("as.matrix")(mod_0)
    numpy2ri.deactivate()

    # Run SVA
    sva_result = r_sva.sva(
        dat=expr_matrix, mod=mod_matrix, mod0=mod_0, n_sv=n_sv, **kwargs
    )

    # Extract results
    n_sv_est = sva_result.rx2("n.sv")[0]

    # Convert SV matrix to numpy array
    numpy2ri.activate()
    sv_matrix = np.array(sva_result.rx2("sv"))
    numpy2ri.deactivate()

    return n_sv_est, sv_matrix


def num_sv(
    data_df: pd.DataFrame,
    mod: np.ndarray,
    mod_0: Optional[np.ndarray] = None,
    method: str = "be",
    **kwargs: Any,
) -> int:
    """Estimate the number of surrogate variables needed.

    This function estimates the number of surrogate variables needed to capture
    the unwanted variation in a dataset without estimating the variables themselves.

    Args:
        data_df: Data matrix (features x samples) where rows represent features
            (genes, proteins, etc.) and columns represent samples.
        mod: Full model matrix including covariates of interest (must include intercept).
        mod_0: Null model matrix excluding covariates of interest. If None,
            an intercept-only model is constructed.
        method: Method for estimating the number of surrogate variables:
            - "be": Based on permutation (Buja and Eyuboglu 1992).
            - "leek": Based on proportion of variance explained.
        **kwargs: Additional arguments to pass to the num.sv function.
            Common parameters include:
            - vfilter: Proportion of features to filter by variance (default: 0).
            - B: Number of permutations for the "be" method (default: 20).
            - seed: Random seed for the permutation.

    Returns:
        int: Estimated number of surrogate variables needed to capture
        unwanted variation in the dataset.

    Notes:
        This function is faster than sva_estimate when only the number of
        surrogate variables is needed, not their actual values.

    References:
        Buja A and Eyuboglu N. Remarks on Parallel Analysis. Multivariate
        Behavioral Research. 1992.
    """
    # Convert inputs to R objects
    expr_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(data_df))

    numpy2ri.activate()
    mod_matrix = ro.r("as.matrix")(mod)

    # Create mod0 if not provided
    if mod_0 is None:
        mod_0 = ro.r("model.matrix")(
            ro.Formula("~1"),
            data=ro.r("data.frame")(x=ro.IntVector([1] * mod.shape[0])),
        )
    else:
        mod_0 = ro.r("as.matrix")(mod_0)
    numpy2ri.deactivate()

    # Run num.sv
    n_sv = r_sva.num_sv(
        dat=expr_matrix, mod=mod_matrix, method=method, mod0=mod_0, **kwargs
    )

    return n_sv[0]


def f_sva(
    data_df: pd.DataFrame, mod: np.ndarray, sv: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Incorporate surrogate variables into a regression model.

    This function fits a linear model including both the primary variables of interest
    and surrogate variables, returning the coefficients, standard errors, and p-values
    for the primary variables.

    Args:
        data_df: Data matrix (features x samples) where rows represent features
            (genes, proteins, etc.) and columns represent samples.
        mod: Full model matrix including covariates of interest (must include intercept).
        sv: Matrix of surrogate variables (dimensions: samples x n_sv).
        **kwargs: Additional arguments to pass to the fsva function.
            Common parameters include:
            - method: Fitting method to use ("linear" or "exact").
            - df: Prior degrees of freedom (for empirical Bayes shrinkage).

    Returns:
        np.ndarray: Array of adjusted data with batch effects removed, with the
        same dimensions as the input data.

    References:
        Leek JT et al. The sva package for removing batch effects and other unwanted
        variation in high-throughput experiments. Bioinformatics. 2012.
    """
    # Convert inputs to R objects
    expr_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(data_df))

    numpy2ri.activate()
    mod_matrix = ro.r("as.matrix")(mod)
    sv_matrix = ro.r("as.matrix")(sv)
    numpy2ri.deactivate()

    # Run fsva
    fsva_result = r_sva.fsva(db=expr_matrix, mod=mod_matrix, sv=sv_matrix, **kwargs)

    # Extract and convert results
    numpy2ri.activate()
    adj_data = np.array(fsva_result.rx2("db"))
    numpy2ri.deactivate()

    return adj_data


def twostep_sva(
    data_df: pd.DataFrame,
    mod: np.ndarray,
    mod_0: np.ndarray,
    n_sv: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[int, np.ndarray]:
    """Perform two-step Surrogate Variable Analysis.

    This function implements the original two-step SVA algorithm, which estimates
    surrogate variables in two steps to better capture batch effects and unwanted variation.

    Args:
        data_df: Data matrix (features x samples) where rows represent features
            (genes, proteins, etc.) and columns represent samples.
        mod: Full model matrix including covariates of interest (must include intercept).
        mod_0: Null model matrix excluding covariates of interest.
        n_sv: Number of surrogate variables to estimate. If None, the function
            will estimate this number using the algorithm from Buja and Eyuboglu 1992.
        **kwargs: Additional arguments to pass to the svaseq function.
            Common parameters include:
            - B: Number of permutations for estimating the number of surrogate variables.

    Returns:
        tuple: A tuple containing:
            - n_sv: Estimated number of surrogate variables.
            - sv: Matrix of surrogate variables (dimensions: samples x n_sv).

    References:
        Leek JT and Storey JD. Capturing heterogeneity in gene expression studies
        by surrogate variable analysis. PLoS Genetics. 2007.
    """
    # Convert inputs to R objects
    expr_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(data_df))

    numpy2ri.activate()
    mod_matrix = ro.r("as.matrix")(mod)
    mod0_matrix = ro.r("as.matrix")(mod_0)
    numpy2ri.deactivate()

    # Run two-step SVA
    sva_result = r_sva.sva(
        dat=expr_matrix,
        mod=mod_matrix,
        mod0=mod0_matrix,
        n_sv=n_sv,
        method="two-step",
        **kwargs,
    )

    # Extract results
    n_sv_est = sva_result.rx2("n.sv")[0]

    # Convert SV matrix to numpy array
    numpy2ri.activate()
    sv_matrix = np.array(sva_result.rx2("sv"))
    numpy2ri.deactivate()

    return n_sv_est, sv_matrix


def sva_seq(
    counts_df: pd.DataFrame,
    mod: np.ndarray,
    mod_0: np.ndarray,
    n_sv: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[int, np.ndarray]:
    """Surrogate Variable Analysis for count-based RNA-Seq data.

    This function performs Surrogate Variable Analysis specifically adapted for
    RNA-Seq count data. It identifies and estimates surrogate variables that
    capture unwanted technical variation in RNA-Seq experiments.

    Args:
        counts_df: Count matrix (genes x samples) where rows represent genes
            and columns represent samples.
        mod: Full model matrix including covariates of interest (must include intercept).
        mod_0: Null model matrix excluding covariates of interest.
        n_sv: Number of surrogate variables to estimate. If None, the function
            will estimate this number automatically.
        **kwargs: Additional arguments to pass to the svaseq function.
            Common parameters include:
            - method: Method for SVA ("irw" (default) or "two-step").
            - vfilter: Whether to filter low-count genes (default: TRUE).
            - alpha: Significance level for including genes in SV estimation.

    Returns:
        tuple: A tuple containing:
            - n_sv: Estimated number of surrogate variables.
            - sv: Matrix of surrogate variables (dimensions: samples x n_sv).

    Notes:
        svaseq is specifically designed for RNA-Seq data and accounts for the
        discrete nature and heteroscedasticity of count data.

    References:
        Leek JT. svaseq: removing batch effects and other unwanted noise from
        sequencing data. Nucleic Acids Research. 2014.
    """
    # Convert inputs to R objects
    counts_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(counts_df))

    numpy2ri.activate()
    mod_matrix = ro.r("as.matrix")(mod)
    mod0_matrix = ro.r("as.matrix")(mod_0)
    numpy2ri.deactivate()

    # Run svaseq
    sva_result = r_sva.svaseq(
        dat=counts_matrix, mod=mod_matrix, mod0=mod0_matrix, n_sv=n_sv, **kwargs
    )

    # Extract results
    n_sv_est = sva_result.rx2("n.sv")[0]

    # Convert SV matrix to numpy array
    numpy2ri.activate()
    sv_matrix = np.array(sva_result.rx2("sv"))
    numpy2ri.deactivate()

    return n_sv_est, sv_matrix


def qsva(
    methylation_df: pd.DataFrame,
    mod: np.ndarray,
    mod_0: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Perform quality surrogate variable analysis for methylation arrays.

    qSVA is designed specifically for adjusting for batch effects in
    DNA methylation microarray data. It uses control probes on the array
    to estimate surrogate variables.

    Args:
        methylation_df: Methylation data matrix (probes x samples) where rows
            represent methylation probes and columns represent samples.
        mod: Full model matrix including covariates of interest (must include intercept).
        mod_0: Null model matrix excluding covariates of interest. If None,
            an intercept-only model is constructed.
        **kwargs: Additional arguments to pass to the qsva function.
            Common parameters include:
            - type: Type of probes to use for qSVA ("ctls" or "all").

    Returns:
        np.ndarray: Matrix of quality surrogate variables (dimensions: samples x n_sv).

    References:
        Jaffe AE et al. Bump hunting to identify differentially methylated regions
        in epigenetic epidemiology studies. Int J Epidemiol. 2012.
    """
    # Convert inputs to R objects
    meth_matrix = ro.r("as.matrix")(pd_df_to_rpy2_df(methylation_df))

    numpy2ri.activate()
    mod_matrix = ro.r("as.matrix")(mod)

    # Create mod0 if not provided
    if mod_0 is None:
        mod_0 = ro.r("model.matrix")(
            ro.Formula("~1"),
            data=ro.r("data.frame")(x=ro.IntVector([1] * mod.shape[0])),
        )
    else:
        mod_0 = ro.r("as.matrix")(mod_0)
    numpy2ri.deactivate()

    # Run qsva
    qsva_result = r_sva.qsva(meth=meth_matrix, mod=mod_matrix, mod0=mod_0, **kwargs)

    # Convert result to numpy array
    numpy2ri.activate()
    qsvs = np.array(qsva_result)
    numpy2ri.deactivate()

    return qsvs
