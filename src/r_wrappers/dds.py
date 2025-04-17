"""
Wrappers for R package DSS

All functions have pythonic inputs and outputs.

DSS (Dispersion Shrinkage for Sequencing) is a statistical approach for
detecting differentially methylated CpG sites and regions from whole genome
bisulfite sequencing (WGBS) and reduced representation bisulfite sequencing
(RRBS) data.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
    R --> data.category
    Python --> data_category
"""

from typing import Any

import pandas as pd
from rpy2.robjects import StrVector
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_source = importr("DSS")


def dml_fit_multi_factor(
    bsseq_obj: Any, design: pd.DataFrame, formula: Any, **kwargs: Any
) -> Any:
    """Fit a linear model for differential methylation analysis with multiple factors.

    This function takes a BSseq object, a data frame for experimental design and a
    model formula and then fits a linear model to identify differential methylation
    while accounting for multiple experimental factors.

    Args:
        bsseq_obj: An object of BSseq class containing the bisulfite sequencing data.
        design: An annotation dataframe for experimental design. Number of rows must
            match the number of columns of the counts in bsseq_obj.
        formula: A formula for the linear model (e.g., ~ Treatment + Batch).
        **kwargs: Additional arguments to pass to the DMLfit.multiFactor function.
            Common parameters include:
            - dispersion: Whether to estimate dispersions.

    Returns:
        Any: A list with the following components:
            - gr: A GRanges object for locations of the CpG sites.
            - design: The input data frame for experimental design.
            - formula: The input formula for the model.
            - X: The design matrix used in regression.
            - fit: The model fitting results containing beta (estimated coefficients),
              var.beta (estimated variance/covariance matrices), and phi (estimated
              beta-binomial dispersion parameters).

    References:
        https://rdrr.io/bioc/DSS/man/DMLfit.multiFactor.html
    """
    return r_source.DMLfit_multiFactor(
        bsseq_obj, pd_df_to_rpy2_df(design), formula, **kwargs
    )


def dml_test_multi_factor(dml_fit: Any, **kwargs: Any) -> pd.DataFrame:
    """Perform statistical tests for differential methylation using multifactor model.

    This function takes the linear model fitting results from dml_fit_multi_factor and
    performs Wald test at each CpG site, then returns test statistics, p-values and FDR.

    Args:
        dml_fit: Result object returned from 'dml_fit_multi_factor' function.
        **kwargs: Additional arguments to pass to the DMLtest.multiFactor function.
            Common parameters include:
            - coef: Which coefficient to test (default: 2, which tests the first treatment effect).
            - term: Test a specific term in the design matrix.

    Returns:
        pd.DataFrame: A data frame with each row corresponding to a CpG site, containing:
            - chr: Chromosome number.
            - pos: Genomic coordinates.
            - stat: Wald statistics.
            - pval: P-values (obtained from normal distribution).
            - fdr: False discovery rate.

    References:
        https://rdrr.io/bioc/DSS/man/DMLtest.multiFactor.html
    """
    return r_source.DMLtest_multiFactor(dml_fit, **kwargs)


def dml_test(
    bsseq_obj: Any, group1: StrVector, group2: StrVector, **kwargs: Any
) -> pd.DataFrame:
    """Perform statistical tests for differential methylation between two groups.

    This function takes a BSseq object and two group labels, then performs statistical
    tests for differential methylation at each CpG site using a beta-binomial model.

    Args:
        bsseq_obj: An object of BSseq class containing the bisulfite sequencing data.
        group1: Vector of sample names or indexes for the first group to be
            tested.
        group2: Vector of sample names or indexes for the second group to
            be tested.
        **kwargs: Additional arguments to pass to the DMLtest function.
            Common parameters include:
            - smoothing: Whether to perform smoothing on the methylation levels.
            - smoothing.span: Smoothing span parameter (default: 500).
            - equal.disp: Whether to assume equal dispersion (default: FALSE).

    Returns:
        pd.DataFrame: A data frame with each row corresponding to a CpG site, containing:
            - chr: Chromosome number.
            - pos: Genomic coordinates.
            - mu1, mu2: Mean methylations of two groups.
            - diff: Difference of mean methylations of two groups (diff=mu1-mu2).
            - diff.se: Standard error of the methylation difference.
            - stat: Wald statistics.
            - pval: P-values (obtained from normal distribution).
            - fdr: False discovery rate.

    References:
        https://rdrr.io/bioc/DSS/man/DMLtest.html
    """
    return r_source.DMLtest(bsseq_obj, group1, group2, **kwargs)


def call_dml(dml_result: Any, **kwargs: Any) -> pd.DataFrame:
    """Call differentially methylated loci (DMLs) from test results.

    This function takes the results from a DML testing procedure ('dml_test' function) and
    identifies statistically significant differentially methylated loci (DMLs).

    Args:
        dml_result: A data frame representing the results for DML detection.
            This should be the result returned from 'dml_test' or
            'dml_test_multi_factor' function.
        **kwargs: Additional arguments to pass to the callDML function.
            Common parameters include:
            - delta: Minimum differential methylation threshold (default: 0).
            - p.threshold: P-value threshold for statistical significance (default: 0.001).
            - keepComp: Whether to keep computational results (default: FALSE).

    Returns:
        pd.DataFrame: A data frame for DMLs. Each row represents one DML, sorted by
        statistical significance. The columns include:
        - chr: Chromosome number.
        - pos: Genomic coordinates.
        - mu1, mu2: Mean methylations of two groups.
        - diff: Difference of mean methylations of two groups.
        - diff.se: Standard error of the methylation difference.
        - stat: Wald statistics.
        - phi1, phi2: Estimated dispersions in two groups.
        - pval: P-values.
        - fdr: False discovery rate.
        - postprob.overThreshold: The posterior probability of the methylation
          difference exceeding delta (only available when delta > 0).

    References:
        https://rdrr.io/bioc/DSS/man/callDML.html
    """
    return r_source.callDML(dml_result, **kwargs)


def call_dmr(dml_result: Any, **kwargs: Any) -> pd.DataFrame:
    """Call differentially methylated regions (DMRs) from test results.

    This function takes the results from a DML testing procedure and identifies
    differentially methylated regions (DMRs). Regions with CpG sites that are
    statistically significant are detected as DMRs. Nearby DMRs are merged into
    longer ones based on specified criteria.

    Args:
        dml_result: A data frame representing the results for DML detection.
            This should be the result returned from 'dml_test' or
            'dml_test_multi_factor' function.
        **kwargs: Additional arguments to pass to the callDMR function.
            Common parameters include:
            - delta: Minimum differential methylation threshold (default: 0).
            - p.threshold: P-value threshold for statistical significance (default: 0.001).
            - minlen: Minimum length of DMR in base pairs (default: 50).
            - minCG: Minimum number of CpG sites in a DMR (default: 3).
            - dis.merge: Maximum distance between DMRs for merging (default: 50).
            - pct.sig: Minimum percentage of CpG sites in a DMR that are significant (default: 0.5).

    Returns:
        pd.DataFrame: A data frame for DMRs. Each row represents one DMR, sorted by
        "areaStat" (sum of test statistics). The columns include:
        - chr: Chromosome number.
        - start, end: Genomic coordinates of the DMR.
        - length: Length of the DMR in base pairs.
        - nCG: Number of CpG sites contained in the DMR.
        - meanMethy1, meanMethy2: Average methylation levels in two conditions.
        - diff.Methy: Difference in methylation levels between conditions (meanMethy1-meanMethy2).
        - areaStat: Sum of test statistics of all CpG sites within the DMR.

    References:
        https://rdrr.io/bioc/DSS/man/callDMR.html
    """
    return r_source.callDMR(dml_result, **kwargs)
