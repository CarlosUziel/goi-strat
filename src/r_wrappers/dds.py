"""
    Wrappers for R package DSS

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

from typing import Any

import pandas as pd
from rpy2.robjects import StrVector
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_source = importr("DSS")


def dml_fit_multi_factor(
    bsseq_obj: Any, design: pd.DataFrame, formula: Any, **kwargs
) -> Any:
    """
    This function takes a BSseq object, a ann_df frame for experimental design and a
        model formula and then fit a linear model.

    See: https://rdrr.io/bioc/DSS/man/DMLfit.multiFactor.html

    Args:
        bsseq_obj: An object of BSseq class for the BS-seq ann_df.
        design: An annotation dataframe for experimental design. Number of rows must
            match the number of columns of the counts in bsseq_obj.
        formula: A formula for the linear model.

    Returns:
        A list with following components:
            gr: An object of 'GRanges' for locations of the CpG sites.
            design: The input data frame for experimental design.
            formula: The input formula for the model.
            X: The design matrix used in regression. It is created based on design and
                formula.
            fit: The model fitting results. This is a list itself, with three
                components: 'beta' - the estimated coefficients; 'var.beta' - estimated
                variance/covariance matrices for beta. 'phi' - estimated beta-binomial
                dispersion parameters. Note that var.beta for a CpG site should be a
                ncol(X) x ncol(X) matrix, but is flattend to a vector so that the
                matrices for all CpG sites can be saved as a matrix.
    """
    return r_source.DMLfit_multiFactor(
        bsseq_obj, pd_df_to_rpy2_df(design), formula, **kwargs
    )


def dml_test_multi_factor(dml_fit: Any, **kwargs) -> Any:
    """
    This function takes the linear model fitting results and performs Wald test at each
        CpG site, then return test statistics, p-values and FDR.

    See: https://rdrr.io/bioc/DSS/man/DMLtest.multiFactor.html

    Args:
        dml_fit: Result object returned from 'DMLfit.multiFactor' function.

    """
    return r_source.DMLtest_multiFactor(dml_fit, **kwargs)


def dml_test(bsseq_obj: Any, group1: StrVector, group2: StrVector, **kwargs) -> Any:
    """
    This function takes a BSseq object and two group labels, then perform statistical
        tests for differential methylation at each CpG site.

    See: https://rdrr.io/github/haowulab/DSS/man/DMLtest.html

    Args:
        bsseq_obj: An object of BSseq class for the BS-seq ann_df.
        group1: Vectors of sample names or indexes for the first group to be
            tested. See more description in details.
        group2: Vectors of sample names or indexes for the second group to
            be tested. See more description in details.

    Returns:
        A data frame with each row corresponding to a CpG site. Rows are sorted by
            chromosome number and genomic coordinates. The columns include:
            chr: Chromosome number.
            pos: Genomic coordinates.
            mu1, mu2: Mean methylations of two groups.
            diff: Difference of mean methylations of two groups. diff=mu1-mu2.
            diff.se: Standard error of the methylation difference.
            stat: Wald statistics.
            pval: P-values. This is obtained from normal distribution.
            fdr: False discovery rate.
    """
    return r_source.DMLtest(bsseq_obj, group1, group2, **kwargs)


def call_dml(dml_result: Any, **kwargs) -> Any:
    """
    This function takes the results from DML testing procedure ('DMLtest' function) and
        calls DMLs. Regions will CpG sites being statistically significant are deemed as
        DMLs.

    See: https://rdrr.io/github/haowulab/DSS/man/callDML.html

    Args:
        dml_result: A ann_df frame representing the results for DML detection.
        This should be the result returned
            from 'DMLtest' function.

    Returns:
        A data frame for DMLs. Each row is for a DML. DMLs are sorted by statistical
            significance. The columns are:
            chr: Chromosome number.
            pos: Genomic coordinates.
            mu1, mu2: Mean methylations of two groups.
            diff: Difference of mean methylations of two groups.
            diff.se: Standard error of the methylation difference.
            stat: Wald statistics.
            phi1, phi2: Estimated dispersions in two groups.
            pval: P-values. This is obtained from normal distribution.
            fdr: False discovery rate.
            postprob.overThreshold: The posterior probability of the difference in
                methylation greater than delta. This columns is only available when
                delta>0.
    """
    return r_source.callDML(dml_result, **kwargs)


def call_dmr(dml_result: Any, **kwargs) -> Any:
    """
    This function takes the results from DML testing procedure ('DMLtest'
    function) and calls DMRs. Regions will CpG
        sites being statistically significant are detected as DMRs. Nearby
        DMRs are merged into longer ones. Some
        restrictions including the minimum length, minimum number of CpG
        sites, etc. are applied.

    See: https://rdrr.io/github/haowulab/DSS/man/callDMR.html

    Args:
        dml_result: A ann_df frame representing the results for DML detection.
        This should be the result returned from 'DMLtest' or 'DMLtest.multiFactor'
            function.

    Returns:
        A data frame for DMRs. Each row is for a DMR. Rows are sorted by "areaStat",
            which is the sum of test statistics of all CpG sites in the region. The
            columns are:
            chr: Chromosome number.
            start, end: Genomic coordinates.
            length: Length of the DMR, in bps.
            nCG: Number of CpG sites contained in the DMR.
            meanMethy1, meanMethy2: Average methylation levels in two conditions.
            diff.Methy: The difference in the methylation levels between two conditions.
                diff.Methy=meanMethy1-meanMethy2.
            areaStat: The sum of the test statistics of all CpG sites within the DMR.
    """
    return r_source.callDMR(dml_result, **kwargs)
