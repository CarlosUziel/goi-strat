"""
Wrappers for R package DESeq2

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import rpy2
from rpy2 import robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_deseq2 = importr("DESeq2")
r_stats = importr("stats")


def get_deseq_dataset_matrix(
    counts_matrix: pd.DataFrame,
    annot_df: pd.DataFrame,
    factors: Iterable[str],
    design_factors: Iterable[str],
    **kwargs: Any,
) -> rpy2.robjects.methods.RS4:
    """Create a DESeqDataSet object from a counts matrix.

    This function creates a DESeqDataSet object for differential expression analysis
    from a counts matrix and sample annotation data.

    Args:
        counts_matrix: A matrix representing read counts with shape [n_features, n_samples].
            Rows represent genes/features and columns represent samples.
        annot_df: Annotation dataframe with sample metadata, whose index is the
            sample/replicate name. Must match the column names in counts_matrix.
        factors: Columns from annot_df to be included in the DESeq dataset's colData.
        design_factors: Columns whose values are used in the design formula for
            the differential expression model.
        **kwargs: Additional arguments to pass to the DESeqDataSetFromMatrix function.
            Common parameters include:
            - tidy: Whether the counts are in tidy format.
            - ignoreRank: Whether to ignore rank deficiency in the design matrix.

    Returns:
        rpy2.robjects.methods.RS4: A DESeqDataSet object ready for differential
        expression analysis.

    Raises:
        ValueError: If any of the specified factors are not found in annot_df columns.

    References:
        https://rdrr.io/bioc/DESeq2/man/DESeqDataSet.html
    """
    # 1. Check factors and build design formula
    # 1.1. Check available factors
    available_factors = annot_df.columns
    unavailable_factors = [f for f in factors if f not in available_factors]
    if unavailable_factors:
        raise ValueError(
            "All factors must reference available columns in annot_df. However,"
            f' factors {unavailable_factors}" were not found in annot_df.'
        )

    # 1.2. Build design formula
    def sanitize_factor(factor: str) -> str:
        return re.sub(r"\W|^(?=\d)", "_", factor)

    rename_map = {f: sanitize_factor(f) for f in design_factors}
    design_factors_safe = list(rename_map.values())

    design = Formula("~ " + " + ".join(design_factors_safe))

    # 2. Keep only samples for which annotation is available and vice-versa
    common_samples = counts_matrix.columns.intersection(annot_df.index)
    counts_matrix = counts_matrix.loc[:, common_samples]
    annot_df = annot_df.loc[common_samples, factors]
    annot_df_safe = annot_df.rename(columns=rename_map)

    # 3. Get DESeq dataset
    with localconverter(ro.default_converter):
        return r_deseq2.DESeqDataSetFromMatrix(
            countData=pd_df_to_rpy2_df(counts_matrix),
            colData=pd_df_to_rpy2_df(annot_df_safe),
            design=design,
            **kwargs,
        )


def get_deseq_dataset_htseq(
    annot_df: pd.DataFrame,
    counts_path: Path,
    factors: Iterable[str],
    design_factors: Iterable[str],
    counts_files_pattern: str = "*.tsv",
    **kwargs: Any,
) -> rpy2.robjects.methods.RS4:
    """Create a DESeqDataSet object from HTSeq count files.

    This function creates a DESeqDataSet object for differential expression analysis
    from HTSeq count files and sample annotation data.

    Args:
        annot_df: Annotation dataframe with sample metadata, whose index is the
            sample/replicate name.
        counts_path: Path to the directory containing the HTSeq count files.
        factors: Columns from annot_df to be included in the DESeq dataset's colData.
        design_factors: Columns whose values are used in the design formula for
            the differential expression model.
        counts_files_pattern: Glob pattern to match the HTSeq count files. Default is "*.tsv".
        **kwargs: Additional arguments to pass to the DESeqDataSetFromHTSeqCount function.

    Returns:
        rpy2.robjects.methods.RS4: A DESeqDataSet object ready for differential
        expression analysis.

    Raises:
        ValueError: If any of the specified factors are not found in annot_df columns.
        AssertionError: If no count files are found in the specified path.

    Notes:
        Format of sample table: A data.frame with two or more columns. Each row
        describes one sample. The first column is the sample name and the remaining
        columns are sample metadata which will be stored in colData.

    References:
        https://rdrr.io/bioc/DESeq2/man/DESeqDataSet.html
    """
    # 1. Check factors and build design formula
    # 1.1. Check available factors
    available_factors = annot_df.columns
    unavailable_factors = [f for f in factors if f not in available_factors]
    if unavailable_factors:
        raise ValueError(
            "All factors must reference available columns in annot_df. However,"
            f' factors {unavailable_factors}" were not found in annot_df.'
        )

    # 1.2. Build design formula
    def sanitize_factor(factor: str) -> str:
        return re.sub(r"\W|^(?=\d)", "_", factor)

    rename_map = {f: sanitize_factor(f) for f in design_factors}
    design_factors_safe = list(rename_map.values())
    annot_df_safe = annot_df.rename(columns=rename_map)

    design = Formula("~ " + " + ".join(design_factors_safe))

    # 2. Create sample table including sample name, file name and factors
    # 2.1. Get sample name from file names
    files_names = [f.name for f in counts_path.glob(counts_files_pattern)]
    assert len(files_names) > 0, "No count files found."

    # 2.2. Use dataframe factor index to filter out the files whose samples
    # are present in the dataframe this also does indirect sorting, so files are sorted
    # according to factor index in the table
    files_names_sorted = []
    indx_to_drop = []
    for sample_name in annot_df_safe.index:
        f = [f for f in files_names if sample_name in f]
        if len(f) == 0:
            logging.warning(
                f'\n\nSample "{sample_name}" '
                "was not found amongst available counts "
                "files. Sample will not be processed. \n\n"
            )
            indx_to_drop.append(sample_name)
        else:
            files_names_sorted.append(f[0])
    data_clean = annot_df_safe.drop(indx_to_drop)

    # 2.3. Build sample table
    sample_table = pd.DataFrame()

    # 2.3.1. Add sample name and file names as first and second columns (
    # this order is important!)
    sample_table["replicate"] = data_clean.index
    sample_table["file_name"] = files_names_sorted

    # 2.3.2. Add the rest of the factors
    sample_table.set_index("replicate", inplace=True)
    for factor in factors:
        sample_table[factor] = data_clean[factor]

    # 2.3.2. Build R DataFrame
    with localconverter(ro.default_converter):
        sample_table = pd_df_to_rpy2_df(sample_table.reset_index())

    # 3. Get DESeq dataset
    with localconverter(ro.default_converter):
        return r_deseq2.DESeqDataSetFromHTSeqCount(
            sampleTable=sample_table,
            directory=str(counts_path),
            design=design,
            **kwargs,
        )


def filter_dds(
    dds: rpy2.robjects.methods.RS4, filter_count: int = 1
) -> rpy2.robjects.methods.RS4:
    """Filter out genes with low expression counts.

    This function filters the DESeqDataSet to keep only genes whose average
    expression across samples is greater than the specified threshold.

    Args:
        dds: A DESeqDataSet object.
        filter_count: Minimum average count threshold for keeping genes.
            Default is 1.

    Returns:
        rpy2.robjects.methods.RS4: A filtered DESeqDataSet object with low-expressed
        genes removed.
    """
    f = ro.r(
        """
        f <- function(dds, filter_count) {
            return(dds[rowMeans(counts(dds)) > filter_count,])
        }
        """
    )
    return f(dds, filter_count)


def run_dseq2(
    dds: rpy2.robjects.methods.RS4, **kwargs: Any
) -> rpy2.robjects.methods.RS4:
    """Run the DESeq2 differential expression analysis workflow.

    This function performs a default DESeq2 analysis through the following steps:
    1) Estimation of size factors (normalization)
    2) Estimation of dispersion
    3) Negative Binomial GLM fitting and Wald statistics testing

    Args:
        dds: A DESeqDataSet object.
        **kwargs: Additional arguments to pass to the DESeq function.
            Common parameters include:
            - fitType: Method for dispersion estimation (default: "parametric").
            - test: Statistical test to use ("Wald" or "LRT").
            - betaPrior: Whether to use beta prior for fold changes (default: TRUE).
            - quiet: Whether to suppress messages (default: FALSE).

    Returns:
        rpy2.robjects.methods.RS4: A DESeqDataSet object with results from the
        differential expression analysis.

    References:
        https://rdrr.io/bioc/DESeq2/man/DESeq.html

    Notes:
        After this function returns, results tables (log2 fold changes and p-values)
        can be generated using the `deseq_results` function. Shrunken log fold changes
        can then be generated using the `lfc_shrink` function.
    """
    return r_deseq2.DESeq(dds, **kwargs)


def norm_transform(
    data: rpy2.robjects.methods.RS4, **kwargs: Any
) -> rpy2.robjects.methods.RS4:
    """Apply a simple normalization transformation to count data.

    This function creates a DESeqTransform object by applying a formula to normalized
    counts: f(count(dds, normalized=TRUE) + pc), where pc is a pseudocount.

    Args:
        data: A DESeqDataSet object.
        **kwargs: Additional arguments to pass to the normTransform function.
            Common parameters include:
            - pc: Pseudocount to add to normalized counts (default: 1).
            - f: Function to apply to the normalized counts (default: log).

    Returns:
        rpy2.robjects.methods.RS4: A DESeqTransform object containing the transformed data.

    References:
        https://rdrr.io/bioc/DESeq2/man/normTransform.html
    """
    return r_deseq2.normTransform(data, **kwargs)


def vst_transform(
    dds: rpy2.robjects.methods.RS4, **kwargs: Any
) -> rpy2.robjects.methods.RS4:
    """Apply variance stabilizing transformation to count data.

    This function applies a variance stabilizing transformation (VST) to normalize
    count data, producing values with approximately constant variance across the
    range of mean values. It's useful for visualization, clustering, and other
    downstream analyses. This is a faster implementation that estimates the dispersion
    trend on a subset of genes.

    Args:
        dds: A DESeqDataSet object.
        **kwargs: Additional arguments to pass to the vst function.
            Common parameters include:
            - blind: Whether the transformation should be blind to sample covariates
              (default: TRUE).
            - nsub: Number of genes to use for dispersion trend estimation (default: 1000).
            - fitType: Method used for dispersion trend fitting (default: "parametric").

    Returns:
        rpy2.robjects.methods.RS4: A DESeqTransform object containing the transformed data.

    Notes:
        If the rapid vst function fails, this function will fall back to using the more
        thorough but slower varianceStabilizingTransformation function.

    References:
        https://rdrr.io/bioc/DESeq2/man/vst.html
    """
    try:
        return r_deseq2.vst(dds, **kwargs)
    except RRuntimeError as e:
        logging.warn(e)
        return r_deseq2.varianceStabilizingTransformation(dds, **kwargs)


def rlog_transform(
    dds: rpy2.robjects.methods.RS4, **kwargs: Any
) -> rpy2.robjects.methods.RS4:
    """Apply regularized logarithm transformation to count data.

    This function transforms count data to the log2 scale using a regularized
    logarithm transformation (rlog). It minimizes differences between samples
    for genes with small counts and normalizes with respect to library size.

    Args:
        dds: A DESeqDataSet object.
        **kwargs: Additional arguments to pass to the rlog function.
            Common parameters include:
            - blind: Whether the transformation should be blind to sample covariates
              (default: TRUE).
            - fitType: Method used for dispersion trend fitting (default: "parametric").

    Returns:
        rpy2.robjects.methods.RS4: A DESeqTransform object containing the transformed data.

    Notes:
        The rlog transformation is more robust than VST when size factors vary widely,
        but is generally slower to compute.

    References:
        https://rdrr.io/bioc/DESeq2/man/rlog.html
    """
    return r_deseq2.rlog(dds, **kwargs)


def deseq_results(
    dds: rpy2.robjects.methods.RS4, **kwargs: Any
) -> rpy2.robjects.methods.RS4:
    """Extract differential expression results from a DESeq analysis.

    This function extracts a result table from a DESeq2 analysis, providing base means
    across samples, log2 fold changes, standard errors, test statistics, p-values
    and adjusted p-values.

    Args:
        dds: A DESeqDataSet object, coming from the `run_dseq2` function.
        **kwargs: Additional arguments to pass to the results function.
            Common parameters include:
            - contrast: Vector of length 3 specifying the contrast to extract.
            - name: Name of the results to extract (alternative to contrast).
            - lfcThreshold: Log2 fold change threshold for null hypothesis.
            - altHypothesis: Alternative hypothesis to test.
            - pAdjustMethod: Method for multiple testing adjustment (default: "BH").
            - alpha: Significance level for independent filtering (default: 0.1).

    Returns:
        rpy2.robjects.methods.RS4: A DESeqResults object containing differential
        expression statistics.

    References:
        https://rdrr.io/bioc/DESeq2/man/results.html
    """
    return r_deseq2.results(dds, **kwargs)


def lfc_shrink(
    dds: rpy2.robjects.methods.RS4, **kwargs: Any
) -> rpy2.robjects.methods.RS4:
    """Apply log fold change shrinkage to DESeq2 results.

    This function adds shrunken log2 fold changes (LFC) and standard errors to a
    results table from DESeq run without LFC shrinkage. Shrinking helps reduce
    the noise in log fold change estimates for genes with low counts or high
    variability.

    Args:
        dds: A DESeqDataSet object that has been run through `run_dseq2`.
        **kwargs: Additional arguments to pass to the lfcShrink function.
            Common parameters include:
            - coef: Coefficient or contrast to apply shrinkage to.
            - contrast: Vector of length 3 specifying the contrast to shrink.
            - type: Shrinkage type ("normal", "apeglm", "ashr", "none").
            - lfcThreshold: Log2 fold change threshold for null hypothesis.
            - res: A DESeqResults object to shrink (if not specifying coef or contrast).

    Returns:
        rpy2.robjects.methods.RS4: A DESeqResults object with shrunken log fold changes.

    References:
        https://rdrr.io/bioc/DESeq2/man/lfcShrink.html
    """
    return r_deseq2.lfcShrink(dds, **kwargs)


def fpkm(dds: rpy2.robjects.methods.RS4, **kwargs: Any) -> Any:
    """Calculate FPKM (Fragments Per Kilobase of transcript per Million mapped reads).

    This function returns fragment counts normalized per kilobase of feature length
    per million mapped fragments. FPKM is commonly used for normalizing RNA-seq data
    to allow comparison of transcript abundances both within and between samples.

    Args:
        dds: A DESeqDataSet object.
        **kwargs: Additional arguments to pass to the fpkm function.
            Common parameters include:
            - robust: Whether to use the robust size factors (default: TRUE).
            - norm.factors: Optional normalization factors.

    Returns:
        Any: A matrix of FPKM values with the same dimensions as the original count
        matrix (genes as rows, samples as columns).

    References:
        https://rdrr.io/bioc/DESeq2/man/fpkm.html
    """
    return r_deseq2.fpkm(dds, **kwargs)


def fpm(dds: rpy2.robjects.methods.RS4, **kwargs: Any) -> Any:
    """Calculate FPM/CPM (Fragments/Counts Per Million mapped reads).

    This function calculates either a robust version (default) or the traditional
    matrix of fragments/counts per million mapped fragments (FPM/CPM).

    Args:
        dds: A DESeqDataSet object.
        **kwargs: Additional arguments to pass to the fpm function.
            Common parameters include:
            - robust: Whether to use the robust size factors (default: TRUE).
            - norm.factors: Optional normalization factors.

    Returns:
        Any: A matrix of FPM/CPM values with the same dimensions as the original count
        matrix (genes as rows, samples as columns).

    References:
        https://rdrr.io/bioc/DESeq2/man/fpm.html
    """
    return r_deseq2.fpm(dds, **kwargs)
