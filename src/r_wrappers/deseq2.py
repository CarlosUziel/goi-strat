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
from typing import Iterable

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
    **kwargs,
) -> rpy2.robjects.methods.RS4:
    """
    Creates a dataset for DESeq2 from a counts matrix.

    See: https://rdrr.io/bioc/DESeq2/man/DESeqDataSet.html

    Args:
        counts_matrix: A matrix representing counts of shape [n_features, n_samples]
        annot_df: Annotated data dataframe, whose index is the sample/replicate
            name.
        factors: Columns to be added to the dseq_dataset.
        design_factors: Define columns whose values are used for comparison.
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
    def sanitize_factor(factor: str):
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
    **kwargs,
) -> rpy2.robjects.methods.RS4:
    """
    Creates a dataset for DESeq2 from htseq files.

    See: https://rdrr.io/bioc/DESeq2/man/DESeqDataSet.html

    Format of sample table:
        A data.frame with two or more columns. Each row describes one
        sample. The first column is the sample name and the remaining columns are
        sample metadata which will be stored in colData.

    Args:
        annot_df: annotated data dataframe, whose index is the sample/replicate
            name.
        ID is used to uniquely identify samples.
        htseq_path: path where the htseq files can be located.
        factors: columns to be added to the dseq_dataset.
        design_factors: define columns whose values are used for comparison.
        htseq_files_pattern: pattern followed by the files storing htseq data.
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
    def sanitize_factor(factor: str):
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
    """
    Keep genes whose average expression across samples is bigger than `filter_count`.

    Args:
        dds: deseq2 dataset
        filter_count: count to filter the values of dds by.
    """
    f = ro.r(
        """
        f <- function(dds, filter_count) {
            return(dds[rowMeans(counts(dds)) > filter_count,])
        }
        """
    )
    return f(dds, filter_count)


def run_dseq2(dds: rpy2.robjects.methods.RS4, **kwargs) -> rpy2.robjects.methods.RS4:
    """
    This function performs a default analysis through the steps:

    1) Estimation of size factors: estimateSizeFactors
    2) Estimation of dispersion: estimateDispersions
    3) Negative Binomial GLM fitting and Wald statistics: nbinomWaldTest

    For complete details on each step, see the manual pages of the respective
    functions. After the DESeq function returns a DESeqDataSet object, results
    tables (log2 fold changes and p-values) can be generated using the results
    function. Shrunken LFC can then be generated using the lfcShrink function.

    (full docs in https://rdrr.io/bioc/DESeq2/man/DESeq.html)

    Args:
        dds: deseq2 dataset
    """
    return r_deseq2.DESeq(dds, **kwargs)


def norm_transform(data: rpy2.robjects.methods.RS4, **kwargs):
    """
    A simple function for creating a DESeqTransform object after applying:
    f(count(dds,normalized=TRUE) + pc).

    (full docs in https://rdrr.io/bioc/DESeq2/man/normTransform.html)

    Args:
        data: DESeqDataSet
    """
    return r_deseq2.normTransform(data, **kwargs)


def vst_transform(
    dds: rpy2.robjects.methods.RS4, **kwargs
) -> rpy2.robjects.methods.RS4:
    """
    [vst]
    This is a wrapper for the varianceStabilizingTransformation (VST) that
    provides much faster estimation of the dispersion trend used to
    determine the formula for the VST. The speed-up is accomplished by
    subsetting to a smaller number of genes in order to estimate this
    dispersion trend. The subset of genes is chosen deterministically,
    to span the range of genes' mean normalized count. This wrapper for the
    VST is not blind to the experimental design: the sample covariate
    information is used to estimate the global trend of genes' dispersion
    values over the genes' mean normalized count. It can be made strictly
    blind to experimental design by first assigning a design of ~1 before
    running this function, or by avoiding subsetting and using
    varianceStabilizingTransformation.

    [varianceStabilizingTransformation]
    This function calculates a variance stabilizing transformation (VST) from the
    fitted dispersion-mean relation(s) and then transforms the count data (normalized
    by division by the size factors or normalization factors), yielding a matrix of
    values which are now approximately homoskedastic (having constant variance along
    the range of mean values). The transformation also normalizes with respect to
    library size. The rlog is less sensitive to size factors, which can be an issue
    when size factors vary widely. These transformations are useful when checking for
    outliers or as input for machine learning techniques such as clustering or linear
    discriminant analysis.

    (full docs in https://rdrr.io/bioc/DESeq2/man/vst.html)

    Args:
        dds: a DESeqDataSet or a matrix of counts
    """
    try:
        return r_deseq2.vst(dds, **kwargs)
    except RRuntimeError as e:
        logging.warn(e)
        return r_deseq2.varianceStabilizingTransformation(dds, **kwargs)


def rlog_transform(
    dds: rpy2.robjects.methods.RS4, **kwargs
) -> rpy2.robjects.methods.RS4:
    """
    This function transforms the count data to the log2 scale in a way which
    minimizes differences between samples for rows with small counts,
    and which normalizes with respect to library size. The rlog
    transformation produces a similar variance stabilizing effect as
    varianceStabilizingTransformation, though rlog is more robust in the
    case when the size factors vary widely. The transformation is useful
    when checking for outliers or as input for machine learning techniques
    such as clustering or linear discriminant analysis. rlog takes as input
    a DESeqDataSet and returns a RangedSummarizedExperiment object.


    (full docs in https://rdrr.io/bioc/DESeq2/man/rlog.html)

    Args:
        dds: a DESeqDataSet or a matrix of counts
    """
    return r_deseq2.rlog(dds, **kwargs)


def deseq_results(dds: rpy2.robjects.methods.RS4, **kwargs):
    """
    Extracts a result table from a DESeq analysis giving base means across
    samples, log2 fold changes, standard errors, test statistics, p-values
    and adjusted p-values.

    (full docs in https://rdrr.io/bioc/DESeq2/man/results.html)

    Args:
        dds: a DESeqDataSet object, coming from "run_dseq2"
    """
    return r_deseq2.results(dds, **kwargs)


def lfc_shrink(dds: rpy2.robjects.methods.RS4, **kwargs):
    """
    Adds shrunken log2 fold changes (LFC) and SE to a results table from
    DESeq run without LFC shrinkage.

    (full docs in https://rdrr.io/bioc/DESeq2/man/lfcShrink.html)
    """
    return r_deseq2.lfcShrink(dds, **kwargs)


def fpkm(dds: rpy2.robjects.methods.RS4, **kwargs):
    """
    The following function returns fragment counts normalized per kilobase of feature
    length per million mapped fragments (by default using a robust estimate of the
    library size, as in estimateSizeFactors).

    See: https://rdrr.io/bioc/DESeq2/man/fpkm.html
    """
    return r_deseq2.fpkm(dds, **kwargs)


def fpm(dds: rpy2.robjects.methods.RS4, **kwargs):
    """
    Calculates either a robust version (default) or the traditional matrix of
    fragments/counts per million mapped fragments (FPM/CPM). Note: this
    function is written very simply and can be easily altered to produce other
    behavior by examining the source code.

    See: https://rdrr.io/bioc/DESeq2/man/fpm.html
    """
    return r_deseq2.fpm(dds, **kwargs)
