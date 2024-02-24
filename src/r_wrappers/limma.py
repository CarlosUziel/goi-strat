"""
    Wrappers for R package limma

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

from pathlib import Path
from typing import Any

import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import StrVector
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_limma = importr("limma")

pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def kegga(gene_names: StrVector, **kwargs):
    """
    KEGG over-representation analysis

    *ref docs: https://rdrr.io/bioc/limma/man/goana.html
    """
    return r_limma.kegga(gene_names, **kwargs)


def top_kegg(
    kegg_result: rpy2.robjects.DataFrame,
    filter_by: str = "P.DE",
    filter_thr: float = 0.05,
):
    """
        Get top kegg pathways from a kegg_result.
    Args:
        kegg_result: Kegg result
        filter_by: Name of column by which to filter
        filter_thr: Value used for filtering. Rows must have a value lower
        than this.
    """
    # 0. Convert R dataframe to Pandas
    df = rpy2_df_to_pd_df(kegg_result)

    # 1. Filter dataframe and sort
    df = df[df[filter_by] < filter_thr]
    df = df.sort_values(by=filter_by)

    # 2. Reconvert to R dataframe and return
    return pd_df_to_rpy2_df(df)


def linear_model_fit(obj: Any, design: Any, **kwargs):
    """
    Fit linear model for each gene given a series of arrays

    *ref docs: https://rdrr.io/bioc/limma/man/lmFit.html

    Args:
        obj: A matrix-like ann_df object containing log-ratios or
            log-expression values for a series of arrays, with rows
            corresponding to genes and columns to samples. Any type of
            ann_df object that can be processed by getEAWP
            is acceptable.
        design: the design matrix of the microarray experiment, with rows
            corresponding to arrays and columns to coefficients to be
            estimated. Defaults to the unit vector meaning that the arrays are
            treated as replicates.
    """
    return r_limma.lmFit(obj, design, **kwargs)


def make_contrasts(contrasts: StrVector, levels: Any):
    """
    Construct the contrast matrix corresponding to specified contrasts of a
    set of parameters.

    *ref docs: https://rdrr.io/bioc/limma/man/makeContrasts.html

    Args:
        contrasts: character vector specifying contrasts
        levels: character vector or factor giving the names of the
        parameters of which contrasts are desired, or a
            design matrix or other object with the parameter names as column
            names.
    """
    return r_limma.makeContrasts(contrasts=contrasts, levels=levels)


def fit_contrasts(fit: Any, contrasts: Any):
    """
     Given a linear model fit to microarray ann_df, compute estimated
     coefficients and standard errors for a given set
        of contrasts.

    *ref docs: https://rdrr.io/bioc/limma/man/contrasts.fit.html

    Args:
        fit: an MArrayLM object or a list object produced by the function
        lm.series or equivalent. Must contain
            components coefficients and stdev.unscaled.
        contrasts: numeric matrix with rows corresponding to coefficients in
        fit and columns containing contrasts.
            May be a vector if there is only one contrast.
    """
    return r_limma.contrasts_fit(fit=fit, contrasts=contrasts)


def empirical_bayes(fit: Any, **kwargs):
    """
    Empirical Bayes Statistics for Differential Expression

    Given a microarray linear model fit, compute moderated t-statistics,
    moderated F-statistic, and log-odds of
        differential expression by empirical Bayes moderation of the
        standard errors towards a common value.

    *ref docs: https://rdrr.io/bioc/limma/man/ebayes.html

    Args:
        fit: an MArrayLM object or a list object produced by the function
        lm.series or equivalent. Must contain
            components coefficients and stdev.unscaled.
    """
    return r_limma.eBayes(fit=fit, **kwargs)


def decide_tests(obj: Any, **kwargs):
    """
    Identify which genes are significantly differentially expressed for each
    contrast from a fit object containing
        p-values and test statistics. A number of different multiple testing
        strategies are offered that adjust for
        multiple testing down the genes as well as across contrasts for each
        gene.

    *ref docs: https://rdrr.io/bioc/limma/man/decideTests.html

    Returns:
        obj: a numeric matrix of p-values or an MArrayLM object from which
        p-values and t-statistics can be extracted.
    """
    return r_limma.decideTests(obj, **kwargs)


def venn_diagram(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Compute classification counts and draw a Venn diagram.

    *ref docs: https://rdrr.io/bioc/limma/man/venn.html

    Args:
        obj: object to plot
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_limma.vennDiagram(obj, **kwargs)
    dev_off()


def volcano_plot(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Creates a volcano plot for a specified coefficient of a linear model.

    *ref docs: https://rdrr.io/bioc/limma/man/volcanoplot.html

    Args:
        obj: object to plot
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_limma.volcanoplot(obj, **kwargs)
    dev_off()


def plot_md(obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs):
    """
     Creates a mean-difference plot (aka MA plot) with color coding for
     highlighted points.

    *ref docs: https://rdrr.io/bioc/limma/man/plotMD.html

    Args:
        obj: object to plot
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_limma.plotMD(obj, **kwargs)
    dev_off()


def top_table(fit: Any, **kwargs):
    """
    Table of Top Genes from Linear Model Fit. Extract a table of the
    top-ranked genes from a linear model fit.

    *ref docs: https://rdrr.io/bioc/limma/man/toptable.html

    Args:
        fit: list containing a linear model fit produced by lmFit,
        lm.series, gls.series or mrlm. For topTable,
            fit should be an object of class MArrayLM as produced by lmFit
            and eBayes.
    """
    return r_limma.topTable(fit, **kwargs)


def remove_batch_effect(x: Any, **kwargs):
    """
    Remove batch effects from expression data.

    *ref docs: https://rdrr.io/bioc/limma/man/removeBatchEffect.html

    Args:
        x: numeric matrix, or any data object that can be processed by getEAWP
            containing log-expression values for a series of samples. Rows correspond
            to probes and columns to samples.

    Returns:
        A numeric matrix of log-expression values with batch and covariate effects
            removed.

    """
    return r_limma.removeBatchEffect(x, **kwargs)
