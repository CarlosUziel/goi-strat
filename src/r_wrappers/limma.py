"""
Wrappers for R package limma

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
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


def kegga(gene_names: StrVector, **kwargs: Any) -> rpy2.robjects.DataFrame:
    """Perform KEGG pathway over-representation analysis.

    This function tests for over-representation of KEGG pathways in a set of genes.

    Args:
        gene_names: A vector of gene identifiers (usually Entrez gene IDs).
        **kwargs: Additional arguments to pass to the kegga function.
            Common parameters include:
            - species: Species name (e.g., "Hs" for human).
            - universe: Background genes to use for enrichment analysis.
            - de: Vector of differentially expressed genes from gene_names.
            - trend: Whether to adjust for gene length or abundance bias.
            - sort: How to sort the results.

    Returns:
        rpy2.robjects.DataFrame: A DataFrame where rows represent KEGG pathways and
        columns contain statistics on the pathway enrichment.

    References:
        https://rdrr.io/bioc/limma/man/goana.html
    """
    return r_limma.kegga(gene_names, **kwargs)


def top_kegg(
    kegg_result: rpy2.robjects.DataFrame,
    filter_by: str = "P.DE",
    filter_thr: float = 0.05,
) -> rpy2.robjects.DataFrame:
    """Filter and sort KEGG pathway enrichment results.

    This function filters KEGG pathway enrichment results based on a specified
    significance threshold and returns them sorted by significance.

    Args:
        kegg_result: Result from the kegga function.
        filter_by: Name of the column by which to filter (default: "P.DE").
        filter_thr: P-value threshold for filtering. Rows must have a value
            lower than this threshold (default: 0.05).

    Returns:
        rpy2.robjects.DataFrame: A filtered and sorted DataFrame containing
        significant KEGG pathway enrichment results.
    """
    # 0. Convert R dataframe to Pandas
    df = rpy2_df_to_pd_df(kegg_result)

    # 1. Filter dataframe and sort
    df = df[df[filter_by] < filter_thr]
    df = df.sort_values(by=filter_by)

    # 2. Reconvert to R dataframe and return
    return pd_df_to_rpy2_df(df)


def linear_model_fit(obj: Any, design: Any, **kwargs: Any) -> Any:
    """Fit a linear model for each gene given a series of arrays.

    This function fits a linear model for each gene (row) in a matrix of expression data,
    allowing for gene-wise variance estimation.

    Args:
        obj: A matrix-like data object containing log-ratios or log-expression
            values for a series of arrays, with rows corresponding to genes and
            columns to samples. Any type of data object that can be processed
            by getEAWP is acceptable.
        design: The design matrix of the experiment, with rows corresponding to arrays
            and columns to coefficients to be estimated. Defaults to the unit vector
            meaning that the arrays are treated as replicates.
        **kwargs: Additional arguments to pass to the lmFit function.
            Common parameters include:
            - weights: Optional numeric matrix of weights to use in the fitting process.
            - method: Method for computing standard errors ("ls" or "robust").

    Returns:
        Any: An MArrayLM object containing the fitted linear model, with components:
        - coefficients: Matrix of estimated coefficients
        - stdev.unscaled: Matrix of unscaled standard errors
        - sigma: Vector of residual standard deviations
        - df.residual: Vector of residual degrees of freedom

    References:
        https://rdrr.io/bioc/limma/man/lmFit.html
    """
    return r_limma.lmFit(obj, design, **kwargs)


def make_contrasts(contrasts: StrVector, levels: Any) -> Any:
    """Construct a contrast matrix for specified comparisons.

    This function constructs a contrast matrix corresponding to specified contrasts
    of a set of parameters, which can then be used for differential expression analysis.

    Args:
        contrasts: Character vector specifying contrasts, e.g., "GroupA-GroupB".
        levels: Character vector or factor giving the names of the parameters
            of which contrasts are desired, or a design matrix or other object
            with the parameter names as column names.

    Returns:
        Any: A numeric matrix with rows corresponding to parameters (levels) and
        columns corresponding to contrasts. Each column defines a contrast as
        a linear combination of the parameters.

    References:
        https://rdrr.io/bioc/limma/man/makeContrasts.html
    """
    return r_limma.makeContrasts(contrasts=contrasts, levels=levels)


def fit_contrasts(fit: Any, contrasts: Any) -> Any:
    """Compute estimated coefficients and standard errors for a given set of contrasts.

    Given a linear model fit to microarray data, this function computes estimated
    coefficients and standard errors for a given set of contrasts.

    Args:
        fit: An MArrayLM object produced by the function linear_model_fit or equivalent.
            Must contain components coefficients and stdev.unscaled.
        contrasts: Numeric matrix with rows corresponding to coefficients in fit
            and columns containing contrasts. May be a vector if there is only one contrast.
            Typically produced by make_contrasts.

    Returns:
        Any: An MArrayLM object containing the estimated coefficients and standard
        errors for the contrasts.

    References:
        https://rdrr.io/bioc/limma/man/contrasts.fit.html
    """
    return r_limma.contrasts_fit(fit=fit, contrasts=contrasts)


def empirical_bayes(fit: Any, **kwargs: Any) -> Any:
    """Apply empirical Bayes moderation of standard errors for differential expression.

    This function computes moderated t-statistics, moderated F-statistics, and log-odds
    of differential expression by empirical Bayes moderation of the standard errors
    towards a common value. This increases statistical power, especially for
    experiments with small numbers of replicates.

    Args:
        fit: An MArrayLM object produced by linear_model_fit or fit_contrasts.
            Must contain components coefficients and stdev.unscaled.
        **kwargs: Additional arguments to pass to the eBayes function.
            Common parameters include:
            - trend: Whether to fit a mean-variance trend.
            - robust: Whether to use robust estimation of the prior variance.
            - proportion: Prior proportion of differentially expressed genes.

    Returns:
        Any: An MArrayLM object with added components:
        - t: Moderated t-statistics
        - p.value: P-values corresponding to the t-statistics
        - lods: Log-odds of differential expression
        - F: Moderated F-statistics (if coef is NULL)
        - F.p.value: P-values corresponding to the F-statistics

    References:
        https://rdrr.io/bioc/limma/man/ebayes.html
    """
    return r_limma.eBayes(fit=fit, **kwargs)


def decide_tests(obj: Any, **kwargs: Any) -> Any:
    """Identify significantly differentially expressed genes for each contrast.

    This function identifies which genes are significantly differentially expressed
    for each contrast from a fit object containing p-values and test statistics.
    It adjusts for multiple testing across both genes and contrasts.

    Args:
        obj: A numeric matrix of p-values or an MArrayLM object from which p-values
            and t-statistics can be extracted.
        **kwargs: Additional arguments to pass to the decideTests function.
            Common parameters include:
            - method: Multiple testing method ("separate", "global", "hierarchical", "nestedF").
            - adjust.method: P-value adjustment method ("BH", "fdr", "none", etc.).
            - p.value: P-value cutoff (default: 0.05).
            - lfc: Log fold-change threshold.

    Returns:
        Any: A numeric matrix containing -1, 0, or 1 for each gene and contrast,
        indicating significantly down-regulated, not significant, or significantly
        up-regulated, respectively.

    References:
        https://rdrr.io/bioc/limma/man/decideTests.html
    """
    return r_limma.decideTests(obj, **kwargs)


def venn_diagram(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create a Venn diagram showing overlaps between significant gene sets.

    This function computes classification counts and draws a Venn diagram showing
    the overlaps between genes that are significant in different contrasts.

    Args:
        obj: Object to plot, typically the result from decide_tests.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the vennDiagram function.
            Common parameters include:
            - include: Which columns of the object to include.
            - names: Names for the circles in the Venn diagram.
            - circle.col: Vector of colors for the circles.
            - counts.col: Color for the counts.

    References:
        https://rdrr.io/bioc/limma/man/venn.html
    """
    pdf(str(save_path), width=width, height=height)
    r_limma.vennDiagram(obj, **kwargs)
    dev_off()


def volcano_plot(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create a volcano plot for a specified coefficient of a linear model.

    This function creates a volcano plot showing the relationship between log-fold
    changes and statistical significance for all genes in a differential expression analysis.

    Args:
        obj: Object to plot, typically an MArrayLM object from empirical_bayes.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the volcanoplot function.
            Common parameters include:
            - coef: Which coefficient/contrast to plot.
            - style: Style of the plot ("p-value" or "B-statistic").
            - highlight: Vector of gene IDs to highlight.
            - names: Vector of gene names to display on the plot.

    References:
        https://rdrr.io/bioc/limma/man/volcanoplot.html
    """
    pdf(str(save_path), width=width, height=height)
    r_limma.volcanoplot(obj, **kwargs)
    dev_off()


def plot_md(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create a mean-difference plot (MA plot) with color coding for highlighted points.

    This function creates an MA plot showing the log-fold changes against the mean
    log-expression values for all genes in a differential expression analysis.

    Args:
        obj: Object to plot, typically an MArrayLM object from empirical_bayes.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the plotMD function.
            Common parameters include:
            - column: Which column of coefficients to plot.
            - status: Vector indicating which points to highlight.
            - values: Whether to plot log-fold-changes or coefficients.
            - hl.col: Color for highlighted points.

    References:
        https://rdrr.io/bioc/limma/man/plotMD.html
    """
    pdf(str(save_path), width=width, height=height)
    r_limma.plotMD(obj, **kwargs)
    dev_off()


def top_table(fit: Any, **kwargs: Any) -> rpy2.robjects.DataFrame:
    """Extract a table of the top-ranked genes from a linear model fit.

    This function extracts a table of the top-ranked genes from a linear model fit,
    sorted by adjusted p-value or other criteria.

    Args:
        fit: An MArrayLM object as produced by linear_model_fit and empirical_bayes.
        **kwargs: Additional arguments to pass to the topTable function.
            Common parameters include:
            - coef: Which coefficient/contrast to extract results for.
            - sort.by: How to sort the results ("p", "B", "logFC", etc.).
            - number: Maximum number of genes to return.
            - adjust.method: Method for adjusting p-values.
            - p.value: P-value cutoff for filtering.
            - lfc: Log fold-change threshold for filtering.

    Returns:
        rpy2.robjects.DataFrame: A DataFrame containing the top-ranked genes
        with statistics such as log-fold changes, p-values, and adjusted p-values.

    References:
        https://rdrr.io/bioc/limma/man/toptable.html
    """
    return r_limma.topTable(fit, **kwargs)


def remove_batch_effect(x: Any, **kwargs: Any) -> Any:
    """Remove batch effects from expression data.

    This function removes batch effects and other unwanted variation from expression data,
    while preserving biological variation of interest.

    Args:
        x: Numeric matrix, or any data object that can be processed by getEAWP,
            containing log-expression values for a series of samples. Rows correspond
            to genes and columns to samples.
        **kwargs: Additional arguments to pass to the removeBatchEffect function.
            Common parameters include:
            - batch: Factor or numeric vector specifying batches.
            - batch2: Optional second batch factor.
            - covariates: Matrix of additional covariates to adjust for.
            - design: Design matrix for outcome of interest (preserved variation).

    Returns:
        Any: A numeric matrix of log-expression values with batch and covariate effects
        removed, having the same dimensions as the input matrix.

    References:
        https://rdrr.io/bioc/limma/man/removeBatchEffect.html
    """
    return r_limma.removeBatchEffect(x, **kwargs)
