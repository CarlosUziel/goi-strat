"""
Wrappers for R package gprofiler2

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from r_wrappers.utils import rpy2_df_to_pd_df

r_gprofiler2 = importr("gprofiler2")
r_ggplot = importr("ggplot2")


def gost(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Perform functional enrichment analysis using g:Profiler.

    This function serves as an interface to the g:Profiler tool g:GOSt
    (https://biit.cs.ut.ee/gprofiler/gost) for functional enrichment
    analysis of gene lists. It can analyze multiple gene lists simultaneously
    and compare their enrichment results.

    Args:
        gene_names: A vector of gene identifiers to be analyzed.
        **kwargs: Additional arguments to pass to the gost function.
            Common parameters include:
            - organism: Organism name (default: "hsapiens").
            - ordered_query: Whether the gene list is ordered by significance (default: FALSE).
            - multi_query: Whether to perform multi-query analysis (default: FALSE).
            - significant: Whether to return only significant results (default: TRUE).
            - exclude_iea: Whether to exclude electronic annotations (default: FALSE).
            - measure_underrepresentation: Whether to measure underrepresentation (default: FALSE).
            - evcodes: Whether to show evidence codes (default: FALSE).
            - user_threshold: Significance threshold (default: 0.05).
            - correction_method: Method for multiple testing correction (default: "g_SCS").
            - domain_scope: Statistical domain scope ("annotated", "known", "custom").
            - custom_bg: Custom background gene list.
            - sources: Data sources to use (GO, KEGG, etc.).

    Returns:
        Any: A named list containing enrichment results and metadata:
            - result: Data frame with enrichment analysis results.
            - meta: Query metadata.
            - query: Original query information.

    References:
        https://rdrr.io/cran/gprofiler2/man/gost.html
    """
    with localconverter(ro.default_converter):
        return r_gprofiler2.gost(query=gene_names, **kwargs)


def save_gost_res(
    gost_res: Any, save_path: Path, sort_by: str = "p_value"
) -> pd.DataFrame:
    """Save g:Profiler enrichment results to a file and return as DataFrame.

    This function extracts the results dataframe from a gost result object,
    saves it to a CSV file, and returns the dataframe sorted by a specified column.

    Args:
        gost_res: The result object returned by the gost function.
        save_path: Path where the CSV file will be saved.
        sort_by: Column name to sort the returned dataframe by. This column
            must contain values that can be converted to floats. Default is "p_value".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the enrichment results,
        sorted by the specified column.

    Notes:
        The gost_res object (which is a named list) also contains metadata
        about the query, but this metadata is not saved to the CSV file.
    """
    f = ro.r(
        """
        f <- function(df){
            res = as.data.frame(apply(df,2,as.character))
    
            if(nrow(df) > 1){
                return(res)
            } else {
                return(t(res))
            }
        }
    """
    )

    # 0. Save result dataframe
    df = rpy2_df_to_pd_df(f(gost_res.rx2("result")))
    df.to_csv(save_path)

    # 1. Sort dataframe by p_value and return
    df[sort_by] = df[sort_by].astype(
        float
    )  # somehow it was converted to string, not to float
    return df.sort_values(sort_by)


def gost_plot(
    gost_res: Any,
    save_path: Optional[Path] = None,
    width: int = 15,
    height: int = 10,
    **kwargs: Any,
) -> Any:
    """Create a Manhattan plot from g:Profiler enrichment results.

    This function generates a Manhattan plot visualization of the enrichment
    results from the g:Profiler gost function. The plot is similar to the one
    shown in the g:GOSt web tool.

    Args:
        gost_res: The result object returned by the gost function.
        save_path: Path where the plot will be saved if not interactive.
            If None, the plot will not be saved. Default is None.
        width: Width of the saved figure in inches. Default is 15.
        height: Height of the saved figure in inches. Default is 10.
        **kwargs: Additional arguments to pass to the gostplot function.
            Common parameters include:
            - interactive: Whether to create an interactive plot (default: TRUE).
            - capped: Whether to cap extremely small p-values (default: TRUE).
            - pal: Color palette for different data sources.

    Returns:
        Any: A ggplot2 object (if interactive=FALSE) or a plotly object
        (if interactive=TRUE) containing the Manhattan plot.

    References:
        https://rdrr.io/cran/gprofiler2/man/gostplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_gprofiler2.gostplot(gost_res, **kwargs)

    if not kwargs.get("interactive", True) and save_path is not None:
        r_ggplot.ggsave(str(save_path), plot, width=width, height=height)

    return plot


def publish_gost_table(
    gost_res: Any, save_path: Path, width: int = 15, height: int = 10, **kwargs: Any
) -> None:
    """Create a publication-ready table from g:Profiler enrichment results.

    This function creates a formatted table from g:Profiler enrichment results
    that is suitable for publication. The table is saved as an image file.

    Args:
        gost_res: The result object returned by the gost function, or any dataframe
            that contains at least 'term_id' and 'p_value' columns.
        save_path: Path where the table image will be saved.
        width: Width of the saved figure in inches. Default is 15.
        height: Height of the saved figure in inches. Default is 10.
        **kwargs: Additional arguments to pass to the publish_gosttable function.
            Common parameters include:
            - highlight_terms: Vector of term IDs to highlight.
            - use_colors: Whether to use colors in the table.
            - show_columns: Which columns to show in the table.
            - filename: Deprecated, use save_path instead.
            - ggplot: Whether to return a ggplot object.

    References:
        https://rdrr.io/cran/gprofiler2/man/publish_gosttable.html
    """
    plot = r_gprofiler2.publish_gosttable(gost_res, **kwargs)
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)


def publish_gost_plot(
    g_plot: Any, save_path: Path, width: int = 15, height: int = 10, **kwargs: Any
) -> None:
    """Create a publication-ready Manhattan plot with highlighted terms.

    This function allows highlighting a list of selected terms on the
    Manhattan plot created with the gostplot function. The resulting plot
    is saved as a publication-ready image.

    Args:
        g_plot: The plot object returned by the gost_plot function.
        save_path: Path where the plot image will be saved.
        width: Width of the saved figure in inches. Default is 15.
        height: Height of the saved figure in inches. Default is 10.
        **kwargs: Additional arguments to pass to the publish_gostplot function.
            Common parameters include:
            - highlight_terms: Vector of term IDs to highlight.
            - highlight_color: Color to use for highlighting.
            - label_size: Size of term labels.
            - term_size: Size of term points.
            - filename: Deprecated, use save_path instead.

    References:
        https://rdrr.io/cran/gprofiler2/man/publish_gostplot.html
    """
    plot = r_gprofiler2.publish_gostplot(g_plot, **kwargs)
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)
