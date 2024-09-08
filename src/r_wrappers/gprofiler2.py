"""
    Wrappers for R package gprofiler2

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

from pathlib import Path
from typing import Any

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

from r_wrappers.utils import rpy2_df_to_pd_df

r_gprofiler2 = importr("gprofiler2")
r_ggplot = importr("ggplot2")


def gost(gene_names: ro.StrVector, **kwargs):
    """
    Interface to the g:Profiler tool g:GOSt (
    https://biit.cs.ut.ee/gprofiler/gost) for functional enrichments
    analysis of gene lists. In case the input 'query' is a list of gene
    vectors, results for multiple queries will be returned in the same data
    frame with column 'query' indicating the corresponding query name. If
    'multi_query' is selected, the result is a data frame for comparing
    multiple input lists, just as in the web tool.

    See: https://rdrr.io/cran/gprofiler2/man/gost.html
    """
    return r_gprofiler2.gost(query=gene_names, **kwargs)


def save_gost_res(
    gost_res: Any, save_path: Path, sort_by: str = "p_value"
) -> pd.DataFrame:
    """
        Saves gost results ann_df.frame as .RDS and .csv

        NOTE: gost_res (which is a named list), also contains metadata of
        the query, but this is not saved at the moment.

    Args:
        gost_res: Result of calling the gost function.
        save_path: path to store the results ann_df frame
        sort_by: Column to sort returned dataframe by, must be convertible
            to numeric (float).

    Returns:
        Saved dataframe as a pandas DataFrame, sorted by p-value.
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
    gost_res: Any, save_path: Path = None, width: int = 15, height: int = 10, **kwargs
):
    """
    This function creates a Manhattan plot out of the results from
    gprofiler2::gost(). The plot is very similar to the one shown in the
    g:GOSt web tool.

    See: https://rdrr.io/cran/gprofiler2/man/gostplot.html
    """
    plot = r_gprofiler2.gostplot(gost_res, **kwargs)

    if not kwargs.get("interactive", True) and save_path is not None:
        r_ggplot.ggsave(str(save_path), plot, width=width, height=height)

    return plot


def publish_gost_table(
    gost_res: Any, save_path: Path, width: int = 15, height: int = 10, **kwargs
):
    """
    This function creates a table mainly for the results from gost()
    function. However, if the input at least contains columns named
    'term_id' and 'p_value' then any enrichment results data frame can be
    visualised in a table with this function.

    See: https://rdrr.io/cran/gprofiler2/man/publish_gosttable.html
    """
    plot = r_gprofiler2.publish_gosttable(gost_res, **kwargs)
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)


def publish_gost_plot(
    g_plot: Any, save_path: Path, width: int = 15, height: int = 10, **kwargs
):
    """
    This function allows to highlight a list of selected terms on the
    Manhattan plot created with the gprofiler2::gostplot() function. The
    resulting plot is saved to a  publication ready image if 'filename'
    is specified. The plot is very similar to the one shown in the
    g:GOSt web tool after clicking on circles.

    See: https://rdrr.io/cran/gprofiler2/man/publish_gostplot.html
    """
    plot = r_gprofiler2.publish_gostplot(g_plot, **kwargs)
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)
