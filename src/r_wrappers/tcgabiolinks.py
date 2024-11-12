"""
Wrappers for R package TCGAbiolinks

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.
Eg:
    R --> annot_df.category
    Python --> data_category
"""

from typing import Any, Iterable

import pandas as pd
import rpy2
from rpy2.robjects.packages import importr

from r_wrappers.utils import rpy2_df_to_pd_df_manual

r_source = importr("TCGAbiolinks")


def gdc_query(
    project: Iterable[str], data_category: str, **kwargs
) -> rpy2.robjects.vectors.DataFrame:
    """
    Uses GDC API to search for search, it searches for both controlled and
    open-access data.

    See: https://rdrr.io/bioc/TCGAbiolinks/man/GDCquery.html

    Returns:
        rpy2 dataframe, can be passed directly to gdc_download

    """
    return r_source.GDCquery(project=project, data_category=data_category, **kwargs)


def get_query_results(query: rpy2.robjects.vectors.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Get the result's table from query, it can select columns with cols
    argument and return a number of rows using rows argument.

    See: https://rdrr.io/bioc/TCGAbiolinks/man/getResults.html

    Args:
        query: TCGAbiolinks query object
    """
    return rpy2_df_to_pd_df_manual(r_source.getResults(query, **kwargs))


def gdc_download(query: Any, **kwargs) -> bool:
    """
    Uses GDC API or GDC transfer tool to download gdc data The user can
    use query argument The annot_df from query will be
        save in a folder: project/annot_df.category

    See: https://rdrr.io/bioc/TCGAbiolinks/man/GDCdownload.html

    Args:
        query: A query for GDCquery function
    """
    r_source.GDCdownload(query=query, **kwargs)


def gdc_prepare(query: Any, **kwargs) -> rpy2.robjects.methods.RS4:
    """
    Reads the data downloaded and prepare it into an R object

    See: https://rdrr.io/bioc/TCGAbiolinks/man/GDCprepare.html

    Args:
        query: A query for GDCquery function
    """
    return r_source.GDCprepare(query=query, **kwargs)
