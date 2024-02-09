"""
    Wrappers for R package CMSCaller

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""
import pandas as pd
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_cms_caller = importr("CMScaller")


def cms_caller(data: pd.DataFrame, **kwargs):
    """
    Cancer-cell intrinsic CMS classification based on pre-defined subtype templates.

    *docs: https://rdrr.io/github/peterawe/CMScaller/man/CMScaller.html
    """
    return r_cms_caller.CMScaller(pd_df_to_rpy2_df(data), **kwargs)
