"""
Wrappers for R package sva

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Iterable

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_sva = importr("sva")


def combat_seq(counts_df: pd.DataFrame, batch: Iterable[int], **kwargs):
    """
    ComBat_seq is an improved model from ComBat using negative binomial regression,
        which specifically targets RNA-Seq count data.

    Args:
        counts_df: Counts dataframe (genes x samples).
        batch: Int vector indicating batches.

    Returns:
        Raw gene counts, adjusted for batch effects.
    """
    return rpy2_df_to_pd_df(
        r_sva.ComBat_seq(
            ro.r("as.matrix")(pd_df_to_rpy2_df(counts_df)),
            ro.IntVector(batch),
            full_mod=False,
            **kwargs,
        )
    ).astype(int)
