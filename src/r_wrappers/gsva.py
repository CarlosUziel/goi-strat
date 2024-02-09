from typing import Dict, Iterable, Union

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from data.utils import supress_stdout
from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_gsva = importr("GSVA")


@supress_stdout
def gsva(
    expr_df: pd.DataFrame, gene_sets: Dict[str, Iterable[Union[str, int]]], **kwargs
) -> pd.DataFrame:
    """
    Estimates GSVA enrichment scores.

    *ref docs: https://rdrr.io/bioc/GSVA/man/gsva.html
    """
    data_matrix = ro.r("data.matrix")(pd_df_to_rpy2_df(expr_df))
    data_matrix.rownames = ro.StrVector(expr_df.index)
    return rpy2_df_to_pd_df(
        r_gsva.gsva(data_matrix, ro.ListVector(gene_sets), **kwargs)
    )
