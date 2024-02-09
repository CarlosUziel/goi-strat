from typing import Dict, Iterable, Union

import pandas as pd
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_msigdbr = importr("msigdbr")


def get_msigdbr(species: str = "Homo sapiens", **kwargs) -> pd.DataFrame:
    """
    Retrieve the msigdbr data frame.

    *ref docs: https://rdrr.io/github/dekanglv/RpacEx/man/msigdbr.html
    """
    return rpy2_df_to_pd_df(r_msigdbr.msigdbr(species=species, **kwargs))


def get_t2g(msigdb_df: pd.DataFrame, gene_id_col: str = "gene_symbol"):
    """
    Get term to gene from Molecular Signatures Database.

    Args:
        msigdb_df: msigdb dataframe, result of calling get_msigdbr().
        gene_id_col: ID column to retrieve. Can be "entrez_gene" to retrieve
            ENTREZID gene ids or "gene_symbol" to retrieve SYMBOL gene ids.
    """
    return pd_df_to_rpy2_df(msigdb_df[["gs_name", gene_id_col]])


def get_msigb_gene_sets(
    species: str = "Homo sapiens", category: str = "H", gene_id_col: str = "gene_symbol"
) -> Dict[str, Iterable[Union[str, int]]]:
    return (
        rpy2_df_to_pd_df(get_t2g(get_msigdbr(species, category=category), gene_id_col))
        .groupby("gs_name")
        .agg(list)[gene_id_col]
        .to_dict()
    )
