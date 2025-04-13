from typing import Any, Dict, Iterable, Union

import pandas as pd
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df, rpy2_df_to_pd_df

r_msigdbr = importr("msigdbr")


def get_msigdbr(species: str = "Homo sapiens", **kwargs: Any) -> pd.DataFrame:
    """Retrieve the Molecular Signatures Database (MSigDB) as a pandas DataFrame.

    This function provides access to the msigdbr database, with gene sets
    organized by collections (hallmark, positional, curated, motif,
    computational, oncogenic, immunologic).

    Args:
        species: The species name for which to retrieve the database. Default is "Homo sapiens".
        **kwargs: Additional arguments to pass to the msigdbr function.
            Common parameters include:
            - category: MSigDB collection to filter. Options include "H" (hallmark),
              "C1" through "C8" (positional, curated, motif, etc.)
            - subcategory: Subcategory within a collection to filter on.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the MSigDB gene sets.

    References:
        https://rdrr.io/github/dekanglv/RpacEx/man/msigdbr.html
    """
    return rpy2_df_to_pd_df(r_msigdbr.msigdbr(species=species, **kwargs))


def get_t2g(msigdb_df: pd.DataFrame, gene_id_col: str = "gene_symbol") -> pd.DataFrame:
    """Get term-to-gene mapping from Molecular Signatures Database.

    This function extracts a dataframe with two columns: gene set names and gene identifiers,
    for use in enrichment analyses.

    Args:
        msigdb_df: MSigDB dataframe, result of calling get_msigdbr().
        gene_id_col: ID column to retrieve. Can be "entrez_gene" to retrieve
            ENTREZID gene ids or "gene_symbol" to retrieve SYMBOL gene ids.

    Returns:
        pd.DataFrame: A pandas DataFrame with two columns: "gs_name" (gene set name)
        and the specified gene_id_col.
    """
    return pd_df_to_rpy2_df(msigdb_df[["gs_name", gene_id_col]])


def get_msigb_gene_sets(
    species: str = "Homo sapiens", category: str = "H", gene_id_col: str = "gene_symbol"
) -> Dict[str, Iterable[Union[str, int]]]:
    """Create a dictionary of gene sets from MSigDB.

    This function retrieves gene sets from the Molecular Signatures Database
    and organizes them as a dictionary mapping gene set names to lists of genes.

    Args:
        species: The species name. Default is "Homo sapiens".
        category: MSigDB collection to retrieve. Default is "H" (hallmark gene sets).
            Other options include "C1" through "C8" for other collections.
        gene_id_col: Type of gene identifiers to use. Options include
            "gene_symbol" for gene symbols or "entrez_gene" for Entrez IDs.

    Returns:
        Dict[str, Iterable[Union[str, int]]]: A dictionary mapping gene set names
        to lists of genes (either gene symbols as strings or Entrez IDs as integers).
    """
    return (
        rpy2_df_to_pd_df(get_t2g(get_msigdbr(species, category=category), gene_id_col))
        .groupby("gs_name")
        .agg(list)[gene_id_col]
        .to_dict()
    )
