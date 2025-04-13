"""
Wrappers for R package ReactomePA

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

r_reactome_pa = importr("ReactomePA")


def enrich_reactome(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Pathway Enrichment Analysis of a gene set using Reactome database.

    This function performs over-representation analysis (ORA) on a set of genes
    using the Reactome pathway database. It returns the enriched pathways with
    false discovery rate (FDR) control.

    Args:
        gene_names: A vector of gene identifiers (usually Entrez gene IDs).
        **kwargs: Additional arguments to pass to the enrichPathway function.
            Common parameters include:
            - pvalueCutoff: Adjusted p-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).

    Returns:
        Any: An R enrichResult object that contains enriched pathways information.

    References:
        https://rdrr.io/bioc/ReactomePA/man/enrichPathway.html
    """
    return r_reactome_pa.enrichPathway(gene=gene_names, **kwargs)


def gsea_reactome(gene_list: ro.FloatVector, **kwargs: Any) -> Any:
    """Gene Set Enrichment Analysis using Reactome Pathway database.

    This function performs Gene Set Enrichment Analysis (GSEA) on a ranked list
    of genes using the Reactome pathway database.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes or other metrics that can be
            used to rank genes). The names should be Entrez gene IDs.
        **kwargs: Additional arguments to pass to the gsePathway function.
            Common parameters include:
            - pvalueCutoff: Adjusted p-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - exponent: Weight used in the enrichment score calculation (default: 1).

    Returns:
        Any: An R gseaResult object that contains GSEA results for Reactome pathways.

    References:
        https://rdrr.io/bioc/ReactomePA/man/gsePathway.html
    """
    return r_reactome_pa.gsePathway(geneList=gene_list, **kwargs)
