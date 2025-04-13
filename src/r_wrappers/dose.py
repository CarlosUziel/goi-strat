"""
Wrappers for R package DOSE

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

from components.functional_analysis.orgdb import OrgDB

r_dose = importr("DOSE")


def set_readable(enrich_result: Any, org_db: OrgDB, **kwargs: Any) -> Any:
    """Map gene IDs to gene symbols in enrichment results.

    This function translates gene IDs (e.g., Entrez IDs) to more readable
    gene symbols in the enrichment results.

    Args:
        enrich_result: An enrichment result object from DOSE or other enrichment
            analysis packages.
        org_db: An organism database object containing mapping information
            between gene IDs and symbols.
        **kwargs: Additional arguments to pass to the setReadable function.
            Common parameters include:
            - keyType: The type of gene ID used in the enrichment result.

    Returns:
        Any: The enrichment result object with gene IDs translated to gene symbols.

    References:
        https://rdrr.io/bioc/DOSE/man/setReadable.html
    """
    return r_dose.setReadable(enrich_result, org_db.db, **kwargs)


def enrich_do(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Perform Disease Ontology (DO) enrichment analysis.

    This function performs over-representation analysis to identify enriched
    disease terms from the Disease Ontology for a given set of genes.

    Args:
        gene_names: A vector of gene identifiers (usually Entrez gene IDs).
        **kwargs: Additional arguments to pass to the enrichDO function.
            Common parameters include:
            - ont: Ontology to use, e.g., "DO", "DOLite".
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).
            - readable: Whether to map gene IDs to gene symbols (default: FALSE).

    Returns:
        Any: An enrichResult object containing enriched disease terms.

    References:
        https://rdrr.io/bioc/DOSE/man/enrichDO.html
    """
    return r_dose.enrichDO(gene=gene_names, **kwargs)


def gse_do(gene_list: ro.FloatVector, **kwargs: Any) -> Any:
    """Perform Gene Set Enrichment Analysis with Disease Ontology.

    This function performs Gene Set Enrichment Analysis (GSEA) to identify
    enriched disease terms from the Disease Ontology for a ranked list of genes.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes or other metrics that can be
            used to rank genes). The names should be Entrez gene IDs.
        **kwargs: Additional arguments to pass to the gseDO function.
            Common parameters include:
            - ont: Ontology to use, e.g., "DO", "DOLite".
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - seed: Random seed for reproducibility.

    Returns:
        Any: A gseaResult object containing GSEA results for disease terms.

    References:
        https://rdrr.io/bioc/DOSE/man/gseDO.html
    """
    return r_dose.gseDO(geneList=gene_list, **kwargs)


def enrich_ncg(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Perform enrichment analysis using the Network of Cancer Genes database.

    This function performs over-representation analysis to identify enriched
    cancer-related gene categories from the Network of Cancer Genes database.

    Args:
        gene_names: A vector of gene identifiers (usually Entrez gene IDs).
        **kwargs: Additional arguments to pass to the enrichNCG function.
            Common parameters include:
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).
            - readable: Whether to map gene IDs to gene symbols (default: FALSE).

    Returns:
        Any: An enrichResult object containing enriched cancer gene categories.

    References:
        https://rdrr.io/bioc/DOSE/man/enrichNCG.html
    """
    return r_dose.enrichNCG(gene=gene_names, **kwargs)


def gse_ncg(gene_list: ro.FloatVector, **kwargs: Any) -> Any:
    """Perform Gene Set Enrichment Analysis with the Network of Cancer Genes database.

    This function performs Gene Set Enrichment Analysis (GSEA) to identify
    enriched cancer-related gene categories from the Network of Cancer Genes database
    for a ranked list of genes.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes or other metrics that can be
            used to rank genes). The names should be Entrez gene IDs.
        **kwargs: Additional arguments to pass to the gseNCG function.
            Common parameters include:
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - seed: Random seed for reproducibility.

    Returns:
        Any: A gseaResult object containing GSEA results for cancer gene categories.

    References:
        https://rdrr.io/bioc/DOSE/man/gseNCG.html
    """
    return r_dose.gseNCG(geneList=gene_list, **kwargs)
