"""
Wrappers for R package clusterProfiler

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

r_cluster_profiler = importr("clusterProfiler")


def enrich_kegg(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Perform KEGG pathway enrichment analysis on a gene set.

    This function performs over-representation analysis to identify enriched
    KEGG pathways for a given set of genes, with FDR control for multiple testing.

    Args:
        gene_names: A vector of gene identifiers (usually Entrez gene IDs).
        **kwargs: Additional arguments to pass to the enrichKEGG function.
            Common parameters include:
            - organism: Organism name, e.g., "hsa" for human.
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).
            - use_internal_data: Whether to use KEGG data from the package (default: FALSE).

    Returns:
        Any: An enrichResult object containing enriched KEGG pathways.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/enrichKEGG.html
    """
    return r_cluster_profiler.enrichKEGG(gene=gene_names, **kwargs)


def gse_kegg(gene_list: ro.FloatVector, **kwargs: Any) -> Any:
    """Perform Gene Set Enrichment Analysis with KEGG pathways.

    This function performs GSEA to identify enriched KEGG pathways
    for a ranked list of genes.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes). The names should be Entrez gene IDs.
        **kwargs: Additional arguments to pass to the gseKEGG function.
            Common parameters include:
            - organism: Organism name, e.g., "hsa" for human.
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - seed: Random seed for reproducibility.
            - by: Method for ranking genes (default: "fgsea").

    Returns:
        Any: A gseaResult object containing GSEA results for KEGG pathways.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/gseKEGG.html
    """
    return r_cluster_profiler.gseKEGG(geneList=gene_list, **kwargs)


def enrich_mkegg(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Perform KEGG Module enrichment analysis on a gene set.

    This function performs over-representation analysis to identify enriched
    KEGG Modules for a given set of genes, with FDR control for multiple testing.
    KEGG Modules are tighter functional units than KEGG Pathways.

    Args:
        gene_names: A vector of gene identifiers (usually Entrez gene IDs).
        **kwargs: Additional arguments to pass to the enrichMKEGG function.
            Common parameters include:
            - organism: Organism name, e.g., "hsa" for human.
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).

    Returns:
        Any: An enrichResult object containing enriched KEGG Modules.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/enrichMKEGG.html
    """
    return r_cluster_profiler.enrichMKEGG(gene=gene_names, **kwargs)


def gse_mkegg(gene_list: ro.FloatVector, **kwargs: Any) -> Any:
    """Perform Gene Set Enrichment Analysis with KEGG Modules.

    This function performs GSEA to identify enriched KEGG Modules
    for a ranked list of genes.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes). The names should be Entrez gene IDs.
        **kwargs: Additional arguments to pass to the gseMKEGG function.
            Common parameters include:
            - organism: Organism name, e.g., "hsa" for human.
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - seed: Random seed for reproducibility.

    Returns:
        Any: A gseaResult object containing GSEA results for KEGG Modules.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/gseMKEGG.html
    """
    return r_cluster_profiler.gseMKEGG(geneList=gene_list, **kwargs)


def group_go(gene_names: ro.StrVector, org_db: OrgDB, **kwargs: Any) -> Any:
    """Generate a functional profile of a gene set at a specific GO level.

    This function classifies genes based on Gene Ontology terms at a specific level.
    Unlike enrichGO, this function does not perform statistical testing.

    Args:
        gene_names: A vector of gene identifiers.
        org_db: An organism database object containing GO annotations.
        **kwargs: Additional arguments to pass to the groupGO function.
            Common parameters include:
            - ont: GO ontology, one of "BP" (Biological Process), "MF" (Molecular Function),
              or "CC" (Cellular Component).
            - level: The GO level for which to extract the terms.
            - keyType: Type of gene identifier provided.
            - readable: Whether to map gene IDs to gene symbols (default: FALSE).

    Returns:
        Any: A groupGOResult object containing GO classification results.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/groupGO.html
    """
    return r_cluster_profiler.groupGO(gene=gene_names, OrgDb=org_db.db, **kwargs)


def enrich_go(gene_names: ro.StrVector, org_db: OrgDB, **kwargs: Any) -> Any:
    """Perform Gene Ontology enrichment analysis on a gene set.

    This function performs over-representation analysis to identify enriched
    GO terms for a given set of genes, with FDR control for multiple testing.

    Args:
        gene_names: A vector of gene identifiers.
        org_db: An organism database object containing GO annotations.
        **kwargs: Additional arguments to pass to the enrichGO function.
            Common parameters include:
            - ont: GO ontology, one of "BP" (Biological Process), "MF" (Molecular Function),
              or "CC" (Cellular Component), or "ALL" for all ontologies.
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - keyType: Type of gene identifier provided.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).
            - readable: Whether to map gene IDs to gene symbols (default: FALSE).

    Returns:
        Any: An enrichResult object containing enriched GO terms.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/enrichGO.html
    """
    return r_cluster_profiler.enrichGO(gene=gene_names, OrgDb=org_db.db, **kwargs)


def gse_go(gene_list: ro.FloatVector, org_db: OrgDB, **kwargs: Any) -> Any:
    """Perform Gene Set Enrichment Analysis with Gene Ontology.

    This function performs GSEA to identify enriched GO terms
    for a ranked list of genes.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes).
        org_db: An organism database object containing GO annotations.
        **kwargs: Additional arguments to pass to the gseGO function.
            Common parameters include:
            - ont: GO ontology, one of "BP" (Biological Process), "MF" (Molecular Function),
              or "CC" (Cellular Component).
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - keyType: Type of gene identifier provided.
            - seed: Random seed for reproducibility.
            - by: Method for ranking genes (default: "fgsea").

    Returns:
        Any: A gseaResult object containing GSEA results for GO terms.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/gseGO.html
    """
    return r_cluster_profiler.gseGO(geneList=gene_list, OrgDb=org_db.db, **kwargs)


def enricher(gene_names: ro.StrVector, **kwargs: Any) -> Any:
    """Perform enrichment analysis with custom gene sets.

    This is a universal enrichment analysis function that can work with any
    user-provided gene sets, not limited to GO, KEGG, etc.

    Args:
        gene_names: A vector of gene identifiers.
        **kwargs: Additional arguments to pass to the enricher function.
            Common parameters include:
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - universe: Background genes to use for enrichment analysis.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - qvalueCutoff: q-value cutoff (default: 0.2).
            - TERM2GENE: A data frame with term to gene mappings.
            - TERM2NAME: Optional data frame with term ID to term name mappings.

    Returns:
        Any: An enrichResult object containing enrichment results.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/enricher.html
    """
    return r_cluster_profiler.enricher(gene=gene_names, **kwargs)


def gsea(gene_list: ro.FloatVector, **kwargs: Any) -> Any:
    """Perform Gene Set Enrichment Analysis with custom gene sets.

    This is a universal GSEA function that can work with any user-provided
    gene sets, not limited to GO, KEGG, etc.

    Args:
        gene_list: A named vector with gene IDs as names and ranking metric
            as values (e.g., log fold changes).
        **kwargs: Additional arguments to pass to the GSEA function.
            Common parameters include:
            - pvalueCutoff: P-value cutoff (default: 0.05).
            - pAdjustMethod: Method for multiple testing correction (default: "BH").
            - nPerm: Number of permutations for calculating significance.
            - minGSSize: Minimum size of gene sets to consider.
            - maxGSSize: Maximum size of gene sets to consider.
            - seed: Random seed for reproducibility.
            - TERM2GENE: A data frame with term to gene mappings.
            - TERM2NAME: Optional data frame with term ID to term name mappings.
            - by: Method for ranking genes (default: "fgsea").

    Returns:
        Any: A gseaResult object containing GSEA results.

    References:
        https://rdrr.io/bioc/clusterProfiler/man/GSEA.html
    """
    return r_cluster_profiler.GSEA(geneList=gene_list, **kwargs)
