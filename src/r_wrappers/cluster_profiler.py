"""
    Wrappers for R package clusterProfiler

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

from components.functional_analysis.orgdb import OrgDB

r_cluster_profiler = importr("clusterProfiler")


def enrich_kegg(gene_names: ro.StrVector, **kwargs):
    """
    KEGG Enrichment Analysis of a gene set. Given a vector of genes, this
    function will return the enrichment KEGG categories with FDR control.

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/enrichKEGG.html
    """
    return r_cluster_profiler.enrichKEGG(gene=gene_names, **kwargs)


def gse_kegg(gene_list: ro.FloatVector, **kwargs):
    """
    Gene Set Enrichment Analysis of KEGG

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/gseKEGG.html
    """
    return r_cluster_profiler.gseKEGG(geneList=gene_list, **kwargs)


def enrich_mkegg(gene_names: ro.StrVector, **kwargs):
    """
    KEGG Module Enrichment Analysis of a gene set. Given a vector of genes,
    this function will return the enrichment KEGG Module categories with FDR
    control.

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/enrichMKEGG.html
    """
    return r_cluster_profiler.enrichMKEGG(gene=gene_names, **kwargs)


def gse_mkegg(gene_list: ro.FloatVector, **kwargs):
    """
    Gene Set Enrichment Analysis of KEGG Module

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/gseMKEGG.html
    """
    return r_cluster_profiler.gseMKEGG(geneList=gene_list, **kwargs)


def group_go(gene_names: ro.StrVector, org_db: OrgDB, **kwargs):
    """
    Functional Profile of a gene set at specific GO level. Given a vector of
    genes, this function will return the GO profile at a specific level.

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/groupGO.html
    """
    return r_cluster_profiler.groupGO(gene=gene_names, OrgDb=org_db.db, **kwargs)


def enrich_go(gene_names: ro.StrVector, org_db: OrgDB, **kwargs):
    """
    GO Enrichment Analysis of a gene set. Given a vector of genes, this
    function will return the enrichment GO categories after FDR control.

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/enrichGO.html
    """
    return r_cluster_profiler.enrichGO(gene=gene_names, OrgDb=org_db.db, **kwargs)


def gse_go(gene_list: ro.FloatVector, org_db: OrgDB, **kwargs):
    """
    Gene Set Enrichment Analysis of Gene Ontology

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/gseGO.html
    """
    return r_cluster_profiler.gseGO(geneList=gene_list, OrgDb=org_db.db, **kwargs)


def enricher(gene_names: ro.StrVector, **kwargs):
    """
    A universal enrichment analyzer.

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/enricher.html
    """
    return r_cluster_profiler.enricher(gene=gene_names, **kwargs)


def gsea(gene_list: ro.FloatVector, **kwargs):
    """
    A universal gene set enrichment analysis tools.

    *ref docs: https://rdrr.io/bioc/clusterProfiler/man/GSEA.html
    """
    return r_cluster_profiler.GSEA(geneList=gene_list, **kwargs)
