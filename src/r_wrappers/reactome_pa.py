"""
    Wrappers for R package ReactomePA

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

r_reactome_pa = importr("ReactomePA")


def enrich_reactome(gene_names: ro.StrVector, **kwargs):
    """
    Pathway Enrichment Analysis of a gene set. Given a vector of genes, this
    function will return the enriched pathways with FDR control.

    *ref docs: https://rdrr.io/bioc/ReactomePA/man/enrichPathway.html
    """
    return r_reactome_pa.enrichPathway(gene=gene_names, **kwargs)


def gsea_reactome(gene_list: ro.FloatVector, **kwargs):
    """
    Gene Set Enrichment Analysis of Reactome Pathway

    *ref docs: https://rdrr.io/bioc/ReactomePA/man/gsePathway.html
    """
    return r_reactome_pa.gsePathway(geneList=gene_list, **kwargs)
