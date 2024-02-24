"""
    Wrappers for R package DOSE

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

from typing import Any

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

from components.functional_analysis.orgdb import OrgDB

r_dose = importr("DOSE")


def set_readable(enrich_result: Any, org_db: OrgDB, **kwargs):
    """
    Mapping geneID to gene Symbol

    *ref docs: https://rdrr.io/bioc/DOSE/man/setReadable.html
    """
    return r_dose.setReadable(enrich_result, org_db.db, **kwargs)


def enrich_do(gene_names: ro.StrVector, **kwargs):
    """
    Given a vector of genes, this function will return the enrichment DO
    categories with FDR control.

    *ref docs: https://rdrr.io/bioc/DOSE/man/enrichDO.html
    """
    return r_dose.enrichDO(gene=gene_names, **kwargs)


def gse_do(gene_list: ro.FloatVector, **kwargs):
    """
    Perform GSEA of DO categories.

    *ref docs: https://rdrr.io/bioc/DOSE/man/gseDO.html
    """
    return r_dose.gseDO(geneList=gene_list, **kwargs)


def enrich_ncg(gene_names: ro.StrVector, **kwargs):
    """
    Enrichment analysis based on the Network of Cancer Genes database
        (http://ncg.kcl.ac.uk/)

    *ref docs: https://rdrr.io/bioc/DOSE/man/enrichNCG.html
    """
    return r_dose.enrichNCG(gene=gene_names, **kwargs)


def gse_ncg(gene_list: ro.FloatVector, **kwargs):
    """
    Perform GSEA based on the Network of Cancer Genes database
        (http://ncg.kcl.ac.uk/)

    *ref docs: https://rdrr.io/bioc/DOSE/man/gseNCG.html
    """
    return r_dose.gseNCG(geneList=gene_list, **kwargs)
