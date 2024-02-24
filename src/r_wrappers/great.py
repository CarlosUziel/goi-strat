"""
    Wrappers for R package rGREAT

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from typing import Any

from rpy2.robjects.packages import importr

r_source = importr("rGREAT")


def great(
    gr: Any, gene_sets: str = "MSigDB:C2", tss_source: str = "txdb:mm10", **kwargs
) -> Any:
    """
     Perform GREAT analysis.

    *ref docs: https://rdrr.io/github/jokergoo/rGREAT/man/great.html
    *see also: https://bioconductor.org/packages/release/bioc/vignettes/rGREAT/inst/doc/
        local-GREAT.html#Perform_local_GREAT_with_great()

    Args:
        gr: A GRanges object. This is the input regions. It is important to keep
            consistent for the chromosome names of the input regions and the
            internal TSS regions. Use getTSS to see the format of internal TSS
            regions.
        gene_sets: A single string of defautly supported gene sets collections
            (see the full list in "Genesets" section), or a named list of vectors
            where each vector correspond to a gene set.
        tss_source: Source of TSS. See "TSS" section.
    """
    return r_source.great(gr, gene_sets=gene_sets, tss_source=tss_source, **kwargs)


def get_region_gene_associations(great_object: Any, **kwargs) -> Any:
    """
    Get region-gene associations.

    *ref docs:
        https://rdrr.io/github/jokergoo/rGREAT/man/getRegionGeneAssociations-
        GreatObject-method.html
    *see also: https://bioconductor.org/packages/release/bioc/vignettes/rGREAT/inst/doc/
        local-GREAT.html#Perform_local_GREAT_with_great()

    Args:
        great_object: A GreatObject-class object returned by great.

    Returns:
        A GRanges object. Please the two meta columns are in formats of CharacterList
        and IntegerList because a region may associate to multiple genes.
    """
    return r_source.getRegionGeneAssociations(great_object, **kwargs)
