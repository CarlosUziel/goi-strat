"""
Wrappers for R package annotatr

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any, Dict

from rpy2.robjects import StrVector
from rpy2.robjects.packages import importr

r_annotatr = importr("annotatr")
r_granges = importr("GenomicRanges")
r_ggplot = importr("ggplot2")


def builtin_annotations():
    """
    This includes the shortcuts. The expand_annotations() function helps handle the
    shortcuts.

    See: https://rdrr.io/bioc/annotatr/man/builtin_annotations.html
    """
    return r_annotatr.builtin_annotations()


def build_ah_annots(genome: str, ah_codes: Dict[str, str], annotation_class: str):
    """
    A helper function to build arbitrary annotatinos from AnnotationHub.

    See: https://rdrr.io/bioc/annotatr/man/build_ah_annots.html
    """
    ah_codes_r = StrVector(ah_codes.values())
    ah_codes_r.names = StrVector(ah_codes.keys())
    return r_annotatr.build_ah_annots(genome, ah_codes_r, annotation_class)


def build_annotations(genome: str, annotations: StrVector):
    """
    Create a GRanges object consisting of all the desired annotations.
    Supported annotation codes are listed by builtin_annotations(). The basis for
    enhancer annotations are FANTOM5 data, the basis for CpG related annotations are
    CpG island tracks from AnnotationHub, and the basis for genic annotations are from
    the TxDb packages and org.db group of packages.

    Reference documentation: https://rdrr.io/bioc/annotatr/man/build_annotations.html

    Args:
        genome: The genome assembly.
        annotations: A character vector of annotations to build.
    """
    return r_annotatr.build_annotations(genome, annotations)


def annotate_regions(regions: Any, annotations: Any, **kwargs):
    """
    Annotate genomic regions to selected genomic annotations while
    preserving the data associated with the genomic regions.

    See: https://rdrr.io/bioc/annotatr/man/annotate_regions.html

    Args:
        regions: The GRanges object returned by read_regions().
        annotations: GRanges object.
    """
    return r_annotatr.annotate_regions(regions, annotations, **kwargs)


def summarize_annotations(annotated_regions: Any, **kwargs):
    """
    Given a GRanges of annotated regions, count the number of regions in
    each annotation type. If annotated_random is not NULL, then the same
    is computed for the random regions.

    See: https://rdrr.io/bioc/annotatr/man/summarize_annotations.html

    Args:
        annotated_regions: The GRanges result of annotate_regions().
    """
    return r_annotatr.summarize_annotations(annotated_regions, **kwargs)


def plot_annotation(
    annotated_regions: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Given a GRanges of annotated regions, plot the number of regions with
    the corresponding genomic annotations used in annotation_order. If a region is
    annotated to multiple annotations of the same annot.type, the region will only be
    counted once in the corresponding bar plot. For example, if a region were annotated
    to multiple exons, it would only count once toward the exon bar in the plot, but if
    it were annotated to an exon and an intron, it would count towards both.

    See: https://rdrr.io/bioc/annotatr/man/plot_annotation.html

    Args:
        annotated_regions: The GRanges result of annotate_regions().
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    plot = r_annotatr.plot_annotation(annotated_regions, **kwargs)
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)


def randomize_regions(regions: Any, **kwargs):
    """
    A wrapper function for regioneR::randomizeRegions() that simplifies the
    creation of randomized regions for an input set of regions read with read_regions().
    It relies on the seqlengths of regions in order to build the appropriate genome
    object for regioneR::randomizeRegions().

    See: https://rdrr.io/bioc/annotatr/man/randomize_regions.html

    Args:
        regions: A GRanges object from read_regions.
    """
    return r_annotatr.randomize_regions(regions, **kwargs)
