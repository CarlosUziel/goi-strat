"""
Wrappers for R package annotatr

All functions have pythonic inputs and outputs.

The annotatr package provides a framework for annotating genomic regions with
genomic features (e.g. promoters, exons, introns, enhancers, CpG islands)
and ranges of annotations within those features.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any, Dict, Union

from rpy2.robjects import StrVector
from rpy2.robjects.packages import importr

r_annotatr = importr("annotatr")
r_granges = importr("GenomicRanges")
r_ggplot = importr("ggplot2")


def builtin_annotations() -> StrVector:
    """Get the list of all available built-in annotations.

    This function returns a character vector of all the built-in genomic annotations
    supported by annotatr, including shortcuts. The expand_annotations() function in
    R helps handle these shortcuts for specific annotation types.

    Returns:
        StrVector: A character vector of available annotation codes.

    Examples:
        >>> builtin_annotations()
        # Returns annotation codes like "hg19_cpgs", "hg19_basicgenes", etc.

    References:
        https://rdrr.io/bioc/annotatr/man/builtin_annotations.html
    """
    return r_annotatr.builtin_annotations()


def build_ah_annots(
    genome: str, ah_codes: Dict[str, str], annotation_class: str
) -> Any:
    """Build arbitrary annotations from AnnotationHub resources.

    This helper function allows users to build custom annotations from
    AnnotationHub resources for use with annotatr.

    Args:
        genome: The genome assembly (e.g., "hg19", "hg38", "mm10").
        ah_codes: A dictionary mapping annotation names to AnnotationHub
            resource codes. The keys are names to assign to each annotation,
            and the values are AnnotationHub resource codes.
        annotation_class: The class name for the custom annotations, used to group
            related annotations.

    Returns:
        Any: A character vector of custom annotation names that can be used with
        build_annotations().

    Examples:
        >>> build_ah_annots(
        ...     "hg19",
        ...     {"vista_enhancers": "AH43583"},
        ...     "vista"
        ... )

    References:
        https://rdrr.io/bioc/annotatr/man/build_ah_annots.html
    """
    ah_codes_r = StrVector(ah_codes.values())
    ah_codes_r.names = StrVector(ah_codes.keys())
    return r_annotatr.build_ah_annots(genome, ah_codes_r, annotation_class)


def build_annotations(genome: str, annotations: StrVector) -> Any:
    """Create a GRanges object with the specified genomic annotations.

    This function builds a GRanges object containing all the requested annotations
    for the specified genome. The annotations can be built-in annotations from
    annotatr or custom annotations created with build_ah_annots().

    Args:
        genome: The genome assembly (e.g., "hg19", "hg38", "mm10").
        annotations: A character vector of annotations to build. Supported
            annotation codes can be listed using builtin_annotations().

    Returns:
        Any: A GRanges object containing the genomic coordinates and metadata
        for all requested annotations.

    Notes:
        The basis for enhancer annotations are FANTOM5 data, the basis for CpG
        related annotations are CpG island tracks from AnnotationHub, and the
        basis for genic annotations are from the TxDb packages and org.db
        group of packages.

    Examples:
        >>> build_annotations("hg19", StrVector(["hg19_cpgs", "hg19_genes"]))

    References:
        https://rdrr.io/bioc/annotatr/man/build_annotations.html
    """
    return r_annotatr.build_annotations(genome, annotations)


def annotate_regions(regions: Any, annotations: Any, **kwargs: Any) -> Any:
    """Annotate genomic regions with selected genomic annotations.

    This function associates genomic regions with annotations while preserving
    the data associated with the genomic regions. It performs intersection of
    the regions with annotations and returns the results.

    Args:
        regions: A GRanges object containing genomic regions to annotate,
            typically returned by read_regions().
        annotations: A GRanges object containing genome annotations, typically
            returned by build_annotations().
        **kwargs: Additional arguments to pass to the annotate_regions function.
            Common parameters include:
            - ignore.strand: Whether to ignore strand information during annotation.
            - minoverlap: Minimum overlap required between regions and annotations.
            - upstream: Size of upstream window for annotations (in bp).
            - downstream: Size of downstream window for annotations (in bp).

    Returns:
        Any: A GRanges object containing the annotated regions with additional
        metadata columns describing the annotations.

    Examples:
        >>> regions = read_regions("my_regions.bed")
        >>> annotations = build_annotations("hg19", StrVector(["hg19_cpgs"]))
        >>> annotated = annotate_regions(regions, annotations)

    References:
        https://rdrr.io/bioc/annotatr/man/annotate_regions.html
    """
    return r_annotatr.annotate_regions(regions, annotations, **kwargs)


def summarize_annotations(annotated_regions: Any, **kwargs: Any) -> Any:
    """Summarize the annotations for a set of genomic regions.

    Given a GRanges of annotated regions, this function counts the number of regions
    in each annotation type. If annotated_random is provided, the same summary is
    computed for the random regions, allowing for comparison.

    Args:
        annotated_regions: A GRanges object containing annotated regions,
            typically the result of annotate_regions().
        **kwargs: Additional arguments to pass to the summarize_annotations function.
            Common parameters include:
            - annotated_random: Annotated random regions for comparison.
            - by_region: Whether to count each region once per annotation type (TRUE)
              or count all overlaps (FALSE, default).
            - annotate_by_group: The annotation column to summarize annotations by.

    Returns:
        Any: A data.frame containing the count of regions in each annotation type.

    Examples:
        >>> summary = summarize_annotations(annotated_regions)

    References:
        https://rdrr.io/bioc/annotatr/man/summarize_annotations.html
    """
    return r_annotatr.summarize_annotations(annotated_regions, **kwargs)


def plot_annotation(
    annotated_regions: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a bar plot of region counts for different annotation types.

    This function plots the number of regions with the corresponding genomic
    annotations. If a region is annotated to multiple annotations of the same type,
    the region will only be counted once in the corresponding bar plot.

    Args:
        annotated_regions: A GRanges object containing annotated regions,
            typically the result of annotate_regions().
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the plot_annotation function.
            Common parameters include:
            - annotated_random: Annotated random regions for comparison.
            - annotation_order: Order of annotations in the plot.
            - plot_title: Title for the plot.
            - x_label: Label for the x-axis.
            - y_label: Label for the y-axis.
            - plot_label_size: Size of the plot labels.
            - legend_title: Title for the legend.

    Notes:
        For example, if a region were annotated to multiple exons, it would only
        count once toward the exon bar in the plot, but if it were annotated to
        an exon and an intron, it would count towards both.

    References:
        https://rdrr.io/bioc/annotatr/man/plot_annotation.html
    """
    plot = r_annotatr.plot_annotation(annotated_regions, **kwargs)
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)


def randomize_regions(regions: Any, **kwargs: Any) -> Any:
    """Generate random genomic regions matching the properties of input regions.

    This function is a wrapper for regioneR::randomizeRegions() that simplifies
    the creation of randomized regions for an input set of regions. It uses the
    seqlengths of the input regions to build the appropriate genome object.

    Args:
        regions: A GRanges object containing the original genomic regions,
            typically returned by read_regions().
        **kwargs: Additional arguments to pass to the randomize_regions function.
            Common parameters include:
            - count: Number of random regions to generate (default: same as input).
            - allow.overlaps: Whether to allow overlaps among random regions.
            - per.chromosome: Whether to maintain the per-chromosome distribution.
            - non.overlapping: Whether random regions should not overlap with original regions.

    Returns:
        Any: A GRanges object containing randomly generated regions with the same
        properties (width distribution, chromosome distribution) as the input regions.

    Examples:
        >>> random_regions = randomize_regions(regions, count=1000)

    References:
        https://rdrr.io/bioc/annotatr/man/randomize_regions.html
    """
    return r_annotatr.randomize_regions(regions, **kwargs)


def read_regions(file_path: Union[str, Path], **kwargs: Any) -> Any:
    """Read genomic regions from a file and create a GRanges object.

    This function reads genomic regions from a file (BED, GFF, etc.) and
    returns a GRanges object that can be used with other annotatr functions.

    Args:
        file_path: Path to the file containing genomic regions.
        **kwargs: Additional arguments to pass to the read_regions function.
            Common parameters include:
            - format: File format (e.g., "bed", "gff", "csv").
            - genome: Genome assembly for the regions.
            - rename: Whether to rename columns based on format.
            - fix_ncbi: Whether to fix NCBI-style chromosome names.

    Returns:
        Any: A GRanges object containing the genomic regions from the file.

    Examples:
        >>> regions = read_regions("my_regions.bed", format="bed", genome="hg19")

    References:
        https://rdrr.io/bioc/annotatr/man/read_regions.html
    """
    return r_annotatr.read_regions(str(file_path), **kwargs)
