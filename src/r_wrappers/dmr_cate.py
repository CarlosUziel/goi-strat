"""
Wrappers for R package DMRcate

All functions have pythonic inputs and outputs.

DMRcate is a package for the discovery and annotation of differentially methylated
regions (DMRs) from both array and sequencing data, using a kernel-based approach.
DMRcate can accommodate complex experimental designs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
    R --> data.category
    Python --> data_category
"""

from pathlib import Path
from typing import Any

from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import IntVector, StrVector
from rpy2.robjects.packages import importr

r_dmrcate = importr("DMRcate")

pdf = robjects.r("pdf")
dev_off = robjects.r("dev.off")


def cpg_annotate(obj: Any, design: Any, **kwargs: Any) -> Any:
    """Annotate CpGs with their chromosome position and test statistic.

    This function performs one of two tasks:

    1. Annotate a matrix of M-values (logit transform of beta) representing 450K or EPIC data
       with probe weights (depending on analysis.type) and chromosomal position, or
    2. Standardize information from DSS:::DMLTest() to the same data format.

    Args:
        obj: Either:
            - A matrix of M-values, with unique Illumina probe IDs as rownames and unique
              sample IDs as column names, or
            - Output from DSS:::DMLtest().
        design: Study design matrix. Identical context to differential analysis pipeline
            in limma. Must have an intercept if contrasts=FALSE. Applies only when
            analysis.type="differential". Only applicable when datatype="array".
        **kwargs: Additional arguments to pass to the cpg.annotate function.
            Common parameters include:
            - datatype: Type of methylation data, either "array" or "sequencing".
            - what: Type of differential methylation analysis, either "M" (default) or "Beta".
            - analysis.type: Either "differential" (default) or "variability".
            - contrasts: Whether design matrix should be converted to contrasts.
            - cont.matrix: Contrast matrix to be used if contrasts=TRUE.
            - coef: Coefficient(s) to test against 0 if analysis.type="differential".
            - fdr: FDR cutoff for significant probes if analysis.type="differential".

    Returns:
        Any: An annotation object with CpG information, including chromosomal
        coordinates and test statistics. This object can be passed to dmr_cate().

    References:
        https://rdrr.io/bioc/DMRcate/man/cpg.annotate.html
    """
    return r_dmrcate.cpg_annotate(object=obj, design=design, **kwargs)


def dmr_cate(obj: Any, **kwargs: Any) -> Any:
    """Identify differentially methylated regions.

    The main function of the DMRcate package. Computes a kernel estimate against
    a null comparison to identify significantly differentially (or variable)
    methylated regions.

    Args:
        obj: A class of type "annot", created from cpg_annotate().
        **kwargs: Additional arguments to pass to the dmrcate function.
            Common parameters include:
            - lambda: Gaussian kernel bandwidth (default: 1000).
            - C: Scaling factor for bandwidth (default: 2).
            - min.cpgs: Minimum number of CpGs for DMR (default: 2).
            - pcutoff: FDR cutoff for DMR significance (default: 0.05).
            - consec: Whether significant CpGs should be consecutive (default: FALSE).
            - conseclambda: Consecutive distance for DMRS (default: lambda).

    Returns:
        Any: A DMResults object containing the identified differentially methylated regions.
        This object can be further processed using extract_ranges().

    References:
        https://rdrr.io/bioc/DMRcate/man/dmrcate.html
    """
    return r_dmrcate.dmrcate(object=obj, **kwargs)


def extract_ranges(dmrs: Any, **kwargs: Any) -> Any:
    """Extract GRanges object from DMResults object.

    Takes a DMResults object and produces the corresponding GRanges object
    containing the genomic coordinates of identified DMRs.

    Args:
        dmrs: A DMResults object from dmr_cate().
        **kwargs: Additional arguments to pass to the extractRanges function.
            Common parameters include:
            - genome: Genome build to use for annotations (default: "hg19").
            - threshold: Significance threshold for DMRs (default: 0.05).

    Returns:
        Any: A GRanges object containing the genomic coordinates and metadata
        for the identified differentially methylated regions.

    References:
        https://rdrr.io/bioc/DMRcate/man/extractRanges.html
    """
    return r_dmrcate.extractRanges(dmrs, **kwargs)


def dmr_plot(
    ranges: Any,
    dmr: int,
    cpgs: Any,
    sample_groups: StrVector,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Plot an individual DMR with genomic context and methylation patterns.

    Plots an individual DMR (in context of possibly other DMRs) as found by
    dmrcate. Heatmaps are shown as well as proximal coding regions, smoothed
    group means and chromosome ideogram.

    Args:
        ranges: A GRanges object (typically created by extract_ranges())
            describing DMR coordinates.
        dmr: Index of ranges (one integer only) indicating which DMR to be plotted.
        cpgs: Either:
            - A matrix of beta values for plotting, with unique Illumina probe IDs as rownames.
            - A GenomicRatioSet, annotated with the appropriate array and data types.
            - A BSseq object containing per-CpG methylation and coverage counts for the samples.
        sample_groups: Target sample groups for color coding in the plot.
        save_path: Path where the generated plot will be saved.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to the DMR.plot function.
            Common parameters include:
            - genome: Genome build to use for annotations (default: "hg19").
            - CpGs.disp: Number of CpGs to display (default: 50).
            - mainCol: Color for main plot title.
            - stat: Test statistic for coloring DMR (default: "beta").

    Raises:
        RRuntimeError: If there is an error in the R function call.

    References:
        https://rdrr.io/bioc/DMRcate/man/DMR.plot.html
    """
    # 0. Setup
    get_colors = robjects.r(
        """
        library(RColorBrewer)
        f <- function(sample_groups){
            pal <- brewer.pal(8,"Dark2")
            groups <- pal[1:length(unique(sample_groups))]
            names(groups) <- levels(factor(sample_groups))
            colors <- groups[as.character(factor(sample_groups))]
            return(colors)
        }
        """
    )
    colors = get_colors(sample_groups)
    samps = IntVector(list(range(1, len(sample_groups) + 1)))

    # 1. Plot DMR
    pdf(str(save_path), width=width, height=height)
    try:
        r_dmrcate.DMR_plot(
            ranges=ranges, dmr=dmr, CpGs=cpgs, phen_col=colors, samps=samps, **kwargs
        )
    except RRuntimeError as e:
        print(e)
    dev_off()
