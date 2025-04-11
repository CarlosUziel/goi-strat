"""
Wrappers for R package DMRcate

All functions have pythonic inputs and outputs.

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


def cpg_annotate(obj: Any, design: Any, **kwargs):
    """
    Annotate CpGs with their chromosome position and test statistic

    Either:

    - Annotate a matrix of M-values (logit transform of beta)
      representing 450K or EPIC data with probe weights
      (depending on analysis.type) and chromosomal position,
    - Standardise this information from DSS:::DMLtest()
      to the same data format.

    See: https://rdrr.io/bioc/DMRcate/man/cpg.annotate.html

    Args:
        obj: Either:

            - A matrix of M-values, with unique Illumina probe IDs
              as rownames and unique sample IDs as column names, or
            - Output from DSS:::DMLtest().
        design: Study design matrix. Identical context to differential
            analysis pipeline in limma. Must have an
            intercept if contrasts=FALSE. Applies only when
            analysis.type="differential". Only applicable when
            datatype="array".
    """
    return r_dmrcate.cpg_annotate(object=obj, design=design, **kwargs)


def dmr_cate(obj: Any, **kwargs):
    """
    DMR identification
    The main function of this package. Computes a kernel estimate against a
    null comparison to identify significantly differentially (or variable)
    methylated regions.

    See: https://rdrr.io/bioc/DMRcate/man/dmrcate.html

    Args:
        obj: A class of type "annot", created from cpg.annotate.
    """
    return r_dmrcate.dmrcate(object=obj, **kwargs)


def extract_ranges(dmrs: Any, **kwargs):
    """
    Takes a DMResults object and produces the corresponding GRanges object.

    See: https://rdrr.io/bioc/DMRcate/man/extractRanges.html

    Args:
        dmrs: A DMResults object.
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
    **kwargs,
):
    """
    Plots an individual DMR (in context of possibly other DMRs) as found by
    dmrcate. Heatmaps are shown as well as proximal coding regions, smoothed
    group means and chromosome ideogram.

    See: https://rdrr.io/bioc/DMRcate/man/DMR.plot.html

    Args:
        ranges: A GRanges object (ostensibly created by extractRanges())
            describing DMR coordinates.
        dmr: Index of ranges (one integer only) indicating which DMR to be
            plotted.
        cpgs: Either:

            - A matrix of beta values for plotting, with unique Illumina
              probe IDs as rownames.
            - A GenomicRatioSet, annotated with the appropriate array and
              data types
            - A BSseq object containing per-CpG methylation and coverage
              counts for the samples to be plotted
        sample_groups: Targets sample groups
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
