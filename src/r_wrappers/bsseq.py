"""
Wrappers for R package bsseq

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any, List

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r_source = importr("bsseq")


def read_bismark(cov_files: List[Path], **kwargs: Any) -> Any:
    """Parse output files from the Bismark bisulfite alignment suite.

    This function reads the coverage output files from Bismark methylation
    extractor and creates a BSseq object containing methylation and coverage data.

    Args:
        cov_files: List of coverage files (.cov or .cov.gz files), obtained from
            running 'bismark_methylation_extractor' with the --bedGraph option.
        **kwargs: Additional arguments to pass to the read.bismark function.
            Common parameters include:
            - colData: A DataFrame with sample information.
            - rmZeroCov: Whether to remove sites with zero coverage (default: TRUE).
            - strandCollapse: Whether to collapse read counts from both strands (default: TRUE).
            - verbose: Whether to print progress messages (default: TRUE).
            - BPPARAM: BiocParallel parameters for parallel processing.

    Returns:
        Any: A BSseq object containing methylation data, with the following components:
        - M: Methylated read counts
        - Cov: Total read coverage
        - coef: Matrix of smoothed methylation estimates
        - parameters: Parameters used for smoothing
        - gr: GRanges object with genomic coordinates
        - pData: Sample information (if provided)

    References:
        https://rdrr.io/bioc/bsseq/man/read.bismark.html
    """
    return r_source.read_bismark(
        files=ro.StrVector(list(map(str, cov_files))), **kwargs
    )


def get_coverage(bsseq_obj: Any, **kwargs: Any) -> Any:
    """Obtain read coverage information from a BSseq object.

    This function extracts the coverage matrix (number of reads) from a BSseq object,
    with options to filter by type, regions, and samples.

    Args:
        bsseq_obj: An object of class BSseq.
        **kwargs: Additional arguments to pass to the getCoverage function.
            Common parameters include:
            - regions: GRanges object specifying regions to extract coverage for.
            - type: Type of coverage to extract, "Cov" for total coverage (default),
              "M" for methylated read counts.
            - what: Return format, can be "perBase", "perRegion" or "perRegionAverage".
            - samples: Vector of sample indices to extract coverage for.

    Returns:
        Any: The coverage information in the format specified by the 'what' parameter:
        - "perBase": A matrix with rows being genomic positions and columns being samples
        - "perRegion": A list where each element is a matrix of positions × samples
        - "perRegionAverage": A matrix with rows being regions and columns being samples

    References:
        https://rdrr.io/bioc/bsseq/man/getCoverage.html
    """
    return r_source.getCoverage(bsseq_obj, **kwargs)


def bs_smooth(bsseq_obj: Any, **kwargs: Any) -> Any:
    """Smooth bisulfite sequencing data.

    This function performs smoothing of methylation data across the genome using
    local likelihood smoothing. Smoothing helps to estimate methylation levels in
    regions with sparse coverage.

    Args:
        bsseq_obj: An object of class BSseq.
        **kwargs: Additional arguments to pass to the BSmooth function.
            Common parameters include:
            - ns: Number of CpGs in a smoothing window (default: 70).
            - h: Minimum smoothing window half-width in base pairs (default: 1000).
            - maxGap: Maximum gap between two CpGs in base pairs (default: 10^8).
            - verbose: Whether to show progress messages (default: TRUE).
            - parallelBy: Whether to parallelize by sample or chromosome (default: "sample").
            - BPPARAM: BiocParallel parameters for parallel processing.

    Returns:
        Any: A BSseq object with smoothed methylation estimates stored in the 'coef' slot.

    References:
        https://rdrr.io/bioc/bsseq/man/BSmooth.html
    """
    return r_source.BSmooth(bsseq_obj, **kwargs)


def get_methylation(bsseq_obj: Any, **kwargs: Any) -> Any:
    """Extract methylation values from a BSseq object.

    This function extracts methylation levels (proportion of methylated reads)
    from a BSseq object, either raw or smoothed.

    Args:
        bsseq_obj: An object of class BSseq.
        **kwargs: Additional arguments to pass to the getMeth function.
            Common parameters include:
            - regions: GRanges object specifying regions to extract methylation for.
            - type: Type of methylation values to extract, "raw" (default) or "smooth".
            - what: Return format, can be "perBase", "perRegion" or "perRegionAverage".
            - samples: Vector of sample indices to extract methylation for.

    Returns:
        Any: The methylation values in the format specified by the 'what' parameter:
        - "perBase": A matrix with rows being genomic positions and columns being samples
        - "perRegion": A list where each element is a matrix of positions × samples
        - "perRegionAverage": A matrix with rows being regions and columns being samples

    References:
        https://rdrr.io/bioc/bsseq/man/getMeth.html
    """
    return r_source.getMeth(bsseq_obj, **kwargs)


def dmr_find(bs_smooth_obj: Any, **kwargs: Any) -> Any:
    """Find differentially methylated regions (DMRs) in bisulfite sequencing data.

    This function identifies genomic regions with differences in methylation
    levels between sample groups, after smoothing with bs_smooth.

    Args:
        bs_smooth_obj: A BSseq object with smoothed methylation estimates.
        **kwargs: Additional arguments to pass to the dmrFinder function.
            Common parameters include:
            - stat: Statistic to compute for each CpG (t-statistic or mean difference).
            - cutoff: Cutoff for the test statistic.
            - maxGap: Maximum distance between CpGs in a DMR (default: 1000).
            - minNumRegion: Minimum number of CpGs in a DMR (default: 3).
            - coef: Specific coefficient from a multi-coefficient model.
            - chrsPerChunk: Number of chromosomes to process per chunk.
            - mc.cores: Number of cores for parallel computing.

    Returns:
        Any: A list containing two elements:
        - table: Data frame with DMR locations and statistics
        - regions: GRanges object with the DMRs

    References:
        https://rdrr.io/bioc/bsseq/man/dmrFinder.html
    """
    return r_source.dmrFinder(bs_smooth_obj, **kwargs)
