"""
Wrappers for R package BiSeq

All functions have pythonic inputs and outputs.

BiSeq provides tools for processing and analyzing bisulfite sequencing data,
particularly focusing on detecting differentially methylated regions (DMRs)
in targeted bisulfite sequencing data.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
    R --> data.category
    Python --> data_category
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntMatrix

from r_wrappers.utils import pd_df_to_rpy2_df

r_source = importr("BiSeq")

pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def bs_raw(
    metadata: Dict[str, str],
    row_ranges: Any,
    col_data: pd.DataFrame,
    total_reads: IntMatrix,
    meth_reads: IntMatrix,
) -> Any:
    """Create a BSraw object containing bisulfite sequencing data.

    The BSraw class is derived from RangedSummarizedExperiment and contains a SimpleList
    of matrices named methReads and totalReads as assays.

    Args:
        metadata: An optional list of arbitrary content describing the overall
            experiment.
        row_ranges: Object of class "GRanges" containing the genome positions of
            CpG-sites covered by bisulfite sequencing.
        col_data: Object of class "DataFrame" containing information on variable values
            of the samples.
        total_reads: Contains the number of reads spanning a CpG-site. The rows
            represent the CpG sites in rowRanges and the columns represent the samples
            in colData.
        meth_reads: Contains the number of methylated reads spanning a CpG-site.

    Returns:
        Any: A BSraw object for use in downstream BiSeq analysis functions.

    References:
        https://rdrr.io/bioc/BiSeq/man/BSraw-class.html
    """
    return r_source.BSraw(
        metadata=ro.ListVector(metadata),
        rowRanges=row_ranges,
        colData=pd_df_to_rpy2_df(col_data),
        totalReads=total_reads,
        meprocesses=meth_reads,
    )


def cov_statistics(biseq_obj: Any) -> pd.DataFrame:
    """Get coverage statistics for bisulfite sequencing data.

    This function produces information per sample about:
    1) The number of covered CpG-sites
    2) The median of their coverages

    Args:
        biseq_obj: A BiSeq object (BSraw or BSrel).

    Returns:
        pd.DataFrame: A DataFrame with coverage statistics for each sample.

    References:
        https://rdrr.io/bioc/BiSeq/man/covStatistics.html
    """
    return r_source.covStatistics(biseq_obj)


def cov_boxplot(
    biseq_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create boxplots of CpG site coverage per sample.

    A boxplot per sample is plotted for the coverages of CpG-sites. It is constrained
    to CpG-sites which are covered in the respective sample (coverage != 0 and not
    NA).

    Args:
        biseq_obj: A BSraw object containing bisulfite sequencing data.
        save_path: Path where the generated plot will be saved.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to the covBoxplots function.
            Common parameters include:
            - col: Colors for the boxplots.
            - main: Plot title.
            - ylim: y-axis limits.

    References:
        https://rdrr.io/bioc/BiSeq/man/covBoxplots.html
    """
    pdf(str(save_path), width=width, height=height)
    r_source.covBoxplots(biseq_obj, **kwargs)
    dev_off()


def cluster_sites(
    biseq_obj: Any,
    groups: Optional[Any] = None,
    perc_samples: float = 10 / 12,
    min_sites: int = 20,
    max_dist: int = 100,
    **kwargs: Any,
) -> Any:
    """Identify clusters of CpG sites across samples.

    Within a BSraw object, this function searches for agglomerations of CpG sites across
    all samples. The process works in two steps:

    1. The data is reduced to CpG sites covered in round(perc_samples * ncol(object)) samples,
       these are called 'frequently covered CpG sites'.
    2. Regions are detected where not less than min_sites frequently covered CpG sites
       are sufficiently close to each other (max_dist).

    Note that the frequently covered CpG sites are considered to define the
    boundaries of the CpG clusters only. For the subsequent analysis, the methylation
    data of all CpG sites within these clusters are used.

    Args:
        biseq_obj: A BiSeq object (BSraw or BSrel).
        groups: Optional. A factor specifying two or more sample groups within the
            given object. See Details in the BiSeq documentation.
        perc_samples: A numeric between 0 and 1. Is passed to filterBySharedRegions.
            Represents the proportion of samples that should have coverage for a CpG
            site to be considered "frequently covered".
        min_sites: A numeric. Clusters should comprise at least min_sites CpG sites
            which are covered in at least perc_samples of samples, otherwise clusters
            are dropped.
        max_dist: A numeric. CpG sites which are covered in at least perc_samples of
            samples within a cluster should not be more than max_dist bp apart from
            their nearest neighbors.
        **kwargs: Additional arguments to pass to the clusterSites function.

    Returns:
        Any: A BSraw or BSrel object with an additional column "cluster.id" in the rowRanges slot,
        indicating the cluster membership of each CpG site.

    References:
        https://rdrr.io/bioc/BiSeq/man/clusterSites.html
    """
    return r_source.clusterSites(
        object=biseq_obj,
        groups=groups,
        perc_samples=perc_samples,
        min_sites=min_sites,
        max_dist=max_dist,
        **kwargs,
    )


def cluster_sites_to_gr(biseq_obj: Any) -> Any:
    """Extract CpG cluster coordinates as a GRanges object.

    This function retrieves the start and end positions of CpG clusters from a
    BSraw or BSrel object when there is a cluster.id column in the rowRanges slot.

    Args:
        biseq_obj: A BSraw or BSrel object with a cluster.id column in the rowRanges
            slot. Usually the output of clusterSites().

    Returns:
        Any: A GRanges object containing the genomic coordinates of the CpG clusters.

    References:
        https://rdrr.io/bioc/BiSeq/man/clusterSitesToGR.html
    """
    return r_source.clusterSitesToGR(biseq_obj)
