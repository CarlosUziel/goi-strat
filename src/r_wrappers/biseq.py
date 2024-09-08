"""
    Wrappers for R package BiSeq

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from pathlib import Path
from typing import Any, Dict

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
    """
    The BSraw class is derived from RangedSummarizedExperiment and contains a SimpleList
        of matrices named methReads and totalReads as assays.

    See: https://rdrr.io/bioc/BiSeq/man/BSraw-class.html

    Args:
        metadata: An optional list of arbitrary content describing the overall
            experiment.
        row_ranges: Object of class "GRanges" containing the genome positions of
            CpG-sites covered by bisulfite sequencing. WARNING: The accessor for this
            slot is rowRanges, not rowRanges!
        col_data: Object of class "DataFrame" containing information on variable values
            of the samples.
        total_reads: Contains the number of reads spanning a CpG-site. The rows
            represent the CpG sites in rowRanges and the columns represent the samples
            in colData. The matrix methReads
        meth_reads: Contains the number of methylated reads spanning a CpG-site.
    """
    return r_source.BSraw(
        metadata=ro.ListVector(metadata),
        rowRanges=row_ranges,
        colData=pd_df_to_rpy2_df(col_data),
        totalReads=total_reads,
        methReads=meth_reads,
    )


def cov_statistics(biseq_obj: Any) -> Any:
    """
    This function produces information per samples about
        1.) the covered CpG-sites
        2.) the median of their coverages.

    See: https://rdrr.io/bioc/BiSeq/man/covStatistics.html

    Args:
        biseq_obj: A BiSeq object.
    """
    return r_source.covStatistics(biseq_obj)


def cov_boxplot(
    biseq_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> None:
    """
     A boxplot per sample is plotted for the coverages of CpG-sites. It is constrained
        to CpG-sites which are covered in the respective sample (coverage != 0 and not
        NA).

    See: https://rdrr.io/bioc/BiSeq/man/covBoxplots.html

    Args:
        biseq_obj: A BSraw.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_source.covBoxplots(biseq_obj, **kwargs)
    dev_off()


def cluster_sites(
    biseq_obj: Any,
    groups: Any,
    perc_samples: float = 10 / 12,
    min_sites: int = 20,
    max_dist: int = 100,
    **kwargs
):
    """
    Within a BSraw object clusterSites searches for agglomerations of CpG sites across
        all samples. In a first step the data is reduced to CpG sites covered in
        round(perc.samples*ncol(object)) samples, these are called 'frequently covered
        CpG sites'. In a second step regions are detected where not less than min.sites
        frequently covered CpG sites are sufficiantly close to each other (max.dist).
        Note, that the frequently covered CpG sites are considered to define the
        boundaries of the CpG clusters only. For the subsequent analysis the methylation
        data of all CpG sites within these clusters are used.

    See: https://rdrr.io/bioc/BiSeq/man/clusterSites.html

    Args:
        biseq_obj: A BiSeq object.
        groups: OPTIONAL. A factor specifying two or more sample groups within the
            given object. See Details.
        perc_samples: A numeric between 0 and 1. Is passed to filterBySharedRegions.
        min_sites: A numeric. Clusters should comprise at least min.sites CpG sites
            which are covered in at least perc.samples of samples, otherwise clusters
            are dropped.
        max_dist: A numeric. CpG sites which are covered in at least perc.samples of
            samples within a cluster should not be more than max.dist bp apart from
            their nearest neighbors.
    """
    return r_source.clusterSites(
        object=biseq_obj,
        groups=groups,
        perc_samples=perc_samples,
        min_sites=min_sites,
        max_dist=max_dist,
        **kwargs
    )


def cluster_sites_to_gr(biseq_obj: Any) -> Any:
    """c
    This function allows to get the start and end positions of CpG clusters from a
        BSraw or BSrel object, when there is a cluster.id column in the rowRanges slot.

    See: https://rdrr.io/bioc/BiSeq/man/clusterSitesToGR.html

    Args:
        biseq_obj: A BSraw or BSrel object with a cluster.id column in the rowRanges
            slot. Usually the output of clusterSites.
    """
    return r_source.clusterSitesToGR(biseq_obj)
