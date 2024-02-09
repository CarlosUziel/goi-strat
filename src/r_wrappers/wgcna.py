"""
    Wrappers for R package WGCNA

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""
import multiprocessing
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_wgcna = importr("WGCNA")
pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def enable_threads(
    threads: int = multiprocessing.cpu_count(),
):
    """
    Enables parallel calculations within user-level R functions as well as within the
    compiled code, and registers an appropriate parallel calculation back-end for the
    operating system/platform.
    *ref docs docs in:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/allowWGCNAThreads
    """
    r_wgcna.enableWGCNAThreads(threads)


def adjacency(data: pd.DataFrame, **kwargs):
    """
    Calculates (correlation or distance) network adjacency from given expression data
    or from a similarity.
    *ref docs docs in:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/adjacency
    """
    return r_wgcna.adjacency(pd_df_to_rpy2_df(data), **kwargs)


def network_concepts(data: pd.DataFrame, **kwargs):
    """
    This functions calculates various network concepts (topological properties, network
    indices) of a network calculated from expression data. See details for a detailed
    description.

    *ref docs docs in:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/networkConcepts
    """
    return r_wgcna.networkConcepts(pd_df_to_rpy2_df(data), **kwargs)


def pick_soft_threshold(data: pd.DataFrame, **kwargs):
    """
    Analysis of scale free topology for multiple soft thresholding powers. The aim is
    to help the user pick an appropriate soft-thresholding power for network
    construction.

    *ref docs:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/pickSoftThreshold
    """
    return r_wgcna.pickSoftThreshold(pd_df_to_rpy2_df(data), **kwargs)


def blockwise_modules(data: pd.DataFrame, **kwargs):
    """
    This function performs automatic network construction and module detection on
    large expression datasets in a block-wise manner.

    *ref docs:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/blockwiseModules
    """
    return r_wgcna.blockwiseModules(pd_df_to_rpy2_df(data), **kwargs)


def blockwise_modules_iterative(data: pd.DataFrame, **kwargs):
    """
    Iterative version of the module detection algorithm.

    *ref docs: https://github.com/cstoeckert/iterativeWGCNA
    """
    iterative_wgcna_path = Path(__file__).parent.joinpath("wgcna_iterative.R")
    ro.r.source(str(iterative_wgcna_path))

    # extract default arguments from standard WGCNA
    kwargs = {
        key: kwargs.get(key, value)
        for key, value in ro.r("as.list")(
            ro.r("args")(r_wgcna.blockwiseModules)
        ).items()
    }

    # clean incompatible arguments
    kwargs.pop("...")
    kwargs.pop("")
    kwargs.pop("datExpr")
    kwargs.pop("nPreclusteringCenters")
    kwargs.pop("minBranchEigennodeDissim")
    kwargs.pop("minCoreKMESize")

    return ro.r.iterativeWGCNA(datExpr=pd_df_to_rpy2_df(data), **kwargs)


def module_eigengenes(data: pd.DataFrame, module_colors: Iterable[str], **kwargs):
    """
    Calculates module eigengenes (1st principal component) of modules in a given
    single dataset.

    *ref docs:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/moduleEigengenes
    """
    return r_wgcna.moduleEigengenes(
        pd_df_to_rpy2_df(data), ro.StrVector(module_colors), **kwargs
    )


def choose_top_hub_in_each_module(
    data: pd.DataFrame, module_colors: Iterable[str], **kwargs
):
    """
    Calculates module eigengenes (1st principal component) of modules in a given
    single dataset.

    *ref docs:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/chooseTopHubInEachModule
    """
    return r_wgcna.chooseTopHubInEachModule(
        pd_df_to_rpy2_df(data), ro.StrVector(module_colors), **kwargs
    )


def plot_dendro_and_colors(
    network_dendro: Any,
    colors: Iterable[str],
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs
):
    """
    This function plots a hierarchical clustering dendrogram and color annotation(s) of
    objects in the dendrogram underneath.

    *ref docs:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/plotDendroAndColors
    """
    pdf(str(save_path), width=width, height=height)
    r_wgcna.plotDendroAndColors(network_dendro, ro.StrVector(colors), **kwargs)
    dev_off()


def labels2colors(labels: Iterable[str], **kwargs):
    """
    Converts a vector or array of numerical labels into a corresponding vector or array
    of colors corresponding to the labels.

    *ref docs:
        https://www.rdocumentation.org/packages/WGCNA/versions/1.70-3/topics/labels2colors
    """
    return r_wgcna.labels2colors(ro.StrVector(labels), **kwargs)
