"""
    Wrappers for R package GWENA

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

import multiprocessing
from pathlib import Path
from typing import Any

import pandas as pd
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_gwena = importr("GWENA")
r_ggplot2 = importr("ggplot2")


def filter_low_var(data: pd.DataFrame, **kwargs):
    """
    Remove low variating genes based on the percentage given and the type of variation
        specified.

    *ref docs: https://rdrr.io/bioc/GWENA/man/filter_low_var.html
    """
    return r_gwena.filter_low_var(pd_df_to_rpy2_df(data), **kwargs)


def filter_RNA_seq(data: pd.DataFrame, **kwargs):
    """
    Keeping genes with at least one sample with count above min_count in RNA-seq data.

    *ref docs: https://rdrr.io/bioc/GWENA/man/filter_RNA_seq.html
    """
    return r_gwena.filter_RNA_seq(pd_df_to_rpy2_df(data), **kwargs)


def build_net(data: pd.DataFrame, **kwargs):
    """
    Compute the adjacency matrix, then the TOM to build the network. Than detect the
    modules by hierarchical clustering and thresholding.

    *ref docs: https://rdrr.io/bioc/GWENA/man/build_net.html
    """
    # set some defaults
    kwargs["cor_func"] = kwargs.get("cor_func", "bicor")
    kwargs["n_threads"] = kwargs.get("n_threads", multiprocessing.cpu_count())
    return r_gwena.build_net(pd_df_to_rpy2_df(data), **kwargs)


def detect_modules(data: pd.DataFrame, corr_matrix: pd.DataFrame, **kwargs):
    """
    Detect the modules by hierarchical clustering.

    *ref docs: https://rdrr.io/bioc/GWENA/man/detect_modules.html
    """
    return r_gwena.detect_modules(
        pd_df_to_rpy2_df(data), pd_df_to_rpy2_df(corr_matrix), **kwargs
    )


def associate_phenotype(eigengenes: pd.DataFrame, phenotypes: pd.DataFrame, **kwargs):
    """
    Compute the correlation between all modules and the phenotypic variables

    *ref docs: https://rdrr.io/bioc/GWENA/man/associate_phenotype.html
    """
    return r_gwena.associate_phenotype(
        pd_df_to_rpy2_df(eigengenes), pd_df_to_rpy2_df(phenotypes), **kwargs
    )


def plot_modules_phenotype(
    modules_phenotype: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Plot a heatmap of the correlation between all modules and the phenotypic variables
        and the p value associated.

    *ref docs: https://rdrr.io/bioc/GWENA/man/plot_modules_phenotype.html
    """
    plot = r_gwena.plot_modules_phenotype(modules_phenotype, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def bio_enrich(modules: Any, **kwargs):
    """
    Enrich genes list from modules.

    *ref docs: https://rdrr.io/bioc/GWENA/man/bio_enrich.html
    """
    return r_gwena.bio_enrich(modules, **kwargs)


def plot_enrichment(
    enrichment_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Wrapper of the gprofiler2::gostplot function. Adding support of colorblind palet
        and selection of subsets if initial multiple query, and/or sources to plot.

    *ref docs: https://rdrr.io/bioc/GWENA/man/plot_enrichment.html
    """
    plot = r_gwena.plot_enrichment(enrichment_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)
