"""
    Wrappers for R package enrichplot

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""
from pathlib import Path
from typing import Any

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r_enrichplot = importr("enrichplot")
r_ggplot2 = importr("ggplot2")


def barplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Barplot of enrichResult (ORA).

    *ref docs: https://rdrr.io/bioc/enrichplot/man/barplot.enrichResult.html
    """
    plot = r_enrichplot.barplot_enrichResult(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def dotplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Dotplot for enrichment result (ORA/GSEA)

    *ref docs: https://rdrr.io/bioc/enrichplot/man/dotplot.html
    """
    plot = r_enrichplot.dotplot(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def gene_concept_net(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Gene-Concept Network

    *ref docs: https://rdrr.io/bioc/enrichplot/man/cnetplot.html
    """
    plot = r_enrichplot.cnetplot(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def heatplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Heatmap like plot for functional classification

    *ref docs: https://rdrr.io/bioc/enrichplot/man/heatplot.html
    """
    plot = r_enrichplot.heatplot(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def pairwise_termsim(x: Any, **kwargs):
    """
    Get the similarity matrix.

    *ref docs: https://rdrr.io/bioc/enrichplot/man/pairwise_termsim.html
    """
    return r_enrichplot.pairwise_termsim(x, **kwargs)


def emapplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Enrichment Map for enrichment result of over-representation test or
    gene set enrichment analysis

    *ref docs: https://rdrr.io/bioc/enrichplot/man/emapplot.html
    """
    plot = r_enrichplot.emapplot(pairwise_termsim(enrich_result), **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def upsetplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Upsetplot method generics

    *ref docs: https://rdrr.io/bioc/enrichplot/man/upsetplot-methods.html
    """
    plot = r_enrichplot.upsetplot(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def ridgeplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Ridgeline plot for GSEA result

    *ref docs:
        https://rdrr.io/github/GuangchuangYu/enrichplot/man/ridgeplot.html
    """
    plot = r_enrichplot.ridgeplot(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def gseaplot(
    enrich_result: Any,
    gene_set_id: int,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs
):
    """
    Visualize analyzing result of GSEA

    *ref docs: https://rdrr.io/bioc/enrichplot/man/gseaplot.html
    """
    plot = r_enrichplot.gseaplot(enrich_result, gene_set_id, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def pmcplot(
    enrich_result: Any,
    save_path: Path,
    period: ro.IntVector = ro.IntVector(range(2010, 2021, 1)),
    width: int = 10,
    height: int = 10,
    **kwargs
):
    """
    PubMed Central Trend plot

    *ref docs: https://rdrr.io/bioc/enrichplot/man/pmcplot.html
    """
    plot = r_enrichplot.pmcplot(enrich_result, period, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def goplot(
    enrich_result: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Plot induced GO DAG of significant terms

    *ref docs: https://rdrr.io/bioc/enrichplot/man/goplot.html
    """
    plot = r_enrichplot.goplot(enrich_result, **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)
