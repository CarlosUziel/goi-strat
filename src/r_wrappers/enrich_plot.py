"""
Wrappers for R package enrichplot

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

r_enrichplot = importr("enrichplot")
r_ggplot2 = importr("ggplot2")


def barplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a barplot visualization of enrichment results.

    This function creates a barplot from over-representation analysis (ORA) results,
    showing enriched terms with their statistical significance.

    Args:
        enrich_result: An enrichResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to barplot_enrichResult.
            Common parameters include:
            - showCategory: Number of categories to show (default: 10).
            - color: Color for bars.
            - font.size: Base font size.
            - title: Plot title.
            - x: The value used for plotting (e.g., "Count", "GeneRatio").

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/barplot.enrichResult.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.barplot_enrichResult(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def dotplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a dotplot visualization of enrichment results.

    This function creates a dotplot from enrichment results (ORA or GSEA),
    showing enriched terms with their statistical significance and gene counts.

    Args:
        enrich_result: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to dotplot.
            Common parameters include:
            - showCategory: Number of categories to show (default: 10).
            - color: Color gradient for p-values.
            - size: Size aesthetic for gene counts.
            - font.size: Base font size.
            - title: Plot title.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/dotplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.dotplot(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def gene_concept_net(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a Gene-Concept Network visualization.

    This function creates a network plot showing the relationships between genes
    and enriched functional terms.

    Args:
        enrich_result: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to cnetplot.
            Common parameters include:
            - showCategory: Number of categories to show.
            - categorySize: Size of category nodes.
            - nodeLabel: Whether to show node labels.
            - colorEdge: Whether to color edges.
            - circular: Whether to layout the network in a circular fashion.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/cnetplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.cnetplot(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def heatplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a heatmap-like plot for functional classification.

    This function creates a heatmap showing the relationships between genes
    and enriched functional terms.

    Args:
        enrich_result: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to heatplot.
            Common parameters include:
            - showCategory: Number of categories to show.
            - foldChange: Named numeric vector of fold changes.
            - label_format: Function to format labels.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/heatplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.heatplot(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def pairwise_termsim(x: Any, **kwargs: Any) -> Any:
    """Calculate similarity matrix between terms based on their shared genes.

    This function calculates the similarity between enriched terms based on
    their shared gene members, which is useful for clustering similar terms.

    Args:
        x: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        **kwargs: Additional arguments to pass to pairwise_termsim.
            Common parameters include:
            - method: Similarity measurement method (default: "Jaccard").
            - showCategory: Number of categories to show.

    Returns:
        Any: An object containing the similarity matrix between terms
           and the original enrichment result.

    References:
        https://rdrr.io/bioc/enrichplot/man/pairwise_termsim.html
    """
    with localconverter(ro.default_converter):
        return r_enrichplot.pairwise_termsim(x, **kwargs)


def emapplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create an Enrichment Map to visualize relationships between enriched terms.

    This function creates a network plot showing the relationships/similarities
    between enriched functional terms.

    Args:
        enrich_result: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to emapplot.
            Common parameters include:
            - showCategory: Number of categories to show.
            - layout: The layout function from igraph.
            - pie: Whether to use pie charts for nodes.
            - edge_threshold: Threshold for filtering edges.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/emapplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.emapplot(pairwise_termsim(enrich_result), **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def upsetplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create an UpSet plot showing the overlap of genes in different gene sets.

    This function creates an UpSet plot visualizing the intersections of genes associated
    with different enriched terms.

    Args:
        enrich_result: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to upsetplot.
            Common parameters include:
            - n: Number of categories to show.
            - order_by: How to order the combinations ("freq", "degree").

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/upsetplot-methods.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.upsetplot(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def ridgeplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a ridgeline plot for GSEA results.

    This function creates a ridgeline plot showing the distribution of
    genes in different gene sets across the ranked list.

    Args:
        enrich_result: A gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to ridgeplot.
            Common parameters include:
            - showCategory: Number of categories to show.
            - fill: Color for filling the ridges.
            - core_enrichment: Whether to show only core enriched genes.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/github/GuangchuangYu/enrichplot/man/ridgeplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.ridgeplot(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def gseaplot(
    enrich_result: Any,
    gene_set_id: int,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a GSEA plot for a specific gene set.

    This function creates a visualization of Gene Set Enrichment Analysis results
    for a specific gene set, showing the enrichment score, running score,
    and gene positions.

    Args:
        enrich_result: A gseaResult object from clusterProfiler or similar tools.
        gene_set_id: The ID or index of the gene set to visualize.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to gseaplot.
            Common parameters include:
            - by: Type of plot ("runningScore", "preranked", "all").
            - title: Plot title.
            - color: Color of the plot elements.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/gseaplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.gseaplot(enrich_result, gene_set_id, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def pmcplot(
    enrich_result: Any,
    save_path: Path,
    period: ro.IntVector = ro.IntVector(range(2010, 2021, 1)),
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a PubMed Central trend plot.

    This function creates a plot showing the publication trends of enriched terms
    in the PubMed Central database over time.

    Args:
        enrich_result: An enrichResult or gseaResult object from clusterProfiler or similar tools.
        save_path: Path where the plot will be saved.
        period: Integer vector specifying the years to include, default is 2010-2020.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to pmcplot.
            Common parameters include:
            - top: Number of terms to display.
            - low.color: Color for low values.
            - high.color: Color for high values.

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/pmcplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.pmcplot(enrich_result, period, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)


def goplot(
    enrich_result: Any,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a GO (Gene Ontology) DAG plot of significant terms.

    This function creates a directed acyclic graph (DAG) visualization of
    the Gene Ontology terms and their relationships.

    Args:
        enrich_result: An enrichResult object from clusterProfiler or similar tools
            containing GO enrichment results.
        save_path: Path where the plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to goplot.
            Common parameters include:
            - showCategory: Number of categories to show.
            - color: Color mapping function.
            - layout: Layout method for the graph.
            - geom: Geometry to use ("text" or "label").

    Returns:
        None: The function saves the plot to the specified path but doesn't return anything.

    References:
        https://rdrr.io/bioc/enrichplot/man/goplot.html
    """
    with localconverter(ro.default_converter):
        plot = r_enrichplot.goplot(enrich_result, **kwargs)
        r_ggplot2.ggsave(str(save_path), plot, width=width, height=height, dpi=320)
