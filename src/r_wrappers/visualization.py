"""
Wrappers for R visualization packages.

This module provides Python wrappers for various R visualization packages. All functions
have pythonic inputs and outputs and are designed to make visualization of genomic data
easier from Python.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation automatically.

Example:
    R --> data.category
    Python --> data_category

Attributes:
    r_deseq2: The imported DESeq2 R package.
    r_enhanced_volcano: The imported EnhancedVolcano R package.
    r_ggplot2: The imported ggplot2 R package.
    r_pheatmap: The imported pheatmap R package.
    r_venn_diagram: The imported VennDiagram R package.
    r_color_brewer: The imported RColorBrewer R package.
    r_vsn: The imported vsn R package.
    r_stats: The imported stats R package.
    r_graphics: The imported graphics R package.
    r_grid: The imported grid R package.
    pdf: R function to create PDF files.
    dev_off: R function to close the PDF device.
"""

from pathlib import Path
from typing import Any, Dict, Iterable

from rpy2 import robjects as ro
from rpy2.robjects import IntVector, StrVector
from rpy2.robjects.packages import importr

r_deseq2 = importr("DESeq2")
r_enhanced_volcano = importr("EnhancedVolcano")
r_ggplot2 = importr("ggplot2")
r_pheatmap = importr("pheatmap")
r_venn_diagram = importr("VennDiagram")
r_color_brewer = importr("RColorBrewer")
r_vsn = importr("vsn")
r_stats = importr("stats")
r_graphics = importr("graphics")
r_grid = importr("grid")
pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def heatmap_sample_distance(
    sample_dist: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> None:
    """
    Creates a color-coded heatmap of the sample distances between samples.

    Given N samples, the plot shows an NxN grid color-coded by the distance
    between the samples at a given i,j coordinates.

    Args:
        sample_dist: A distance structure, containing all sample distances.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to pheatmap function.

    Note:
        For more details see: https://rdrr.io/cran/pheatmap/man/pheatmap.html
    """
    colors = ro.r("colorRampPalette(rev(brewer.pal(9, 'Blues')))(255)")
    plot = r_pheatmap.pheatmap(
        ro.r("as.matrix")(sample_dist),
        clustering_distance_rows=sample_dist,
        clustering_distance_cols=sample_dist,
        col=colors,
        **kwargs,
    )
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def mean_sd_plot(
    data: ro.methods.RS4, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> None:
    """
    Plots row standard deviations versus row means.

    This plot is useful to visualize the mean-variance relationship in the data and
    to check if variance stabilization was effective.

    Args:
        data: A DESeqDataSet or DESeqTransform object.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to meanSdPlot function.

    Note:
        For more details see: https://rdrr.io/bioc/vsn/man/meanSdPlot.html
    """
    plot = r_vsn.meanSdPlot(ro.r("assay")(data), **kwargs).rx2("gg")
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def pca_plot(
    data: ro.methods.RS4,
    intgroup: Iterable[str],
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Creates a principal component analysis (PCA) plot.

    This function generates a PCA plot from transformed count data, with points colored
    by variables of interest.

    Args:
        data: A DESeqTransform object used to compute and plot PCA.
        intgroup: Interesting groups: a character vector of names in
            colData(x) to use for grouping samples.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to plotPCA function.

    Note:
        For more details see: https://rdrr.io/bioc/DESeq2/man/plotPCA.html
    """
    plot = r_deseq2.plotPCA_DESeqTransform(data, intgroup=StrVector(intgroup), **kwargs)
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def mds_plot(
    sample_dist: Any,
    data: Any,
    color: str,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Creates a multidimensional scaling (MDS) plot.

    Classical multidimensional scaling (MDS) of a data matrix, also known as principal
    coordinates analysis (Gower, 1966). The plot shows the samples in 2D space based
    on their distances.

    Args:
        sample_dist: A distance structure containing all sample distances.
        data: The data object the sample_dist matrix comes from (e.g., DESeqDataSet).
        color: Factor that defines the color groups (each factor level will have a
            different color).
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to qplot function.

    Note:
        For more details see: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/cmdscale
    """
    # 1. Get MDS matrix
    mds = ro.r("data.frame")(r_stats.cmdscale(ro.r("as.matrix")(sample_dist)))
    mds = ro.r("cbind")(mds, ro.r("as.data.frame")(ro.r("colData")(data)))

    # 2. Plot MDS and save
    plot = r_ggplot2.qplot(
        ro.r("sym")("X1"),
        ro.r("sym")("X2"),
        data=mds,
        color=ro.r("sym")(color),
        **kwargs,
    )
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def gene_counts(
    data: ro.methods.RS4,
    gene: str,
    intgroup: Iterable[str],
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Plots gene counts for one gene across sample groups.

    This function generates a plot showing the normalized counts for a single gene
    across different sample groups.

    Args:
        data: A DESeqDataSet object containing count data.
        gene: The gene identifier for which counts amongst samples will be shown.
        intgroup: Interesting groups: a character vector of names in
            colData(x) to use for grouping.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to plotCounts function.

    References:
        https://rdrr.io/bioc/DESeq2/man/plotCounts.html
    """
    pdf(str(save_path), width=width, height=height)
    r_deseq2.plotCounts(data, gene, StrVector(intgroup), **kwargs)
    dev_off()


def ma_plot(
    deseq_result: ro.methods.RS4,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Creates an MA plot of DESeq2 results.

    An MA plot shows the log2 fold changes (on the y-axis) versus the mean of
    normalized counts (on the x-axis). It's useful for visualizing the relationship
    between expression change and expression magnitude.

    Args:
        deseq_result: A DESeqResults object, ideally after lfcShrink.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to plotMA function.

    References:
        - https://rdrr.io/bioc/DESeq2/man/plotMA.html
        - https://rdrr.io/bioc/DESeq2/man/lfcShrink.html
    """
    # 1. MA plot
    pdf(str(save_path), width=width, height=height)
    r_deseq2.plotMA_DESeqResults(deseq_result, ylim=IntVector([-5, 5]), **kwargs)
    r_graphics.grid()
    dev_off()


def volcano_plot(
    data: ro.DataFrame,
    lab: str,
    x: str,
    y: str,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Creates an enhanced volcano plot of differential expression results.

    A volcano plot displays statistical significance versus magnitude of change.
    The EnhancedVolcano package provides publication-ready volcano plots with
    customizable features.

    Args:
        data: A data frame of test statistics. Requires at least:
            - A column for variable names (can be rownames)
            - A column for log2 fold changes
            - A column for nominal or adjusted p-values
        lab: Column name in data containing variable names (can be "rownames").
        x: Column name in data containing log2 fold changes.
        y: Column name in data containing nominal or adjusted p-values.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to EnhancedVolcano function.

   References:
        - http://bioconductor.org/packages/release/bioc/vignettes/EnhancedVolcano/inst/doc/EnhancedVolcano.html
        - https://rdrr.io/bioc/EnhancedVolcano/man/EnhancedVolcano.html
    """
    plot = r_enhanced_volcano.EnhancedVolcano(
        toptable=data, lab=lab, x=x, y=y, **kwargs
    )
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def venn_diagram(
    contrasts_degs: Dict[str, Iterable[str]],
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Creates a publication-quality Venn diagram.

    This function takes a dictionary of lists and creates a TIFF or PDF Venn diagram
    showing the intersections between different sets of genes.

    Args:
        contrasts_degs: A mapping of contrast names to lists of DEGs.
            Length must be between 2 and 5.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments to pass to venn.diagram function.

    Raises:
        ValueError: If the length of contrasts_degs is not between 2 and 5.

    References:
        https://rdrr.io/cran/VennDiagram/man/venn.diagram.html
    """
    if not (2 <= len(contrasts_degs) <= 5):
        raise ValueError("The number of sets must be between 2 and 5.")

    contrasts_degs = {
        k: StrVector(filtered_genes) for k, filtered_genes in contrasts_degs.items()
    }

    # 1. Generate Venn Diagram plot
    futile_logger = importr("futile.logger")
    futile_logger.flog_threshold(futile_logger.ERROR, name="VennDiagramLogger")
    plot = r_venn_diagram.venn_diagram(
        list(contrasts_degs.values()),
        ro.NULL,
        category_names=StrVector(list(contrasts_degs.keys())),
        disable_logging=True,
        **kwargs,
    )

    # 2. Draw and save plot
    pdf(str(save_path), width=width, height=height)
    r_grid.grid_draw(plot)
    dev_off()
