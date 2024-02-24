"""
    Wrappers for R visualization tools

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> annot_df.category
        Python --> data_category
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
):
    """
    Color-coded plot of the sample distance between samples. Given N
    number of samples, the plot shows a NxN grid color-coded depending on
    the distance between the samples at a given i,j coordinates.

    *ref docs: https://rdrr.io/cran/pheatmap/man/pheatmap.html

    Args:
        sample_dist: a distance structure, containing all sample
            distances
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    colors = ro.r("colorRampPalette(rev(brewer.pal(9, 'Blues')))(255)")
    plot = r_pheatmap.pheatmap(
        ro.r("as.matrix")(sample_dist),
        clustering_distance_rows=sample_dist,
        clustering_distance_cols=sample_dist,
        col=colors,
        **kwargs
    )
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def mean_sd_plot(
    data: ro.methods.RS4, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Plot row standard deviations versus row means

    *ref docs in https://rdrr.io/bioc/vsn/man/meanSdPlot.html

    Args:
        data: DESeqDataSet or DESeqTransform
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    plot = r_vsn.meanSdPlot(ro.r("assay")(data), **kwargs).rx2("gg")
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def pca_plot(
    data: ro.methods.RS4,
    intgroup: Iterable[str],
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs
):
    """
    Principal component analysis plot.

    *ref docs: https://rdrr.io/bioc/DESeq2/man/plotPCA.html

    Args:
        data: data used to compute and plot PCA
        intgroup: interesting groups: a character vector of names in
            colData(x) to use for grouping
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
    **kwargs
):
    """
        Multidimensional scaling (MDS) plot. Classical multidimensional
        scaling (MDS) of a annot_df matrix. Also known as
            principal coordinates analysis (Gower, 1966).

        Underlying function docs:
            https://www.rdocumentation.org/packages/stats/versions/3.6.2
            /topics/cmdscale

    Args:
         sample_dist: a distance structure, containing all sample
            distances
         data: data the sample_dist_matrix comes from.
         color: factor that defines the color (each factor level will have a
            different color)
         save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
        **kwargs
    )
    r_ggplot2.ggsave(str(save_path), plot, width=width, height=height)


def gene_counts(
    data: ro.methods.RS4,
    gene: str,
    intgroup: Iterable[str],
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs
):
    """
        (full docs in https://rdrr.io/bioc/DESeq2/man/plotCounts.html)
    Args:
        data: a DESeqDataSet object
        gene: gene for which counts amongst samples are shown
        intgroup: interesting groups: a character vector of names in
            colData(x) to use for grouping
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_deseq2.plotCounts(data, gene, StrVector(intgroup), **kwargs)
    dev_off()


def ma_plot(
    deseq_result: ro.methods.RS4,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs
):
    """
    A scatter plot of log2 fold changes (on the y-axis) versus the mean of
    normalized counts (on the x-axis).

    *ref docs in https://rdrr.io/bioc/DESeq2/man/plotMA.html)

    Makes use of "lfcShrink"
        *ref docs in https://rdrr.io/bioc/DESeq2/man/lfcShrink.html

    Args:
        deseq_result: a DESeqResults object, after lfcShrink.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
    **kwargs
):
    """
        (reference docs in
            http://bioconductor.org/packages/release/bioc/vignettes
            /EnhancedVolcano/inst/doc/EnhancedVolcano.html)
        (full docs in https://rdrr.io/bioc/EnhancedVolcano/man
        /EnhancedVolcano.html)


    Args:
        data: A data-frame of test statistics (if not, a data frame,
            annotate_deseq_resultan attempt will be made to convert it to one).
            Requires at least the following: column for variable names (can
            be rownames); a column for log2 fold
            changes; a column for nominal or adjusted p-value.
        lab: A column name in annot_df containing variable names. Can be
            rownames (toptable).
        x: A column name in toptable containing log2 fold changes.
        y: A column name in toptable containing nominal or adjusted p-values.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
    **kwargs
):
    """
    This function takes a list and creates a publication-quality TIFF Venn
        Diagram.

    *ref docs in https://rdrr.io/cran/VennDiagram/man/venn.diagram.html

    Args:
        contrasts_degs: A mapping of contrast and DEGs. Length must be between 2 and 5.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure

    """
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
        **kwargs
    )

    # 2. Draw and save plot
    pdf(str(save_path), width=width, height=height)
    r_grid.grid_draw(plot)
    dev_off()
