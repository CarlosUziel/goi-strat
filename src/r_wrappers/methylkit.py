"""
Wrappers for R package methylKit.

This module provides Python wrappers for the R methylKit package, which is a suite of tools
for the analysis of DNA methylation data. All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation automatically.

Example:
    R --> data.category
    Python --> data_category

Attributes:
    r_source: The imported methylKit R package.
    r_ggplot: The imported ggplot2 R package.
    r_granges: The imported GenomicRanges R package.
    r_graphics: The imported graphics R package.
    r_scatter3d: The imported scatterplot3d R package.
    pdf: R function to create PDF files.
    dev_off: R function to close the PDF device.
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rpy2 import robjects as ro
from rpy2.robjects import FloatVector, IntVector, StrVector
from rpy2.robjects.packages import importr

from r_wrappers.annotatr import annotate_regions
from r_wrappers.utils import (
    homogeinize_seqlevels_style,
    make_granges_from_dataframe,
    pd_df_to_rpy2_df,
    rpy2_df_to_pd_df,
)

r_source = importr("methylKit")
r_ggplot = importr("ggplot2")
r_granges = importr("GenomicRanges")
r_graphics = importr("graphics")
r_scatter3d = importr("scatterplot3d")

pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def process_bismark_aln(
    files: Iterable[Path], sample_ids: Iterable[str], assembly: str = "mm10", **kwargs
) -> Any:
    """
    Process Bismark alignment files to extract methylation information.

    This function calls methylation percentage per base from sorted Bismark SAM or BAM
    files and reads methylation information as methylKit objects. Bismark is a
    popular aligner for high-throughput bisulfite sequencing experiments that
    outputs its results in SAM format by default, which can be converted to BAM.
    Bismark SAM/BAM format contains aligner-specific tags which are necessary
    for methylation percentage calling.

    Args:
        files: Iterable of Path objects pointing to SAM or BAM files containing
            aligned bisulfite-treated reads.
        sample_ids: Iterable of strings with the ID for each sample, in the same order
            as files. Must have the same length as files.
        assembly: String that determines the genome assembly (e.g., mm9, hg18).
            This is a reference string for bookkeeping and can be any string, but
            should be consistent when using multiple files from the same assembly.
        **kwargs: Additional arguments passed to methylKit's processBismarkAln function.

    Returns:
        Any: A methylRaw, methylRawList, methylRawDB, or methylRawListDB object
            containing methylation information extracted from the input files.

    Raises:
        ValueError: If files and sample_ids have different lengths or if a file
            does not exist or is a directory.

    Note:
        SAM/BAM files from aligners other than Bismark will not work with this function.

    Refrences:
        https://rdrr.io/bioc/methylKit/man/processBismarkAln-methods.html
    """
    # 0. Check arguments
    if len(files) != len(sample_ids):
        raise ValueError("Files and samples must have the same length!")
    for f in files:
        if not f.exists() or f.is_dir():
            raise ValueError("File does not exist or is a directory")

    # 1. Run
    return r_source.processBismarkAln(
        location=[str(p) for p in files],
        sample_id=sample_ids,
        assembly=assembly,
        **kwargs,
    )


def meth_read(
    files: Iterable[Path], sample_ids: Iterable[str], assembly: str = "mm10", **kwargs
) -> Any:
    """
    Read methylation information from files into methylKit objects.

    This function reads a list of files containing methylation information for
    bases/regions in the genome and creates a methylRawList or methylRaw object.
    The information can also be stored as a flat file database by creating a
    methylRawListDB or methylRawDB object.

    Args:
        files: Iterable of Path objects pointing to methylation data files.
        sample_ids: Iterable of strings with the ID for each sample, in the same order
            as files. Must have the same length as files.
        assembly: String that determines the genome assembly (e.g., mm9, hg18).
            This is a reference string for bookkeeping and can be any string, but
            should be consistent when using multiple files from the same assembly.
        **kwargs: Additional arguments passed to methylKit's methRead function,
            such as pipeline, header, context, resolution, treatment, etc.

    Returns:
        Any: A methylRaw, methylRawList, methylRawDB, or methylRawListDB object
            containing methylation information from the input files.

    Raises:
        ValueError: If files and sample_ids have different lengths or if a file
            does not exist or is a directory.

    References:
        https://rdrr.io/bioc/methylKit/man/methRead-methods.html
    """
    # 0. Check arguments
    if len(files) != len(sample_ids):
        raise ValueError("Files and samples must have the same length!")
    for f in files:
        if not f.exists() or f.is_dir():
            raise ValueError("File does not exist or is a directory")

    # 1. Run
    return r_source.methRead(
        location=list(map(str, files)),
        sample_id=sample_ids,
        assembly=assembly,
        **kwargs,
    )


def filter_by_coverage(methyl_obj: Any, **kwargs) -> Any:
    """
    Filter methylation objects based on read coverage thresholds.

    This function filters methylRaw, methylRawDB, methylRawList and methylRawListDB
    objects based on coverage criteria. You can filter using both lower and upper read
    coverage thresholds. Upper read coverage cutoffs help eliminate PCR duplications
    and other artifacts, while lower read cutoffs improve statistical power.

    Args:
        methyl_obj: A methylRaw, methylRawDB, methylRawList or methylRawListDB object
            containing methylation data.
        **kwargs: Additional arguments passed to methylKit's filterByCoverage function:
            - lo_count: Lower count threshold for coverage, default 10
            - lo_perc: Lower percentage threshold for coverage, default NULL
            - hi_count: Upper count threshold for coverage, default NULL
            - hi_perc: Upper percentage threshold for coverage, default 99.9

    Returns:
        Any: A filtered version of the input object (methylRaw, methylRawDB,
            methylRawList or methylRawListDB) containing only positions that pass
            the coverage thresholds.

    References:
        https://rdrr.io/bioc/methylKit/man/filterByCoverage-methods.html
    """
    return r_source.filterByCoverage(methylObj=methyl_obj, **kwargs)


def unite(methyl_obj: Any, **kwargs) -> Any:
    """
    Unite methylRawList or methylRawListDB objects by common genomic locations.

    This function merges methylRawList and methylRawListDB objects, retaining only bases or
    regions with coverage from all samples. The resulting object is either a methylBase
    or methylBaseDB object, depending on the input.

    Args:
        methyl_obj: A methylRawList or methylRawListDB object to be merged by common
            genomic locations covered by reads.
        **kwargs: Additional arguments passed to methylKit's unite function:
            - destrand: If TRUE, merge reads on both strands
            - min.per.group: Minimum number of samples per group to cover a region
            - conv.rate.threshold: Conversion rate threshold, default 0 (no filtering)
            - coverage.threshold: Minimum coverage threshold, default 0 (no filtering)

    Returns:
        Any: A methylBase or methylBaseDB object containing the merged methylation data
            for positions covered in all samples.

    References:
        https://rdrr.io/bioc/methylKit/man/unite-methods.html
    """
    return r_source.unite(object=methyl_obj, **kwargs)


def reorganize(
    methyl_obj: Any, sample_ids: Iterable[str], treatment: IntVector, **kwargs
) -> Any:
    """
    Reorganize a methylation object by selecting or reordering samples.

    This function creates a new methylRawList, methylRawListDB, methylBase or
    methylBaseDB object by selecting a subset of samples from the input object.
    It can be used to partition a large methylation object into smaller objects
    based on sample IDs, or to reorder samples and/or assign a new treatment vector.

    Args:
        methyl_obj: A methylRawList, methylRawListDB, methylBase or methylBaseDB object
            containing the original methylation data.
        sample_ids: Iterable of strings with sample IDs to include in the new object.
            Order is important and should correspond to the treatment vector.
            Must be a subset or reordered version of sample IDs in the input object.
        treatment: An IntVector of 0s and 1s indicating the treatment group of each
            sample. Should have the same length as sample_ids.
        **kwargs: Additional arguments passed to methylKit's reorganize function.

    Returns:
        Any: A methylRawList, methylRawListDB, methylBase or methylBaseDB object
            (same type as input) containing only the selected samples in the
            specified order with the new treatment assignments.

    References:
        https://rdrr.io/bioc/methylKit/man/reorganize-methods.html
    """
    return r_source.reorganize(
        methylObj=methyl_obj, sample_ids=sample_ids, treatment=treatment, **kwargs
    )


def calculate_diff_meth(methyl_obj: Any, **kwargs) -> Any:
    """
    Calculate differential methylation statistics between two groups of samples.

    This function performs statistical tests to identify differentially methylated
    regions or bases between two groups of samples. It uses either logistic regression
    (default) or Fisher's Exact test to calculate differential methylation.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing the methylation
            data for all samples to be compared.
        **kwargs: Additional arguments passed to methylKit's calculateDiffMeth function:
            - slim: Use SLIM method for p-value adjustment (default TRUE)
            - weighted.mean: Use coverage-weighted means (default FALSE)
            - test: Statistical test, either "Chisq" or "F" (default "F")
            - overdispersion: String or character, "MN" or "shrinkMN"
            - effect.size: Minimum absolute value of methylation difference (default 10)

    Returns:
        Any: A methylDiff or methylDiffDB object containing differential methylation
            statistics and genomic locations for all analyzed bases or regions.

    References:
        https://rdrr.io/bioc/methylKit/man/calculateDiffMeth-methods.html
    """
    return r_source.calculateDiffMeth(methyl_obj, **kwargs)


def get_methylation_stats(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> Any:
    """
    Calculate and plot methylation statistics from methylation data.

    This function returns basic statistics about methylation percentages and can optionally
    generate a histogram plot of methylation distribution.

    Args:
        methyl_obj: A methylRaw or methylRawDB object containing methylation data.
        save_path: Path where to save the generated plot if plot=True.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments passed to methylKit's getMethylationStats function:
            - plot: Boolean indicating whether to plot the histogram (default FALSE)
            - both.strands: Whether to process both strands together (default FALSE)
            - labels: Labels for plot

    Returns:
        Any: A vector of summary statistics including min, max, mean, standard deviation,
            and various percentiles of methylation values.

    References:
        https://rdrr.io/bioc/methylKit/man/getMethylationStats-methods.html
    """
    if kwargs.get("plot", False):
        pdf(str(save_path), width=width, height=height)

    res = r_source.getMethylationStats(methyl_obj, **kwargs)

    if kwargs.get("plot", False):
        dev_off()

    return res


def get_coverage_stats(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> Any:
    """
    Calculate and plot read coverage statistics.

    This function returns basic statistics about read coverage per base and can optionally
    generate a histogram plot of coverage distribution.

    Args:
        methyl_obj: A methylRaw or methylRawDB object containing methylation data.
        save_path: Path where to save the generated plot if plot=True.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments passed to methylKit's getCoverageStats function:
            - plot: Boolean indicating whether to plot the histogram (default FALSE)
            - both.strands: Whether to process both strands together (default FALSE)
            - labels: Labels for plot

    Returns:
        Any: A vector of summary statistics including min, max, mean, standard deviation,
            and various percentiles of coverage values.

    References:
        https://rdrr.io/bioc/methylKit/man/getCoverageStats-methods.html
    """
    if kwargs.get("plot", False):
        pdf(str(save_path), width=width, height=height)

    res = r_source.getCoverageStats(methyl_obj, **kwargs)

    if kwargs.get("plot", False):
        dev_off()

    return res


def get_correlation(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> Any:
    """
    Calculate correlation between samples and optionally plot scatterplots.

    This function returns a matrix of correlation coefficients between samples and can
    generate scatterplots showing the relationship between samples. The scatterplots include
    fitted lines using linear regression and LOWESS for polynomial regression.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data
            from multiple samples.
        save_path: Path where to save the generated plot if plot=True.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments passed to methylKit's getCorrelation function:
            - plot: Boolean indicating whether to plot scatterplots (default FALSE)
            - method: Correlation method (e.g., "pearson", "spearman")
            - new: Whether to open a new plotting window for each pair (default TRUE)
            - noise.filter: Adds random jitter to avoid overplotting (default TRUE)

    Returns:
        Any: A correlation matrix object showing pairwise correlations between samples.

    References:
        https://rdrr.io/bioc/methylKit/man/getCorrelation-methods.html
    """
    if kwargs.get("plot", False):
        pdf(str(save_path), width=width, height=height)

    res = r_source.getCorrelation(methyl_obj, **kwargs)

    if kwargs.get("plot", False):
        dev_off()

    return res


def cluster_samples(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 5, **kwargs
) -> Any:
    """
    Perform hierarchical clustering on samples based on methylation data.

    This function clusters samples using the hclust function and various distance metrics
    derived from percent methylation per base or region for each sample. It can generate
    a dendrogram showing the hierarchical relationship between samples.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data
            from multiple samples.
        save_path: Path where to save the generated plot if plot=True.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments passed to methylKit's clusterSamples function:
            - plot: Boolean indicating whether to plot the dendrogram (default FALSE)
            - dist: Distance metric to use (default "correlation")
            - method: Clustering method (default "ward.D")
            - sd.filter: Whether to filter by standard deviation (default TRUE)
            - sd.threshold: Standard deviation threshold for filtering (default 0.5)

    Returns:
        Any: A hierarchical cluster tree object containing clustering information.

    References:
        https://rdrr.io/bioc/methylKit/man/clusterSamples-methods.html
    """
    if kwargs.get("plot", False):
        pdf(str(save_path), width=width, height=height)

    res = r_source.clusterSamples(methyl_obj, **kwargs)

    if kwargs.get("plot", False):
        dev_off()

    return res


def pca_samples(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> Any:
    """
    Perform principal component analysis (PCA) on methylation data and plot results.

    This function performs PCA using the prcomp function with percent methylation matrix
    as input. It generates a plot showing sample relationships in the principal component
    space.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data
            from multiple samples.
        save_path: Path where to save the generated PCA plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments passed to methylKit's PCASamples function:
            - screeplot: Whether to produce a scree plot instead (default FALSE)
            - obj.return: Whether to return the PCA object (default TRUE)
            - comp: Which components to plot (default c(1,2))
            - sd.filter: Whether to filter by standard deviation (default TRUE)
            - sd.threshold: Standard deviation threshold for filtering (default 0.5)
            - adj.lim: Whether to adjust plot limits (default TRUE)

    Returns:
        Any: A summary of principal component analysis containing loadings,
            variance explained, etc.

    References:
        https://rdrr.io/bioc/methylKit/man/PCASamples-methods.html
    """
    pdf(str(save_path), width=width, height=height)

    res = r_source.PCASamples(methyl_obj, **kwargs)

    dev_off()

    return res


def get_dd(methyl_obj: Any, condition_sample: Dict[str, Iterable[str]]) -> Any:
    """
    Format methylation data for plotting.

    Provides an intermediate dataframe needed for plotting methylation data by
    samples and conditions.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        condition_sample: Dictionary mapping condition names to lists of sample IDs
            that belong to each condition.

    Returns:
        Any: An R dataframe containing formatted methylation percentages with columns
            for methylation levels (mCpG), sample IDs, and condition labels.
    """
    df = rpy2_df_to_pd_df(r_source.getData(methyl_obj))
    sample_ids = ro.r("attr")(methyl_obj, "sample.ids")

    df_dict = {"mCpG": [], "sample": [], "condition": []}
    for i, sample_id in enumerate(sample_ids, 1):
        df_dict["mCpG"] += list(df[f"numCs{i}"] / df[f"coverage{i}"] * 100)
        df_dict["sample"] += [sample_id] * len(df)
        df_dict["condition"] += [
            k for k, v in condition_sample.items() if sample_id in v
        ] * len(df)

    return pd_df_to_rpy2_df(pd.DataFrame(df_dict))


def violin_plot(
    methyl_obj: Any,
    condition_sample: Dict[str, Iterable[str]],
    save_path: Path,
    width: int = 10,
    height: int = 10,
) -> None:
    """
    Create a violin plot showing global distributions of methylation levels.

    This function generates a violin plot to visualize the distribution of methylation
    percentages across samples, grouped by experimental conditions.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        condition_sample: Dictionary mapping condition names to lists of sample IDs
            that belong to each condition.
        save_path: Path where to save the generated violin plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plot to the specified path.
    """
    # 0. Get DD dataframe
    dd = get_dd(methyl_obj, condition_sample)

    # 1. Data Summary function
    data_summary = ro.r(
        """
        library(R.filesets)
        data_summary <- function(x) {
            m <- mean(x)
            ymin <- m-sd(x)
            ymax <- m+sd(x)
            return(c(y=m,ymin=ymin,ymax=ymax))
        }
        """
    )

    # 2. Create figure plot
    f = ro.r(
        """
        library(ggplot2)
        f <- function(DD, data_summary) {
            return(
                ggplot(DD, aes(x=sample, y=mCpG, fill=condition)) +
                geom_violin(trim=FALSE) +
                scale_fill_manual(values=c("#a6cee3","#1f78b4","#b2df8a","#33a02c")) +
                coord_flip() +
                labs(x="Sample ID", y = "% mCpG") +
                stat_summary(fun.data=data_summary) +
                geom_boxplot(width=0.1)
            )
        }
        """
    )
    plot = f(dd, data_summary)

    # 3. Save figure
    r_ggplot.ggsave(str(save_path), plot, width=width, height=height)


def get_cpg_neighbours(methyl_obj: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify neighboring methylated CpG sites in genomic data.

    This function identifies methylated sites that are neighbors (positions separated by
    exactly 1 base pair). It categorizes the first neighbor (advancing from left to right
    along the chromosome) as the 'forward' neighbor and the second as the 'reverse' neighbor.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - First element: DataFrame of forward neighbor CpG sites
            - Second element: DataFrame of reverse neighbor CpG sites
    """
    # 0. MehtylKit object to dataframe
    df = rpy2_df_to_pd_df(r_source.getData(methyl_obj))

    # 1. Get neighbours per chromosome
    cpg_neighbours_forward = pd.DataFrame(columns=df.columns)
    cpg_neighbours_reverse = pd.DataFrame(columns=df.columns)
    for chr_id in df["chr"].unique():
        # 1.1. Compute distances between immediate neighbours
        chr_cpgs = df.loc[df["chr"] == chr_id, "start"]
        dists = pd.DataFrame(
            chr_cpgs[1:].values - chr_cpgs[:-1].values,
            chr_cpgs[:-1].index,
            ["(i+1)-(i)"],
        )

        # 1.2. Select CpG sites with neighbours at distance 1 in the chromosome
        indices = dists.index[dists["(i+1)-(i)"] == 1]
        indices_reverse = [
            str(int(i) + 1) for i in dists.index[dists["(i+1)-(i)"] == 1]
        ]
        cpg_neighbours_forward = pd.concat([cpg_neighbours_forward, df.loc[indices]])
        cpg_neighbours_reverse = pd.concat(
            [cpg_neighbours_reverse, df.loc[indices_reverse]]
        )

    return cpg_neighbours_forward, cpg_neighbours_reverse


def global_strand_specific_scatter(
    methyl_obj: Any, save_path_prefix: Path, width: int = 10, height: int = 10
) -> None:
    """
    Create strand-specific methylation scatter plots for each sample.

    This function generates plots showing the relationship between methylation on the
    forward strand versus the reverse strand for CpG sites that are covered on both
    strands with coverage >10 in all samples.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<sample_id>.pdf'.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plots to the specified paths.
    """
    # 0. Get CpG neighbours
    cpg_neighbours_forward, cpg_neighbours_reverse = get_cpg_neighbours(methyl_obj)

    # 1. Generate scatter plots for each sample
    sample_ids = ro.r("attr")(methyl_obj, "sample.ids")
    for i, sample_id in enumerate(sample_ids, 1):
        pdf(str(save_path_prefix) + f"_{sample_id}.pdf", width=width, height=height)

        r_graphics.smoothScatter(
            list(
                cpg_neighbours_forward[f"numCs{i}"]
                / cpg_neighbours_forward[f"coverage{i}"]
            ),
            list(
                cpg_neighbours_reverse[f"numCs{i}"]
                / cpg_neighbours_reverse[f"coverage{i}"]
            ),
            xlab="mCpG, forward strand",
            ylab="mCpG, reverse strand",
            main=sample_id,
            colramp=ro.r("colorRampPalette(topo.colors(100))"),
            cex_lab=2,
            cex_axis=2,
            cex_main=2,
        )

        dev_off()


def get_methyl_diff(methyl_obj: Any, **kwargs) -> Any:
    """
    Extract differentially methylated regions that meet specified thresholds.

    This function subsets a methylDiff or methylDiffDB object to extract differentially
    methylated bases or regions that satisfy filtering thresholds (e.g., minimum difference
    in methylation percentage, q-value cutoffs).

    Args:
        methyl_obj: A methylDiff or methylDiffDB object containing differential
            methylation statistics.
        **kwargs: Additional arguments passed to methylKit's getMethylDiff function:
            - difference: Minimum absolute value of methylation difference (default 25)
            - qvalue: Q-value threshold (default 0.01)
            - type: Type of difference to return: "hyper", "hypo", or "all" (default)

    Returns:
        Any: A methylDiff or methylDiffDB object containing only the differentially
            methylated locations that satisfy the specified criteria.

    References:
        https://rdrr.io/bioc/methylKit/man/getMethylDiff-methods.html
    """
    return r_source.getMethylDiff(methyl_obj, **kwargs)


def methyl_diff_barplot(
    methyl_diff_df: ro.vectors.DataFrame,
    save_path: Path,
    width: int = 20,
    height: int = 10,
    **kwargs,
) -> None:
    """
    Create a barplot showing the number of differentially methylated CpG sites.

    This function generates a barplot visualization of the number of differentially
    methylated CpG sites across different comparisons.

    Args:
        methyl_diff_df: R DataFrame containing the counts of differentially methylated
            CpG sites per comparison.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.
        **kwargs: Additional arguments passed to the R barplot function.

    Returns:
        None: The function saves the generated plot to the specified path.
    """
    pdf(str(save_path), width=width, height=height)
    r_graphics.barplot(methyl_diff_df, **kwargs)
    dev_off()


def methyl_diff_density_plot(
    methyl_diffs: Dict[str, Any], save_path: Path, width: int = 20, height: int = 10
) -> None:
    """
    Create a density plot showing distributions of methylation changes.

    This function generates a plot with the distribution curves of methylation differences
    for multiple comparisons overlaid on the same graph.

    Args:
        methyl_diffs: Dictionary mapping comparison names (as strings) to their
            corresponding methylDiff objects.
        save_path: Path where to save the generated plot.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plot to the specified path.
    """
    # TODO: for some reason, colors has the wrong length the first time it is called,
    # but correct the second time
    # 0. Compute histogram metrics for each methylation difference
    meth_hists = {
        comp: r_graphics.hist(
            methyl_diff.rx2("meth.diff"), ro.r("seq(-100,100)"), plot=False
        )
        for comp, methyl_diff in methyl_diffs.items()
    }
    colors = ro.r(f"palette(rainbow({len(meth_hists)}))")

    # 1. Plot hist counts
    pdf(str(save_path), width=width, height=height)
    for i, meth_hist in enumerate(meth_hists.values()):
        if i == 0:
            ro.r("plot")(
                meth_hist.rx2("mids"),
                meth_hist.rx2("counts"),
                t="l",
                xlim=IntVector([-40, 40]),
                ylim=IntVector([0, 20000]),
                ylab="number",
                xlab="methylation change",
                lwd=5,
                lty=1,
                col=colors[i],
            )
        else:
            r_graphics.points(
                meth_hist.rx2("mids"),
                meth_hist.rx2("counts"),
                t="l",
                lwd=3,
                lty=2,
                col=colors[i],
            )

    r_graphics.grid()
    r_graphics.abline(v=0)
    r_graphics.legend(
        "topright",
        col=colors,
        lty=IntVector([1, 2, 3]),
        lwd=2,
        legend=StrVector(list(meth_hists.keys())),
    )
    dev_off()


def percentage_methylation(methyl_obj: Any, **kwargs) -> Any:
    """
    Calculate percent methylation scores for each base or region.

    This function extracts percent methylation values (0-100%) from a methylBase
    or methylBaseDB object. The percentages represent the ratio of methylated reads
    to the total coverage at each CpG site.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        **kwargs: Additional arguments passed to methylKit's percMethylation function:
            - rowids: Whether to include row IDs (default FALSE)
            - save.txt: Whether to save output to a text file (default FALSE)

    Returns:
        Any: A data frame containing percent methylation values for each
            CpG site (rows) across all samples (columns).

    Refrences:
        https://rdrr.io/bioc/methylKit/man/percMethylation-methods.html
    """
    return ro.r("as.data.frame")(r_source.percMethylation(methyl_obj, **kwargs))


def methylation_change_wrt_condition(
    methyl_obj: Any,
    methyl_diffs: Dict[str, Any],
    condition_sample: Dict[str, Iterable[str]],
    wrt_condition: str,
    save_path_prefix: Path,
    width: int = 20,
    height: int = 20,
) -> None:
    """
    Analyze how methylation changes relate to baseline methylation levels.

    This function examines whether methylation changes depend on the baseline methylation
    level in a reference condition. It generates scatter plots showing the relationship
    between baseline methylation levels (x-axis) and methylation changes (y-axis),
    and includes a fitted curve to visualize the trend.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        methyl_diffs: Dictionary mapping comparison names (as strings) to their
            corresponding methylDiff objects.
        condition_sample: Dictionary mapping condition names to lists of sample IDs
            that belong to each condition.
        wrt_condition: Reference condition name with respect to which the methylation
            changes are compared.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<test>_vs_<control>.pdf'.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plots to the specified paths.
    """
    # 0. Setup ann_df inputs for plotting
    p_meth = pd.DataFrame(rpy2_df_to_pd_df(percentage_methylation(methyl_obj)) / 100)
    p_meth_condition = {
        condition: p_meth[samples].median(axis=1)
        for condition, samples in condition_sample.items()
    }

    # 1. Define plotting function
    plotter = ro.r(
        """
        library(earth)
        library(psych)

        plotMetDiffMeth <- function(CH){
            smoothScatter(
                CH[[1]],CH[[2]],
                xlab=names(CH)[1],
                ylab='methylation changes',
                xlim=c(0,1),
                ylim=c(-100,100),
                main=names(CH)[2],
                colramp=colorRampPalette(topo.colors(100)),
                cex.lab=4,
                cex.axis=2,
                cex.main=4,
                cex=4
            )
            abline(h=0)
            text(.2,60,paste("corr=",round(cor(CH[[1]],CH[[2]]),4)),cex=4)
            form = as.formula(paste(names(CH)[2],names(CH)[1],sep='~'))
            mars <- earth(
                formula=form,
                ann_df=CH,
                pmethod="backward",
                nprune=20,
                nfold=10
            );
            cuts <- mars$cuts[mars$selected.terms, ];
            xf <- predict(mars, seq(0,1,length.out=100));
            points(seq(0,1,length.out=100 ),xf,t='l',col='red',lwd=3);
        }
        """
    )

    # 2. Plot figures for each comparison that has 'wrt_condition' as control
    for comparison, methyl_diff in methyl_diffs.items():
        # 2.1. Check if 'wrt_condition' is the control
        test, control = comparison.replace(" ", "").split("vs")
        if wrt_condition != control:
            continue

        # 2.2. Build ann_df frame for visualization
        df = pd.DataFrame(
            {
                wrt_condition: p_meth_condition[wrt_condition],
                f"{test}.vs.{control}": methyl_diff.rx2("meth.diff"),
            }
        )
        df = pd_df_to_rpy2_df(df)

        # 2.3. Plot and save figure
        pdf(
            str(save_path_prefix) + f"_{test}_vs_{control}.pdf",
            width=width,
            height=height,
        )
        plotter(df)
        dev_off()


def meth_levels_anno_boxplot(
    methyl_obj: Any,
    cpgs_ann: Any,
    condition_sample: Dict[str, Iterable[str]],
    save_path_prefix: Path,
    width: int = 10,
    height: int = 10,
) -> None:
    """
    Create boxplots showing methylation levels by CpG annotation categories.

    This function generates boxplots that display methylation levels across different
    genomic annotation categories (e.g., CpG islands, shores, shelves) for each
    experimental condition. This visualization helps identify region-specific
    methylation patterns.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        cpgs_ann: A GRanges object containing genome annotations for CpG sites.
        condition_sample: Dictionary mapping condition names to lists of sample IDs
            that belong to each condition.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<annotation_type>.pdf'.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plots to the specified paths.
    """
    # 0. Get needed inputs
    regions = homogeinize_seqlevels_style(
        make_granges_from_dataframe(
            ro.r("data.frame")(methyl_obj), keep_extra_columns=True
        ),
        cpgs_ann,
    )
    meth_ann = annotate_regions(
        regions=regions,
        annotations=cpgs_ann,
        ignore_strand=True,
        quiet=False,
    )
    meth_ann_df = rpy2_df_to_pd_df(ro.r("as.data.frame")(meth_ann))

    p_meth = rpy2_df_to_pd_df(percentage_methylation(methyl_obj)) / 100
    p_meth_condition = {
        condition: p_meth[samples].median(axis=1)
        for condition, samples in condition_sample.items()
    }

    # 1. Create a figure for each annotation type
    for anno_type in meth_ann_df["annot.type"].unique():
        # 1.1 Filter annotations by annotation type and get indices
        inds = list(meth_ann_df.index[meth_ann_df["annot.type"] == anno_type])

        # 1.2. Build ann_df frame for visualization
        df = pd_df_to_rpy2_df(
            pd.DataFrame({c: v[inds] for c, v in p_meth_condition.items()})
        )

        # 1.3. Plot and save figure
        pdf(str(save_path_prefix) + f"_{anno_type}.pdf", width=width, height=height)
        r_graphics.boxplot(
            df,
            ylab="methylation",
            main=anno_type.replace("_", " ") + f" (N = {len(inds)})",
            cex_main=4,
            cex_axis=4,
        )
        r_graphics.grid()
        dev_off()


def meth_changes_anno_boxplot(
    methyl_obj: Any,
    cpgs_ann: Any,
    wrt_condition: str,
    methyl_diffs: Dict[str, Any],
    save_path_prefix: Path,
    width: int = 10,
    height: int = 10,
) -> None:
    """
    Create boxplots showing methylation changes by CpG annotation categories.

    This function generates boxplots that display methylation changes (differences)
    across different genomic annotation categories (e.g., CpG islands, shores, shelves)
    for each comparison relative to a reference condition. This visualization helps
    identify region-specific differential methylation patterns.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        cpgs_ann: A GRanges object containing genome annotations for CpG sites.
        wrt_condition: Reference condition name with respect to which the methylation
            changes are measured.
        methyl_diffs: Dictionary mapping comparison names (as strings) to their
            corresponding methylDiff objects.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<annotation_type>.pdf'.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plots to the specified paths.
    """
    # 0. Get needed inputs
    regions = homogeinize_seqlevels_style(
        make_granges_from_dataframe(
            ro.r("data.frame")(methyl_obj), keep_extra_columns=True
        ),
        cpgs_ann,
    )
    meth_ann = annotate_regions(
        regions=regions,
        annotations=cpgs_ann,
        ignore_strand=True,
        quiet=False,
    )
    meth_ann_df = rpy2_df_to_pd_df(ro.r("as.data.frame")(meth_ann))

    # 1. Create a figure for each annotation type
    for anno_type in meth_ann_df["annot.type"].unique():
        # 1.1 Filter annotations by annotation type and get indices
        inds = list(meth_ann_df.index[meth_ann_df["annot.type"] == anno_type])

        # 1.2. Build ann_df frame for visualization
        df = pd_df_to_rpy2_df(
            pd.DataFrame(
                {
                    c.replace(" ", "_"): [v.rx2("meth.diff")[int(i) - 1] for i in inds]
                    for c, v in methyl_diffs.items()
                    if wrt_condition in c.split("vs")[1]
                }
            )
        )

        # 1.3. Plot and save figure
        pdf(str(save_path_prefix) + f"_{anno_type}.pdf", width=width, height=height)
        r_graphics.boxplot(
            df,
            ylab="methylation change",
            main=anno_type.replace("_", " ") + f" (N = {len(inds)})",
            cex_main=4,
            cex_axis=4,
        )
        r_graphics.grid()
        dev_off()


def meth_changes_anno_scatter3d(
    methyl_obj: Any,
    cpgs_ann: Any,
    condition_sample: Dict[str, Iterable[str]],
    wrt_condition: str,
    methyl_diffs: Dict[str, Any],
    save_path_prefix: Path,
    width: int = 5,
    height: int = 5,
) -> None:
    """
    Create 3D scatter plots of methylation changes by annotation categories.

    This function generates 3D scatter plots visualizing the relationship between
    methylation changes from two different comparisons (X and Y axes) and the baseline
    methylation level (Z axis) for different genomic annotation categories. The plots
    help identify complex relationships between multiple methylation changes and
    baseline methylation levels within specific genomic contexts.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        cpgs_ann: A GRanges object containing genome annotations for CpG sites.
        condition_sample: Dictionary mapping condition names to lists of sample IDs
            that belong to each condition.
        wrt_condition: Reference condition name with respect to which the methylation
            changes are measured.
        methyl_diffs: Dictionary mapping comparison names (as strings) to their
            corresponding methylDiff objects.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<annotation_type>_<comparison1>_<comparison2>.pdf'.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plots to the specified paths.
    """
    # 0. Get needed inputs
    # 0.1. Annotate regions and merge to original methylation object,
    # keeping only the successful mappings
    regions = homogeinize_seqlevels_style(
        make_granges_from_dataframe(
            ro.r("data.frame")(methyl_obj), keep_extra_columns=True
        ),
        cpgs_ann,
    )
    meth_ann = annotate_regions(
        regions=regions,
        annotations=cpgs_ann,
        ignore_strand=True,
        quiet=False,
    )
    meth_ann_df = rpy2_df_to_pd_df(ro.r("as.data.frame")(meth_ann))

    colors_fn = ro.r(
        """
        f <- function(methyl_obj, inds){
            rbPal <- colorRampPalette(c('blue','yellow','red'))
            Col <- rbPal(100)[as.numeric(cut(methyl_obj, breaks= seq(0,1,.01)))]
            return(Col[inds])
        }
        """
    )

    # 0.2. Get percentage methylation values per condition
    # todo: added casting because PyCharm wrongly complains about types
    p_meth = pd.DataFrame(rpy2_df_to_pd_df(percentage_methylation(methyl_obj)) / 100)
    p_meth_condition = {
        condition: p_meth[samples].median(axis=1)
        for condition, samples in condition_sample.items()
    }

    # 1. Create a figure for each annotation type
    for anno_type in meth_ann_df["annot.type"].unique():
        # 1.1 Filter annotations by annotation type and get indices
        inds = list(meth_ann_df.index[meth_ann_df["annot.type"] == anno_type])

        if len(inds) == 0:
            logging.warning(
                f"Warning: Annotation type '{anno_type}' was not found in annotation."
                " Continuing."
            )
            continue

        # 1.2. Get differential methylation objects where meth_diffs_wrt_condition
        # is the control
        meth_diffs_wrt_condition = [
            (c, meth_diff)
            for c, meth_diff in methyl_diffs.items()
            if wrt_condition in c.split("vs")[1]
        ]

        # 1.3. Create scatterplot3D with meth_diffs combinations (with order,
        # without replacement) in the X, Y axis and wrt_condition in the Z axis.
        for (c0, meth_diff_0), (c1, meth_diff_1) in combinations(
            meth_diffs_wrt_condition, 2
        ):
            data = pd.DataFrame(
                {
                    c0.replace(" ", "_"): [
                        meth_diff_0.rx2("meth.diff")[int(i) - 1] for i in inds
                    ],
                    c1.replace(" ", "_"): [
                        meth_diff_1.rx2("meth.diff")[int(i) - 1] for i in inds
                    ],
                    wrt_condition: list(p_meth_condition[wrt_condition][inds]),
                }
            )

            pdf(
                str(save_path_prefix)
                + f"_{anno_type}_{c0.replace(' ', '_')}_{c1.replace(' ', '_')}.pdf",
                width=width,
                height=height,
            )
            d = r_scatter3d.scatterplot3d(
                pd_df_to_rpy2_df(data),
                pch=".",
                angle=45,
                xlab=c0,
                ylab=c1,
                zlab=wrt_condition,
                color=colors_fn(
                    FloatVector(p_meth_condition[wrt_condition]),
                    IntVector([int(i) for i in inds]),
                ),
            )
            d.rx2("points3d")(
                x=[-100, 100], y=[0, 0], z=[1, 1], type="l", col="grey", lwd=2
            )
            d.rx2("points3d")(
                x=[0, 0], y=[-100, 100], z=[1, 1], type="l", col="grey", lwd=2
            )
            d.rx2("points3d")(x=[0, 0], y=[0, 0], z=[0, 1], type="l", col="grey", lwd=2)
            dev_off()


def methylation_correlation(
    methyl_diffs: Dict[str, Any],
    wrt_condition: str,
    save_path_prefix: Path,
    width: int = 10,
    height: int = 10,
) -> None:
    """
    Analyze correlations between methylation changes in different conditions.

    This function examines whether changes in methylation levels across different
    experimental comparisons are correlated with each other. It generates scatter plots
    with fitted curves to visualize relationships between methylation changes from
    different comparisons that share the same reference condition.

    Args:
        methyl_diffs: Dictionary mapping comparison names (as strings) to their
            corresponding methylDiff objects.
        wrt_condition: Reference condition name with respect to which the methylation
            changes are measured. Only comparisons using this as control will be analyzed.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<comparison1>__vs__<comparison2>.pdf'.
        width: Width of saved figure in inches.
        height: Height of saved figure in inches.

    Returns:
        None: The function saves the generated plots to the specified paths.
    """
    # 0. Get differential methylation objects where meth_diffs_wrt_condition is the
    # control
    meth_diffs_wrt_condition = [
        (c, meth_diff)
        for c, meth_diff in methyl_diffs.items()
        if wrt_condition in c.split("vs")[1]
    ]
    # 1. Define plotting function
    plotter = ro.r(
        """
        library(earth);
        plotCorDiffMeth = function(CH){
            smoothScatter(
                CH[[1]],
                CH[[2]],
                xlab=names(CH)[1],
                ylab=names(CH)[2],
                xlim=c(-60,60),ylim=c(-60,60),
                main='Methylation changes correlation',
                colramp=colorRampPalette(topo.colors(100)),
                cex.lab=2,
                cex.axis=2,
                cex.main=2
            )
            abline(h=0)
            abline(v=0)
            text(-40,40,paste("corr=",round(cor(CH[[1]],CH[[2]]),4)))
                form = as.formula(paste(names(CH)[2],names(CH)[1],sep='~'))
            mars <-earth(
                formula=form,
                ann_df=CH,
                pmethod="backward",
                nprune=20,
                nfold=10
            );
            cuts <- mars$cuts[mars$selected.terms, ];
            xf <- predict(mars, seq(-100,100));
            points(seq(-100,100),xf,t='l',col='red',lwd=3);
        }
        """
    )

    # 2. Plot figures for each 2 comparison permutations that has 'wrt_condition' as
    # control
    for (c0, meth_diff_0), (c1, meth_diff_1) in combinations(
        meth_diffs_wrt_condition, 2
    ):
        # 2.2. Build ann_df frame for visualization
        df = pd.DataFrame(
            {
                c0.replace(" ", ""): meth_diff_0.rx2("meth.diff"),
                c1.replace(" ", ""): meth_diff_1.rx2("meth.diff"),
            }
        )
        df = pd_df_to_rpy2_df(df)

        # 2.3. Plot and save figure
        pdf(
            str(save_path_prefix)
            + f"_{c1.replace(' ', '')}__vs__{c1.replace(' ', '')}.pdf",
            width=width,
            height=height,
        )
        plotter(df)
        dev_off()


def get_pos_neg_neighbours_inds(
    methyl_obj: Any, delta_th: int = 0, d_min_th: int = 0, d_max_th: int = 20
) -> Tuple[List[int], List[int], List[int]]:
    """
    Find indices of positive and negative neighboring CpG sites.

    This function identifies the positive (downstream) and negative (upstream) neighboring
    CpG sites for each methylated position in the genome, applying distance thresholds to
    control which neighbors are considered valid.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        delta_th: Minimum required difference between the distances to positive and
            negative neighbors (helps select positions with asymmetric neighbor distances).
        d_min_th: Minimum distance threshold for a position to be considered a neighbor.
            Positions closer than this will be filtered out.
        d_max_th: Maximum distance threshold for a position to be considered a neighbor.
            Positions further than this will be filtered out.

    Returns:
        Tuple[List[int], List[int], List[int]]: A tuple containing three lists of indices:
            - First element: Indices of valid CpG positions
            - Second element: Indices of corresponding positive neighbors
            - Third element: Indices of corresponding negative neighbors
    """
    # 0. Extract methylation dataframe
    data = r_source.getData(methyl_obj)

    # 1. Get GRanges object
    pos = ro.r("GRanges")(
        seqnames=data.rx2("chr"),
        ranges=ro.r("IRanges")(start=data.rx2("start"), end=data.rx2("end")),
    )
    pos_df = rpy2_df_to_pd_df(ro.r("data.frame")(pos))

    # 2. Obtain indices of positive and negative neighbours and convert to 0-indexing
    inds_pos = list(ro.r("precede")(pos))
    inds_neg = list(ro.r("follow")(pos))

    # 3. Get indices where all positions are ints (non-NA values) and convert to
    # 0-indexing
    inds_valid = [
        i
        for i, (x, y) in enumerate(zip(inds_pos, inds_neg))
        if isinstance(x, int) and isinstance(y, int)
    ]
    inds_pos_valid = [inds_pos[i] - 1 for i in inds_valid]
    inds_neg_valid = [inds_neg[i] - 1 for i in inds_valid]

    # 4. Compute distances between position and positive/negative neighbour
    d_pos = (
        abs(
            pos_df.iloc[inds_valid]["start"].reset_index(drop=True)
            - pos_df.iloc[inds_pos_valid]["start"].reset_index(drop=True)
        )
        - 1
    )
    d_neg = (
        abs(
            pos_df.iloc[inds_valid]["start"].reset_index(drop=True)
            - pos_df.iloc[inds_neg_valid]["start"].reset_index(drop=True)
        )
        - 1
    )

    # 5. Get minimum and maximum distance for each position (which neighbour is closer?)
    d_min = pd.concat([d_pos, d_neg], axis=1).min(axis=1)
    d_max = pd.concat([d_pos, d_neg], axis=1).max(axis=1)
    delta = d_max - d_min

    # 7. Get final indices that satisfy distance thresholds
    inds_valid_tmp = d_min.index[
        (abs(d_max) < d_max_th) & (abs(d_min) > d_min_th) & (delta > delta_th)
    ]

    inds_valid_final = [inds_valid[i] for i in inds_valid_tmp]
    inds_pos_valid_final = [inds_pos_valid[i] for i in inds_valid_tmp]
    inds_neg_valid_final = [inds_neg_valid[i] for i in inds_valid_tmp]

    return inds_valid_final, inds_pos_valid_final, inds_neg_valid_final


def triplet_analysis(
    methyl_obj: Any,
    save_path_prefix: Path,
    delta_th: int = 0,
    d_min_th: int = 0,
    d_max_th: int = 20,
) -> None:
    """
    Analyze methylation patterns in triplets of adjacent CpG sites.

    This function examines how the methylation status of a CpG site relates to the
    methylation status of its neighboring CpG sites. For each sample, it generates
    a heatmap showing the average methylation level of a CpG site based on the
    methylation levels of its positive (downstream) and negative (upstream) neighbors.

    Args:
        methyl_obj: A methylBase or methylBaseDB object containing methylation data.
        save_path_prefix: Path prefix for saving the plot files. Final path will be
            prefix + '_<sample_id>.pdf'.
        delta_th: Minimum required difference between the distances to positive and
            negative neighbors.
        d_min_th: Minimum distance threshold for a position to be considered a neighbor.
        d_max_th: Maximum distance threshold for a position to be considered a neighbor.

    Returns:
        None: The function saves the generated heatmaps to the specified paths.

    Note:
        The heatmap's x-axis represents methylation levels of positive neighbors,
        the y-axis represents methylation levels of negative neighbors, and the
        color represents the average methylation level of the central CpG site.
    """
    # 0. Setup
    df = rpy2_df_to_pd_df(r_source.getData(methyl_obj))
    inds_valid, inds_pos, inds_neg = get_pos_neg_neighbours_inds(
        methyl_obj, delta_th, d_min_th, d_max_th
    )

    # 1. One triplet plot per sample_id
    for i, sample_id in enumerate(ro.r("attr")(methyl_obj, "sample.ids"), 1):
        # 1.1. Get percentage methylation for sample id
        sample_perc_meth = list(df[f"numCs{i}"] / df[f"coverage{i}"] * 100)

        # 1.2. Get percentage methylation for position and positive/negative neighbours
        triplet_dict = {
            "p_meth_pos": [sample_perc_meth[i] for i in inds_pos],
            "p_meth_i": [sample_perc_meth[i] for i in inds_valid],
            "p_meth_neg": [sample_perc_meth[i] for i in inds_neg],
        }
        triplet_df = pd.DataFrame(triplet_dict)

        # 1.3. Divide ann_df into bins and aggregate by mean
        bins_pos = pd.cut(triplet_df["p_meth_pos"], bins=25)
        bins_neg = pd.cut(triplet_df["p_meth_neg"], bins=25)
        triplet_agg = triplet_df.groupby([bins_neg, bins_pos])[["p_meth_i"]].agg("mean")

        # 1.4. Build final aggregate dataframe to plot
        triplet_dict_final = {
            f"{sample_id} (Positive Neighbour)": [
                x[0].right for x in triplet_agg.index.to_list()
            ],
            f"{sample_id} (Negative Neighbour)": [
                x[1].right for x in triplet_agg.index.to_list()
            ],
            f"{sample_id} (Position Methylation)": triplet_agg["p_meth_i"].to_list(),
        }
        triplet_df_final = pd.DataFrame(triplet_dict_final)

        # 1.5. Reformat and plot
        triplet_df_final_plot = triplet_df_final.reset_index().pivot(
            columns=f"{sample_id} (Positive Neighbour)",
            index=f"{sample_id} (Negative Neighbour)",
            values=f"{sample_id} (Position Methylation)",
        )
        plt.figure()
        # ticks_labels = [str(x[1]) for x in triplet_agg.index.to_list()][:25]
        ax = sns.heatmap(triplet_df_final_plot, cmap="rainbow", vmin=0, vmax=100)
        ax.invert_yaxis()
        plt.savefig(str(save_path_prefix) + f"_{sample_id}.pdf")
