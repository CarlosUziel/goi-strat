"""
    Wrappers for R package methylkit

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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
    The function calls methylation percentage per base from sorted Bismark SAM or BAM
        files and reads methylation information as methylKit objects. Bismark is a
        popular aligner for high-throughput bisulfite sequencing experiments and it
        outputs its results in SAM format by default, and can be converted to BAM.
        Bismark SAM/BAM format contains aligner specific tags which are absolutely
        necessary for methylation percentage calling using processBismarkAln.
        SAM/BAM files from other aligners will not work with this function.

    See: https://rdrr.io/bioc/methylKit/man/processBismarkAln-methods.html

    Args:
        files: location of sam or bam file(s).
        sample_ids: the id(s) of samples in the same order as file. Must be the same
            length as files.
        assembly: string that determines the genome assembly. Ex: mm9,hg18 etc.
            This is just a string for book keeping. It can be any string. Although,
            when using multiple files from the same assembly, this string should be
            consistent in each object.

    Returns:
        methylRaw, methylRawList, methylRawDB, methylRawListDB object
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
    The function reads a list of files or single files with methylation information for
        bases/region in the genome and creates a methylrawList or methylraw object.
        The information can be stored as flat file database by creating a
        methylrawlistDB or methylrawDB object.

    See: https://rdrr.io/bioc/methylKit/man/methRead-methods.html

    Args:
        files: file location(s), either a list of locations (each a character string)
            or one location string.
        sample_ids: the id(s) of samples in the same order as file. Must be the same
            length as files.
        assembly: string that determines the genome assembly. Ex: mm9,hg18 etc.
            This is just a string for book keeping. It can be any string. Although,
            when using multiple files from the same assembly, this string should be
            consistent in each object.

    Returns:
        methylRaw, methylRawList, methylRawDB, methylRawListDB object
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
    This function filters methylRaw, methylRawDB, methylRawList and methylRawListDB
        objects. You can filter based on lower read cutoff or high read cutoff. Higher
        read cutoff is usefull to eliminate PCR effects Lower read cutoff is usefull
        for doing better statistical tests.

    See: https://rdrr.io/bioc/methylKit/man/filterByCoverage-methods.html

    Args:
        methyl_obj: a methylRaw, methylRawDB, methylRawList or methylRawListDB object

    Returns:
        methylRaw, methylRawDB, methylRawList or methylRawListDB object depending on
            input object.
    """
    return r_source.filterByCoverage(methylObj=methyl_obj, **kwargs)


def unite(methyl_obj: Any, **kwargs) -> Any:
    """
    This functions unites methylRawList and methylRawListDB objects that only bases with
        coverage from all samples are retained. The resulting object is either of class
        methylBase or methylBaseDB depending on input.

    See: https://rdrr.io/bioc/methylKit/man/unite-methods.html

    Args:
        methyl_obj: a methylRawList or methylRawListDB object to be merged by common
            locations covered by reads.

    Returns:
        A methylBase or methylBaseDB object depending on input
    """
    return r_source.unite(object=methyl_obj, **kwargs)


def reorganize(
    methyl_obj: Any, sample_ids: Iterable[str], treatment: IntVector, **kwargs
) -> Any:
    """
    The function creates a new methylRawList, methylRawListDB, methylBase or
        methylBaseDB object by selecting a subset of samples from the input object,
        which is a methylRawList or methylBase object. You can use the function to
        partition a large methylRawList or methylBase object to smaller object based
        on sample ids or when you want to reorder samples and/or give a new treatmet
        vector.

    See: https://rdrr.io/bioc/methylKit/man/reorganize-methods.html

    Args:
        methyl_obj: a methylRawList, methylRawListDB, methylBase or methylBaseDB object
        sample_ids: a vector for sample.ids to be subset. Order is important and the
            order should be similar to treatment. sample.ids should be a subset or
            reordered version of sample ids in the input object.
        treatment: treatment vector, should be same length as sample.ids vector.

    Returns:
        A methylRawList, methylRawListDB, methylBase or methylBaseDB object depending
            on the input object
    """
    return r_source.reorganize(
        methylObj=methyl_obj, sample_ids=sample_ids, treatment=treatment, **kwargs
    )


def calculate_diff_meth(methyl_obj: Any, **kwargs) -> Any:
    """
    The function calculates differential methylation statistics between two groups of
        samples. The function uses either logistic regression test or Fisher's Exact
        test to calculate differential methylation. See the rest of the help page and
        references for detailed explanation on statistics.

    See: https://rdrr.io/bioc/methylKit/man/calculateDiffMeth-methods.html

    Args:
        methyl_obj: a methylBase or methylBaseDB object to calculate differential
            methylation.

    Returns:
        A methylDiff object containing the differential methylation statistics and
            locations for regions or bases.
    """
    return r_source.calculateDiffMeth(methyl_obj, **kwargs)


def get_methylation_stats(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
) -> Any:
    """
    Get Methylation stats from methylRaw or methylRawDB object.

    See: https://rdrr.io/bioc/methylKit/man/getMethylationStats-methods.html

    Args:
        methyl_obj: a methylRaw or methylRawDB object.
        save_path: where to save the generated plot.
        width: width of saved figure.
        height: height of saved figure.

    Returns:
        A summary of Methylation statistics or plot a histogram of coverage.
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
    The function returns basic statistics about read coverage per base. It can also plot
        a histogram of read coverage values.

    See: https://rdrr.io/bioc/methylKit/man/getCoverageStats-methods.html

    Args:
        methyl_obj: a methylRaw or methylRawDB object.
        save_path: where to save the generated plot.
        width: width of saved figure.
        height: height of saved figure.

    Returns:
        A summary of coverage statistics or plot a histogram of coverage.
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
    The functions returns a matrix of correlation coefficients and/or a set of
        scatterplots showing the relationship between samples. The scatterplots will
        contain also fitted lines using lm() for linear regression and lowess for
        polynomial regression.

    See: https://rdrr.io/bioc/methylKit/man/getCorrelation-methods.html

    Args:
        methyl_obj: a methylBase or methylBaseDB object
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure

    Returns:
        A correlation matrix object and plot scatterPlot
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
    Hierarchical Clustering using methylation data. The function clusters samples
        using hclust function and various distance metrics derived from percent
        methylation per base or per region for each sample.

    See: https://rdrr.io/bioc/methylKit/man/clusterSamples-methods.html

    Args:
        methyl_obj: a methylBase or methylBaseDB object.
        save_path: where to save the generated plot.
        width: width of saved figure.
        height: height of saved figure.

    Returns:
        A tree object of a hierarchical cluster analysis using a set of dissimilarities
            for the n objects being clustered.
    """

    if kwargs.get("plot", False):
        pdf(str(save_path), width=width, height=height)

    res = r_source.clusterSamples(methyl_obj, **kwargs)

    if kwargs.get("plot", False):
        dev_off()

    return res


def pca_samples(
    methyl_obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    The function does a PCA analysis using prcomp function using percent methylation
        matrix as an input.

    See: https://rdrr.io/bioc/methylKit/man/PCASamples-methods.html

    Args:
        methyl_obj: a methylBase or methylBaseDB object.
        save_path: where to save the generated plot.
        width: width of saved figure.
        height: height of saved figure.

    Returns:
        The form of the value returned by PCASamples is the summary of principal
            component analysis by prcomp.
    """

    pdf(str(save_path), width=width, height=height)

    res = r_source.PCASamples(methyl_obj, **kwargs)

    dev_off()

    return res


def get_dd(methyl_obj: Any, condition_sample: Dict[str, Iterable[str]]) -> Any:
    """
        Provides intermediate ann_df-frame needed for plotting.

    Args:
        methyl_obj: A methylBase or methylBaseDB object.
        condition_sample: Mapping between sample names and the condition they belong to.

    Returns:
        Intermediate ann_df-frame needed for plotting
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
        Global distributions of methylation levels.

    Args:
        methyl_obj: a methylBase or methylBaseDB object
        condition_sample: mapping between sample names and the condition they belong to
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure

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
                stat_summary(fun.ann_df=data_summary) +
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
    Obtains those methylated sites that are neighbours (whose positions are 1 distance
        apart). The first neighbour (advancing the chromosome index from left to right)
        is the 'forward' neighbour, whereas the second is the 'reverse' neighbour.

    Args:
        methyl_obj: a methylBase or methylBaseDB object
    """
    # 0. MehtylKit object to ann_df.frame
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
    Global strand-specific Effects. Check out strand-specific methylation status on CpGs
        which are covered on both strands >10 in all samples.

    Args:
        methyl_obj: A methylBase or methylBaseDB object.
        save_path_prefix: Path prefix for each plot saved.
            Final path: prefix + '_<sample_id>.pdf'.
        width: Width of saved figure.
        height: Height of saved figure.
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
    The function subsets a methylDiff or methylDiffDB object in order to get
        differentially methylated bases/regions satisfying thresholds.

    See: https://rdrr.io/bioc/methylKit/man/getMethylDiff-methods.html

    Args:
        methyl_obj: a methylDiff or methylDiffDB object

    Returns:
        A methylDiff or methylDiffDB object containing the differential methylated
            locations satisfying the criteria.
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
        Plot number of differentially methylated CpG sites

    Args:
        methyl_diff_df: Dataframe containing the number of differentially methylated
            CpG sites per comparison.
        save_path: where to save the generated plot.
        width: width of saved figure.
        height: height of saved figure.
    """
    pdf(str(save_path), width=width, height=height)
    r_graphics.barplot(methyl_diff_df, **kwargs)
    dev_off()


def methyl_diff_density_plot(
    methyl_diffs: Dict[str, Any], save_path: Path, width: int = 20, height: int = 10
) -> None:
    """
    Density plot of differentially methylated comparisons.

    Args:
        methyl_diffs: Mapping between comparisons (described as strings) and the
            differentially methylated objects.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
    Get percent methylation scores from methylBase or methylBaseDB object.

    See: https://rdrr.io/bioc/methylKit/man/percMethylation-methods.html

    Args:
        methyl_obj: a methylBase or methylBaseDB object
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
):
    """
    Does methylation change depend on methylation level in with respect to the chosen
        condition?

    Args:
        methyl_diffs: Mapping between comparisons (described as strings) and the
            differentially methylated objects.
        methyl_obj: a methylBase or methylBaseDB object.
        condition_sample: mapping between sample names and the condition they belong to
        wrt_condition: condition with respect to the change comparison is done.
        save_path_prefix: path prefix for each plot saved.
            Final path: prefix + '_<sample_id>.pdf'
        width: width of saved figure.
        height: height of saved figure.

    """
    # 0. Setup ann_df inputs for plotting
    # todo: added casting because PyCharm wrongly complains about types
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
    Methylation Levels with regard to CpG Islands Annotation.

    Args:
        methyl_obj: a methylBase or methylBaseDB object
        cpgs_ann: annotated cpgs
        condition_sample: mapping between sample names and the condition they belong to
        save_path_prefix: path prefix for each plot saved.
            Final path: prefix + '_<sample_id>.pdf'
        width: width of saved figure
        height: height of saved figure
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
    Methylation Changes with regard to CpG Islands Annotation.
    Args:
        methyl_obj: a methylBase or methylBaseDB object
        cpgs_ann: annotated cpgs.
        wrt_condition: condition with respect to the change comparison is done.
        methyl_diffs: Mapping between comparisons (described as strings) and the
            differentially methylated objects.
        save_path_prefix: path prefix for each plot saved.
            Final path: prefix + '_<sample_id>.pdf'
        width: width of saved figure.
        height: height of saved figure.
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
    Methylation Changes with regard to CpG Islands Annotation, as a 3D scatter plot.

    Args:
        methyl_obj: a methylBase or methylBaseDB object
        cpgs_ann: annotated cpgs
        condition_sample: mapping between sample names and the condition they belong to
        wrt_condition: condition with respect to the change comparison is done.
        methyl_diffs: Mapping between comparisons (described as strings) and the
            differentially methylated objects.
        save_path_prefix: path prefix for each plot saved.
            Final path: prefix + '_<sample_id>.pdf'
        width: width of saved figure
        height: height of saved figure
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
                + f'_{anno_type}_{c0.replace(" ", "_")}_{c1.replace(" ", "_")}.pdf',
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
    Are Changes in different conditions correlated?

    Args:
        methyl_diffs: Mapping between comparisons (described as strings) and the
            differentially methylated objects.
        wrt_condition: condition with respect to the change comparison is done.
        save_path_prefix: path prefix for each plot saved.
            E.g. final path: prefix + '_<sample_id>.pdf'
        width: width of saved figure
        height: height of saved figure

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
            + f'_{c1.replace(" ", "")}__vs__{c1.replace(" ", "")}.pdf',
            width=width,
            height=height,
        )
        plotter(df)
        dev_off()


def get_pos_neg_neighbours_inds(
    methyl_obj: Any, delta_th: int = 0, d_min_th: int = 0, d_max_th: int = 20
) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
    """
    Obtain the indices where the positive and negative neighbours of each methylated
        position are.

    Args:
        methyl_obj: a methylBase or methylBaseDB object.
        delta_th: Minimum distance between positive and negative neighbours.
        d_min_th: Minimum distance to position be considered neighbour.
        d_max_th: Maximum distance to position be considered neighbour.
    """
    # 0. Extract methylation ann_df
    data = r_source.getData(methyl_obj)

    # 1. Get GRanges object
    pos = ro.r("GRanges")(
        seqnames=data.rx2("chr"),
        ranges=ro.r("IRanges")(start=data.rx2("start"), end=data.rx2("end")),
    )
    pos_df = rpy2_df_to_pd_df(ro.r("ann_df.frame")(pos))

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
    For each sample, plot a triplet heatmap of the average methylation of negative and
        positive neighbours.

    Args:
        methyl_obj: a methylBase or methylBaseDB object.
        save_path_prefix: path prefix for each plot saved.
            Final path: prefix + '_<sample_id>.pdf'.
        delta_th: Minimum distance between positive and negative neighbours.
        d_min_th: Minimum distance to position be considered neighbour.
        d_max_th: Maximum distance to position be considered neighbour.

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
