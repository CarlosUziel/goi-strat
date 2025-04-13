"""
Wrappers for R package minfi

All functions have pythonic inputs and outputs.

The minfi package provides tools for analyzing Illumina DNA methylation arrays,
including reading data, normalization, detecting differentially methylated regions,
and visualizing results. It supports various Illumina platforms including 450k and EPIC arrays.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from pathlib import Path
from typing import Any, Union

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr

r_minfi = importr("minfi")
r_graphics = importr("graphics")
r_grdevices = importr("grDevices")
r_scatterplot3d = importr("scatterplot3d")
r_color_brewer = importr("RColorBrewer")

pdf = ro.r("pdf")
dev_off = ro.r("dev.off")


def get_annotation(ann_file: Union[str, Any], **kwargs: Any) -> Any:
    """Access annotation data for Illumina methylation arrays.

    This function retrieves annotation information for Illumina methylation arrays,
    including probe location, CpG context, and nearby SNPs.

    Args:
        ann_file: Name of the annotation package to load (e.g.
            "IlluminaHumanMethylation450kanno.ilmn12.hg19") or an R object
            containing the annotation.
        **kwargs: Additional arguments to pass to the getAnnotation function.
            Common parameters include:
            - what: Type of annotation to return, e.g. "all", "Locations", "Islands".
            - lociNames: Subset of loci to return annotation for.

    Returns:
        Any: A DataFrame containing the requested annotation information.
        The exact columns depend on the what parameter but typically include
        chromosome position, relation to CpG islands, and nearby genes.

    Examples:
        >>> # Using a package name
        >>> annotation = get_annotation("IlluminaHumanMethylation450kanno.ilmn12.hg19")
        >>> # Using an R object
        >>> annotation = get_annotation(ann_obj, what="Islands")

    References:
        https://rdrr.io/bioc/minfi/man/getAnnotation.html
    """
    # 0. Import annotation file
    if isinstance(ann_file, str):
        ro.r(f"library({ann_file})")
        ann_obj = ro.r(f"{ann_file}")
    else:
        ann_obj = ann_file

    # 1. Get data annotation and return
    return r_minfi.getAnnotation(ann_obj, **kwargs)


def read_metharray_sheet(data_dir: Path, **kwargs: Any) -> ro.DataFrame:
    """Read an Illumina methylation sample sheet containing phenotype data.

    This function reads a sample sheet for an Illumina methylation experiment,
    containing information about samples, such as sample IDs, sentrix IDs,
    and phenotype variables.

    Args:
        data_dir: The base directory containing the sample sheet and .idat files.
        **kwargs: Additional arguments to pass to the read.metharray.sheet function.
            Common parameters include:
            - pattern: Regular expression pattern to find the sample sheet.
            - ignore.case: Whether to ignore case in the pattern matching.
            - recursive: Whether to look recursively in subdirectories.
            - verbose: Whether to print verbose output.

    Returns:
        ro.DataFrame: A DataFrame containing the sample information from the
        sample sheet. The columns typically include Sample_Name, Sentrix_ID,
        Sentrix_Position, and any phenotypic variables included in the sheet.

    References:
        https://rdrr.io/bioc/minfi/man/read.metharray.sheet.html
    """
    return r_minfi.read_metharray_sheet(str(data_dir), **kwargs)


def read_metharray_exp(
    data_dir: Path, targets: ro.DataFrame, id_col: str = "ID", **kwargs: Any
) -> Any:
    """Read an entire methylation array experiment using a sample sheet.

    This function reads .idat files for an entire methylation array experiment
    using the sample information provided in the targets DataFrame.

    Args:
        data_dir: The base directory containing the .idat files.
        targets: A DataFrame containing sample information, typically obtained
            from read_metharray_sheet().
        id_col: The column in targets containing sample IDs to use for naming.
        **kwargs: Additional arguments to pass to the read.metharray.exp function.
            Common parameters include:
            - recursive: Whether to look recursively in subdirectories.
            - verbose: Whether to print verbose output.
            - force: Whether to force reading of files that don't match targets.

    Returns:
        Any: An RGChannelSet object containing the raw methylation data from
        the .idat files, with samples named according to the id_col in targets.

    Notes:
        This function first creates an RGChannelSet object and then renames
        the samples using the values in the specified id_col of the targets DataFrame.

    References:
        https://rdrr.io/bioc/minfi/man/read.metharray.exp.html
    """
    # 0. Get RG Set
    rg_set = r_minfi.read_metharray_exp(base=str(data_dir), targets=targets, **kwargs)

    # 1. Give the samples descriptive names based on information from targets
    f = ro.r(
        """
            f <- function(rgSet, sampleNames){
                sampleNames(rgSet) <- sampleNames
                return(rgSet)
            }
            """
    )
    return f(rg_set, targets.rx2(id_col))


def combine_arrays(rg_set_0: Any, rg_set_1: Any, **kwargs: Any) -> Any:
    """Combine different types of Illumina methylation arrays.

    This function combines data from different generations of Illumina methylation
    arrays (27k, 450k, EPIC) into a virtual array. This is particularly useful
    when integrating data from 450k and EPIC arrays, which share many common probes.

    Args:
        rg_set_0: First RGChannelSet or MethylSet object.
        rg_set_1: Second RGChannelSet or MethylSet object.
        **kwargs: Additional arguments to pass to the combineArrays function.
            Common parameters include:
            - outType: Output type, one of "IlluminaHumanMethylation450k" or
              "IlluminaHumanMethylationEPIC".
            - verbose: Whether to print verbose output.

    Returns:
        Any: A combined object of the same class as the input objects, containing
        data for probes common to both arrays or present in at least one array
        (depending on the outType parameter).

    Notes:
        The resulting object will be like the specified outType array with
        many probes potentially missing.

    References:
        https://rdrr.io/bioc/minfi/man/combineArrays.html
    """
    return r_minfi.combineArrays(rg_set_0, rg_set_1, **kwargs)


def detection_p(rg_set: Any, **kwargs: Any) -> Any:
    """Calculate detection p-values for methylation probes.

    This function identifies failed positions by comparing signal intensities
    to background noise. Positions where both methylated and unmethylated
    channels report background signal levels are considered failed.

    Args:
        rg_set: An RGChannelSet object.
        **kwargs: Additional arguments to pass to the detectionP function.
            Common parameters include:
            - type: Type of detection p-value to compute, either "m+u" (default)
              or "negative".
            - controllocs: Optional locations of control probes.

    Returns:
        Any: A matrix of detection p-values, with rows corresponding to probes
        and columns corresponding to samples. Lower p-values indicate higher
        confidence that the signal is above background noise.

    References:
        https://rdrr.io/bioc/minfi/man/detectionP.html
    """
    return r_minfi.detectionP(rg_set, **kwargs)


def detection_p_barplot(
    det_p: Any,
    sample_groups: ro.StrVector,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Plot mean detection p-values across samples to identify failed samples.

    This function creates a barplot of mean detection p-values for each sample,
    color-coded by sample group. Samples with high mean detection p-values
    (e.g., above 0.05) may indicate poor quality or failed samples.

    Args:
        det_p: A matrix of detection p-values from detection_p().
        sample_groups: A vector of sample group labels for color-coding.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the barplot function.
            Common parameters include:
            - main: Plot title.
            - ylim: y-axis limits.

    Notes:
        A horizontal red line is drawn at p=0.05 as a common threshold
        for identifying potentially problematic samples.
    """
    palette = r_color_brewer.brewer_pal(8, "Dark2")
    col_f = ro.r(
        """
                    f <- function(pal, sample_groups){
                        return(pal[factor(sample_groups)])
                    }
                    """
    )
    pdf(str(save_path), width=width, height=height)
    r_graphics.barplot(
        ro.r("colMeans")(det_p),
        col=col_f(palette, sample_groups),
        las=2,
        cex_names=0.8,
        ylab="Mean detection p-values",
        **kwargs,
    )

    r_graphics.abline(h=0.05, col="red")
    r_graphics.legend(
        "topleft",
        legend=ro.r("levels")(ro.r("factor")(sample_groups)),
        fill=palette,
        bg="white",
    )
    dev_off()


def get_qc(rg_set: Any) -> ro.DataFrame:
    """Estimate sample-specific quality control metrics for methylation data.

    This function computes quality control metrics for each sample in a
    methylation dataset, including median methylated and unmethylated signal
    intensities across all probes.

    Args:
        rg_set: An object of class RGChannelSet or MethylSet.

    Returns:
        ro.DataFrame: A DataFrame with two columns: mMed and uMed which are
        the chipwide medians of the methylated and unmethylated channels,
        respectively. Each row corresponds to a sample.

    Notes:
        High-quality samples typically show both high mMed and uMed values,
        while failed samples often have low values for both metrics.

    References:
        https://rdrr.io/bioc/minfi/man/getQC.html
    """
    return r_minfi.getQC(rg_set)


def plot_qc(qc: Any, save_path: Path, width: int = 10, height: int = 10) -> None:
    """Plot sample-specific quality control metrics.

    This function creates a scatterplot of the methylated vs. unmethylated
    signal intensities for each sample, which helps visualize the overall
    quality of samples.

    Args:
        qc: A DataFrame with QC metrics as produced by get_qc().
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.

    Notes:
        Samples clustering in the lower-left corner (low methylated and
        unmethylated intensities) are potential candidates for exclusion
        due to poor quality.

    References:
        https://rdrr.io/bioc/minfi/man/getQC.html
    """
    pdf(str(save_path), width=width, height=height)
    r_minfi.plotQC(qc)
    dev_off()


def qc_report(rg_set: Any, **kwargs: Any) -> None:
    """Generate a comprehensive PDF QC report for methylation arrays.

    This function produces a PDF quality control report for Illumina Infinium
    Human Methylation arrays, including various plots and statistics useful
    for identifying failed samples.

    Args:
        rg_set: An RGChannelSet object.
        **kwargs: Additional arguments to pass to the qcReport function.
            Common parameters include:
            - sampNames: Sample names for the report.
            - sampGroups: Sample grouping for color-coding.
            - pdf: File name for the PDF report.
            - maxSamplesPerPage: Maximum number of samples to show per page.
            - controls: Whether to include control probe plots.

    Notes:
        The report includes density plots of methylated and unmethylated signals,
        control probe plots, and sample quality metrics.

    References:
        https://rdrr.io/bioc/minfi/man/qcReport.html
    """
    r_minfi.qcReport(rg_set, **kwargs)


def preprocess_raw(rg_set: Any) -> Any:
    """Convert raw methylation signals to methylation values without normalization.

    This function converts the Red/Green channel data from an Illumina
    methylation array into methylation values (Beta or M-values) without
    applying any normalization.

    Args:
        rg_set: An object of class RGChannelSet.

    Returns:
        Any: A MethylSet object containing the raw methylation and unmethylation
        signals for each probe and sample.

    Notes:
        This is the most basic preprocessing step and is often followed by
        more advanced normalization methods like Noob, SWAN, or Functional
        normalization.

    References:
        https://rdrr.io/bioc/minfi/man/preprocessRaw.html
    """
    return r_minfi.preprocessRaw(rg_set)


def preprocess_funnorm(rg_set: Any, **kwargs: Any) -> Any:
    """Apply Functional Normalization to methylation array data.

    Functional normalization (FunNorm) is a between-array normalization method
    that removes unwanted technical variation by regressing out variability
    explained by the control probes present on the array.

    Args:
        rg_set: An object of class RGChannelSet.
        **kwargs: Additional arguments to pass to the preprocessFunnorm function.
            Common parameters include:
            - nPCs: Number of principal components to use (default: 2).
            - sex: Whether to normalize males and females separately (default: TRUE).
            - bgCorr: Whether to apply Noob background correction (default: TRUE).
            - dyeCorr: Whether to apply dye-bias correction (default: TRUE).
            - keepCN: Whether to keep copy number information (default: TRUE).

    Returns:
        Any: A GenomicRatioSet object containing normalized methylation data
        with genomic coordinates.

    Notes:
        Functional normalization is particularly useful for studies with global
        methylation differences between samples, such as cancer/normal comparisons.

    References:
        https://rdrr.io/bioc/minfi/man/preprocessFunnorm.html
    """
    return r_minfi.preprocessFunnorm(rg_set, **kwargs)


def preprocess_illumina(rg_set: Any, **kwargs: Any) -> Any:
    """Apply Illumina's standard preprocessing to methylation array data.

    This function implements the preprocessing for Illumina methylation arrays
    as used in GenomeStudio, the standard software provided by Illumina.

    Args:
        rg_set: An object of class RGChannelSet.
        **kwargs: Additional arguments to pass to the preprocessIllumina function.
            Common parameters include:
            - bg.correct: Whether to apply background correction (default: TRUE).
            - normalize: Whether to apply normalization (default: "controls").
            - reference: Reference array for normalization.

    Returns:
        Any: A MethylSet object containing the preprocessed methylation data.

    Notes:
        This normalization method is based on internal control probes and
        is Illumina's standard approach, but more advanced methods are often
        preferred in the scientific community.

    References:
        https://rdrr.io/bioc/minfi/man/preprocessIllumina.html
    """
    return r_minfi.preprocessIllumina(rg_set, **kwargs)


def preprocess_noob(rg_set: Any, **kwargs: Any) -> Any:
    """Apply Noob background correction and dye-bias normalization.

    Noob (normal-exponential out-of-band) is a background correction method
    with dye-bias normalization for Illumina Infinium methylation arrays,
    which reduces technical variation.

    Args:
        rg_set: An object of class RGChannelSet.
        **kwargs: Additional arguments to pass to the preprocessNoob function.
            Common parameters include:
            - offset: Offset to add to intensities (default: 15).
            - dyeCorr: Whether to perform dye-bias correction (default: TRUE).
            - dyeMethod: Method for dye-bias correction (default: "single").

    Returns:
        Any: A MethylSet object containing the background-corrected and
        dye-bias normalized methylation data.

    Notes:
        Noob is often used as a first step in preprocessing, followed by
        other normalization methods like quantile normalization.

    References:
        https://rdrr.io/bioc/minfi/man/preprocessNoob.html
    """
    return r_minfi.preprocessNoob(rg_set, **kwargs)


def preprocess_swan(rg_set: Any, **kwargs: Any) -> Any:
    """Apply Subset-quantile Within Array Normalization (SWAN).

    SWAN is a within-array normalization method for Illumina methylation arrays
    that allows Infinium I and II type probes on a single array to be normalized
    together, reducing technical bias between probe types.

    Args:
        rg_set: An object of class RGChannelSet.
        **kwargs: Additional arguments to pass to the preprocessSWAN function.
            Common parameters include:
            - mSet: Optional pre-processed MethylSet.
            - verbose: Whether to print progress messages.

    Returns:
        Any: A MethylSet object containing the SWAN-normalized methylation data.

    Notes:
        SWAN addresses the technical differences between Infinium I and II probe
        types, which can introduce bias in downstream analyses if not corrected.

    References:
        https://rdrr.io/bioc/minfi/man/preprocessSwan.html
    """
    return r_minfi.preprocessSWAN(rg_set, **kwargs)


def preprocess_quantile(obj: Any, **kwargs: Any) -> Any:
    """Apply stratified quantile normalization to methylation array data.

    This function implements stratified quantile normalization for Illumina
    methylation arrays, where probes are stratified by genomic region
    (CpG island, shore, etc.) and probe type before normalization.

    Args:
        obj: An object of class RGChannelSet or MethylSet.
        **kwargs: Additional arguments to pass to the preprocessQuantile function.
            Common parameters include:
            - fixOutliers: Whether to fix outliers before normalization (default: TRUE).
            - removeBadSamples: Whether to remove low-quality samples (default: TRUE).
            - badSampleCutoff: Cutoff for identifying bad samples (default: 10.5).
            - quantileNormalize: Whether to perform quantile normalization (default: TRUE).
            - stratified: Whether to stratify by probe type (default: TRUE).
            - mergeManifest: Whether to merge with manifest probe info (default: FALSE).
            - sex: Whether to normalize males and females separately (default: NULL).

    Returns:
        Any: A GenomicRatioSet object containing the quantile-normalized
        methylation data with genomic coordinates.

    Notes:
        Stratified quantile normalization is one of the most commonly used
        normalization methods for 450k and EPIC arrays, as it effectively
        reduces technical variation while respecting biological differences.

    References:
        https://rdrr.io/bioc/minfi/man/preprocessQuantile.html
    """
    return r_minfi.preprocessQuantile(obj, **kwargs)


def mds_plot(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create a Multi-Dimensional Scaling (MDS) plot of methylation data.

    This function creates an MDS plot showing a 2D projection of distances
    between samples based on their methylation profiles, which helps visualize
    sample relationships and identify outliers.

    Args:
        obj: An RGChannelSet, MethylSet, or matrix containing Beta values.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the mdsPlot function.
            Common parameters include:
            - sampGroups: Sample grouping for color-coding.
            - sampNames: Sample names to show in the plot.
            - numPositions: Number of positions to use (default: 1000).
            - numDimensions: Number of dimensions to compute (default: 2).
            - returnData: Whether to return the MDS coordinates (default: FALSE).
            - main: Plot title.

    Notes:
        MDS plots are useful for visualizing overall sample similarity and
        can help identify batch effects, outliers, or sample grouping by
        biological variables.

    References:
        https://rdrr.io/bioc/minfi/man/mdsPlot.html
    """
    # todo: improve color handling
    # 0. Setup color
    palette = r_color_brewer.brewer_pal(8, "Dark2")
    r_grdevices.palette(palette)

    # 1. Plot
    pdf(str(save_path), width=width, height=height)
    r_minfi.mdsPlot(obj, **kwargs)
    dev_off()


def density_plot(
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create density plots of methylation Beta values.

    This function creates density plots of methylation Beta values across
    samples, which can be used for quality control and to visualize the
    distribution of methylation levels.

    Args:
        obj: An RGChannelSet, MethylSet, or matrix containing Beta values.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the densityPlot function.
            Common parameters include:
            - sampGroups: Sample grouping for color-coding.
            - main: Plot title.
            - xlab: x-axis label.
            - legend: Whether to show the legend (default: TRUE).
            - legendPos: Position of the legend.

    Notes:
        Density plots help visualize the distribution of beta values across
        samples and can reveal quality issues or global methylation differences
        between sample groups.

    References:
        https://rdrr.io/bioc/minfi/man/densityPlot.html
    """
    # todo: improve color handling
    # 0. Setup color
    palette = r_color_brewer.brewer_pal(8, "Dark2")
    r_grdevices.palette(palette)

    # 1. Plot
    pdf(str(save_path), width=width, height=height)
    r_minfi.densityPlot(obj, **kwargs)
    dev_off()


def density_plot_minfi_pair(
    obj_0: Any,
    obj_1: Any,
    sample_groups: ro.StrVector,
    save_path: Path,
    title_0: str = "Figure 1",
    title_1: str = "Figure 2",
    x_lab_0: str = "values",
    x_lab_1: str = "values",
    width: int = 10,
    height: int = 10,
) -> None:
    """Create side-by-side density plots of raw and normalized methylation data.

    This function creates density plots to compare the distribution of beta values
    before and after normalization, which helps assess the effect of normalization.

    Args:
        obj_0: Raw data object (RGChannelSet, MethylSet, or matrix).
        obj_1: Normalized data object (RGChannelSet, MethylSet, or matrix).
        sample_groups: Sample grouping for color-coding.
        save_path: Path where the generated plot will be saved.
        title_0: Title for the first plot (raw data).
        title_1: Title for the second plot (normalized data).
        x_lab_0: x-axis label for the first plot.
        x_lab_1: x-axis label for the second plot.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.

    Notes:
        This function is useful for visualizing the effect of normalization
        on the distribution of beta values across samples.
    """
    # todo: improve color handling
    # 0. Setup
    palette = r_color_brewer.brewer_pal(8, "Dark2")
    legend = ro.r("levels")(ro.r("factor")(sample_groups))

    # 1. Start plot
    pdf(str(save_path), width=width, height=height)
    r_graphics.par(mfrow=[1, 2])

    # 2. Raw plot
    r_minfi.densityPlot(
        obj_0, sampGroups=sample_groups, main=title_0, legend=False, xlab=x_lab_0
    )
    r_graphics.legend("top", legend=legend, text_col=palette)

    # 3. Normalized plot
    r_minfi.densityPlot(
        obj_1, sampGroups=sample_groups, main=title_1, legend=False, xlab=x_lab_1
    )
    r_graphics.legend("top", legend=legend, text_col=palette)

    # 4. Finish plot
    dev_off()


def drop_loci_with_snps(obj: Any, **kwargs: Any) -> Any:
    """Remove probes potentially affected by SNPs.

    This function removes methylation probes that overlap with known SNPs,
    which can affect methylation measurements and lead to false positives
    in differential methylation analysis.

    Args:
        obj: A minfi object (MethylSet, RatioSet, or GenomicRatioSet).
        **kwargs: Additional arguments to pass to the dropLociWithSnps function.
            Common parameters include:
            - snps: Which SNPs to consider for dropping probes (default: NULL).
            - maf: Minor allele frequency cutoff for SNPs (default: 0).
            - population: Population for MAF filtering (default: "EUR").
            - drop: Logical indicating whether to drop probes (default: TRUE).

    Returns:
        Any: A filtered object of the same class as the input, with SNP-affected
        probes removed.

    Notes:
        Removing probes overlapping with SNPs is a standard quality control step
        in methylation analysis, as SNPs can cause artifactual methylation signals.

    References:
        https://rdrr.io/bioc/minfi/man/getAnnotation.html
    """
    return r_minfi.dropLociWithSnps(obj, **kwargs)


def pca_scatterplot3d(
    data: Any,
    sample_colors: ro.StrVector,
    legend_groups: ro.StrVector,
    legend_colors: ro.StrVector,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Create a 3D scatter plot of principal component analysis results.

    This function performs PCA on methylation data and creates a 3D scatter
    plot of the first three principal components, colored by sample groups.

    Args:
        data: A matrix or dataframe containing methylation data (samples as columns).
        sample_colors: Colors for each sample in the plot.
        legend_groups: Group labels for the legend.
        legend_colors: Colors for each group in the legend.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the scatterplot3d function.
            Common parameters include:
            - main: Plot title.
            - xlab: Label for the x-axis.
            - ylab: Label for the y-axis.
            - zlab: Label for the z-axis.
            - pch: Point type.
            - cex: Point size.

    Notes:
        PCA is a useful technique for dimensionality reduction and visualization
        of high-dimensional methylation data, helping to identify patterns and
        sample groupings.
    """
    # 0. Compute PCA
    tpca = ro.r("data.frame")(
        ro.r("prcomp")(ro.r("t")(data), **{"scale.": True}).rx2("x")
    )

    # 2. Plot
    dev_off()
    pdf(str(save_path), width=width, height=height)
    r_scatterplot3d.scatterplot3d(
        tpca.rx2("PC1"), tpca.rx2("PC2"), tpca.rx2("PC3"), color=sample_colors, **kwargs
    )
    r_graphics.legend(
        "topleft",
        pch=19,
        legend=legend_groups,
        col=legend_colors,
    )
    dev_off()


def plot_cpgs(
    dat: Any,
    cpg: ro.StrVector,
    pheno: ro.StrVector,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs: Any,
) -> None:
    """Plot methylation values for specific CpG sites by phenotype.

    This function creates plots of methylation values for selected CpG sites
    as a function of a categorical or continuous phenotype, which is useful
    for visualizing differential methylation.

    Args:
        dat: An RGChannelSet, MethylSet, or matrix containing methylation values.
        cpg: A vector of CpG probe IDs to plot.
        pheno: A vector of phenotype values for grouping samples.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the plotCpg function.
            Common parameters include:
            - type: Type of plot ("categorical", "continuous", or "heatmap").
            - measure: Measure to plot ("Beta" or "M").
            - main: Plot title.
            - ylab: y-axis label.
            - col: Colors for categorical phenotypes.
            - add: Whether to add to an existing plot.

    Notes:
        This function is useful for visualizing methylation differences between
        groups or correlations with continuous phenotypes for specific CpG sites
        of interest, such as differentially methylated positions.

    References:
        https://rdrr.io/bioc/minfi/man/plotCpg.html
    """
    pdf(str(save_path), width=width, height=height)
    r_minfi.plotCpg(dat, cpg, pheno, **kwargs)
    dev_off()


def density_bean_plot(
    dat: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs: Any
) -> None:
    """Create density 'bean' plots of methylation Beta values.

    This function creates density bean plots of methylation Beta values,
    which combine aspects of boxplots and density plots to visualize the
    distribution of methylation levels.

    Args:
        dat: An RGChannelSet, MethylSet, or matrix containing Beta values.
        save_path: Path where the generated plot will be saved.
        width: Width of the saved figure in inches.
        height: Height of the saved figure in inches.
        **kwargs: Additional arguments to pass to the densityBeanPlot function.
            Common parameters include:
            - sampGroups: Sample grouping for dividing the plot.
            - sampNames: Sample names to show in the plot.
            - main: Plot title.
            - bw: Bandwidth for density estimation.

    Notes:
        Bean plots provide a more detailed view of the distribution of beta values
        than standard boxplots, showing the full density shape.

    References:
        https://rdrr.io/bioc/minfi/man/densityBeanPlot.html
    """
    pdf(str(save_path), width=width, height=height)
    r_minfi.densityBeanPlot(dat, **kwargs)
    dev_off()


def map_to_genome(meth_obj: Any, **kwargs: Any) -> Any:
    """Map methylation array data to genomic coordinates.

    This function maps Illumina methylation array data to genomic coordinates
    using an annotation package, creating a GenomicMethylSet or GenomicRatioSet
    that includes genomic location information.

    Args:
        meth_obj: A MethylSet, RGChannelSet, or RatioSet object.
        **kwargs: Additional arguments to pass to the mapToGenome function.
            Common parameters include:
            - genomeBuild: Genome build to use for mapping.
            - mergeManifest: Whether to merge with manifest probe information.

    Returns:
        Any: A GenomicMethylSet or GenomicRatioSet object containing the
        methylation data with genomic coordinates.

    Notes:
        Mapping to genomic coordinates is necessary for many downstream analyses,
        such as identifying differentially methylated regions or annotating
        methylation sites with nearby genes.

    References:
        https://rdrr.io/bioc/minfi/man/mapToGenome-methods.html
    """
    return r_minfi.mapToGenome(meth_obj, **kwargs)


def read_tcga(filename: Path, **kwargs: Any) -> Any:
    """Read TCGA methylation data from a tab-delimited file.

    This function reads DNA methylation data from The Cancer Genome Atlas (TCGA)
    in tab-delimited format and creates a MethylSet object.

    Args:
        filename: Path to the TCGA methylation data file.
        **kwargs: Additional arguments to pass to the readTCGA function.
            Common parameters include:
            - beta: Whether the file contains Beta values (default: TRUE).
            - sep: Field separator in the file (default: "\\t").
            - verbose: Whether to print progress messages.

    Returns:
        Any: A MethylSet object containing the methylation data from the TCGA file.

    References:
        https://rdrr.io/bioc/minfi/man/readTCGA.html
    """
    return r_minfi.readTCGA(str(filename), **kwargs)


def ratio_convert(meth_obj: Any, **kwargs: Any) -> Any:
    """Convert methylation data to Beta and M-values.

    This function converts methylation data from methylated and unmethylated
    signal intensities to Beta values (proportion of methylation) and M-values
    (log2 ratio of methylated to unmethylated intensities).

    Args:
        meth_obj: A MethylSet or GenomicMethylSet object.
        **kwargs: Additional arguments to pass to the ratioConvert function.
            Common parameters include:
            - what: Which ratios to compute, "Beta", "M", or both.
            - keepCN: Whether to keep copy number information.
            - offset: Offset to add when calculating Beta values (default: 100).

    Returns:
        Any: A RatioSet or GenomicRatioSet object containing Beta and/or M-values.

    Notes:
        Beta values (0-1 scale) are more biologically interpretable but can have
        heteroscedasticity issues at extreme values. M-values (log2 scale) have
        better statistical properties for differential analysis but are less
        interpretable biologically.

    References:
        https://rdrr.io/bioc/minfi/man/ratioConvert-methods.html
    """
    return r_minfi.ratioConvert(meth_obj, **kwargs)


def bump_hunter(obj: Any, design: Any, coef: Union[int, str] = 2, **kwargs: Any) -> Any:
    """Identify differentially methylated regions (DMRs) using bump hunting.

    This function identifies differentially methylated regions by finding
    genomic regions where methylation values show consistent differences
    between groups defined in the design matrix.

    Args:
        obj: A GenomicRatioSet or GenomicMethylSet object.
        design: A design matrix for the model fit.
        coef: Which coefficient to use for DMR detection (default: 2).
        **kwargs: Additional arguments to pass to the bumphunter function.
            Common parameters include:
            - cutoff: Cutoff for calling a bump (default: 0.1).
            - maxGap: Maximum distance for probes in the same bump (default: 1000).
            - nullMethod: Method for assessing statistical significance.
            - pickCutoff: Whether to pick the cutoff automatically.
            - smooth: Whether to smooth the data.
            - smoothFunction: Function used for smoothing.
            - B: Number of permutations for computing null distribution.
            - verbose: Whether to print progress messages.

    Returns:
        Any: A list containing the identified DMRs and various statistics.

    Notes:
        Bump hunting identifies regions with coordinated differential methylation
        across adjacent CpG sites, which are more likely to have biological
        significance than isolated differentially methylated positions.

    References:
        https://rdrr.io/bioc/minfi/man/bumphunter-methods.html
    """
    return r_minfi.bumphunter(obj, design=design, coef=coef, **kwargs)
