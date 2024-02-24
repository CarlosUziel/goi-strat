"""
    Wrappers for R package minfi

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
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


def get_annotation(ann_file: Union[str, Any], **kwargs):
    """
    These functions access provided annotation for various Illumina
    methylation objects.

    *ref docs: https://rdrr.io/bioc/minfi/man/getAnnotation.html

    Args:
        ann_file: Name of the annotation to load or R object. E.g.:
        "IlluminaHumanMethylation450kanno.ilmn12.hg19"
    """
    # 0. Import annotation file
    if isinstance(ann_file, str):
        ro.r(f"library({ann_file})")
        ann_obj = ro.r(f"{ann_file}")
    else:
        ann_obj = ann_file

    # 1. Get data annotation and return
    return r_minfi.getAnnotation(ann_obj, **kwargs)


def read_metharray_sheet(data_dir: Path, **kwargs):
    """
    Reading an Illumina methylation sample sheet, containing pheno-data
    information for the samples in an experiment.

    *ref docs: https://rdrr.io/bioc/minfi/man/read.metharray.sheet.html

    Args:
        data_dir: The base directory from which the search is started. The
            samples files (.idat) should be located here, as well as the sample
            sheet file.
    """
    return r_minfi.read_metharray_sheet(str(data_dir), **kwargs)


def read_metharray_exp(
    data_dir: Path, targets: ro.DataFrame, id_col: str = "ID", **kwargs
):
    """
    Reads an entire methylation array experiment using a sample sheet or
    (optionally) a target like data.frame.

    *ref docs: https://rdrr.io/bioc/minfi/man/read.metharray.exp.html

    Args:
        data_dir: The base directory from which the search is started. The
        samples files (.idat) should be located here,
            as well as the sample sheet file.
        targets: Targets dataframe, as obtained from 'read_metharray_sheet'
        id_col: Targets column containing sample ids
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


def combine_arrays(rg_set_0: Any, rg_set_1: Any, **kwargs):
    """
    A method for combining different types of methylation arrays into a virtual array.
    The three generations of Illumina methylation arrays are supported: the 27k, the
    450k and the EPIC arrays. Specifically, the 450k array and the EPIC array share
    many probes in common. This function combines data from the two different array
    types and outputs a data object of the user-specified type. Essentially, this
    new object will be like (for example) an EPIC array with many probes missing.

    *ref docs: https://rdrr.io/bioc/minfi/man/combineArrays.html
    """
    return r_minfi.combineArrays(rg_set_0, rg_set_1, **kwargs)


def detection_p(rg_set: Any, **kwargs):
    """
    This function identifies failed positions defined as both the methylated
    and unmethylated channel reporting background signal levels.

    *ref docs: https://rdrr.io/bioc/minfi/man/detectionP.html

    Args:
        rg_set: An RGChannelSet.
    """
    return r_minfi.detectionP(rg_set, **kwargs)


def detection_p_barplot(
    det_p: Any,
    sample_groups: ro.StrVector,
    save_path: Path,
    width: int = 10,
    height: int = 10,
    **kwargs,
):
    """
    Plot detection p-values across all samples to identify any failed samples

    Args:
        det_p: A matrix with detection p-values.
        sample_groups: Targets sample groups
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
    """
    Estimate sample-specific quality control (QC) for methylation data.

    *ref docs: https://rdrr.io/bioc/minfi/man/getQC.html

    Args:
        rg_set: An object of class [Genomic]MethylSet.

    Returns:
        DataFrame with two columns: mMed and uMed which are the chipwide
        medians of the Meth and Unmeth channels.
    """
    return r_minfi.getQC(rg_set)


def plot_qc(qc: Any, save_path: Path, width: int = 10, height: int = 10):
    """
    Plot sample-specific quality control (QC) for methylation data.

    *ref docs: https://rdrr.io/bioc/minfi/man/getQC.html

    Args:
        qc: An object as produced by getQC.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_minfi.plotQC(qc)
    dev_off()


def qc_report(rg_set: Any, **kwargs):
    """
    Produces a PDF QC report for Illumina Infinium Human Methylation 450k
    arrays, useful for identifying failed samples.

    **ref docs: https://rdrr.io/bioc/minfi/man/qcReport.html

    Args:
        rg_set: An RGChannelSet.
    """
    r_minfi.qcReport(rg_set, **kwargs)


def preprocess_raw(rg_set: Any):
    """
    Converts the Red/Green channel for an Illumina methylation array into
    methylation signal, without using any
        normalization.

    *ref docs: https://rdrr.io/bioc/minfi/man/preprocessRaw.html

    Args:
        rg_set: An object of class RGChannelSet.
    """
    return r_minfi.preprocessRaw(rg_set)


def preprocess_funnorm(rg_set: Any, **kwargs):
    """
    Functional normalization (FunNorm) is a between-array normalization
    method for the Illumina Infinium
        HumanMethylation450 platform. It removes unwanted variation by
        regressing out variability explained
        by the control probes present on the array.

    *ref docs: https://rdrr.io/bioc/minfi/man/preprocessFunnorm.html

    Args:
        rg_set: An object of class RGChannelSet.
    """
    return r_minfi.preprocessFunnorm(rg_set, **kwargs)


def preprocess_illumina(rg_set: Any, **kwargs):
    """
    These functions implements preprocessing for Illumina methylation
    microarrays as used in Genome Studio, the
        standard software provided by Illumina.

    *ref docs: https://rdrr.io/bioc/minfi/man/preprocessIllumina.html

    Args:
        rg_set: An object of class RGChannelSet.
    """
    return r_minfi.preprocessIllumina(rg_set, **kwargs)


def preprocess_noob(rg_set: Any, **kwargs):
    """
    Noob (normal-exponential out-of-band) is a background correction method
    with dye-bias normalization for Illumina
        Infinium methylation arrays.

    *ref docs: https://rdrr.io/bioc/minfi/man/preprocessNoob.html

    Args:
        rg_set: An object of class RGChannelSet.
    """
    return r_minfi.preprocessNoob(rg_set, **kwargs)


def preprocess_swan(rg_set: Any, **kwargs):
    """
    Subset-quantile Within Array Normalisation (SWAN) is a within array
    normalisation method for the Illumina Infinium
        HumanMethylation450 platform. It allows Infinium I and II type
        probes on a single array to be normalized
        together.

    *ref docs: https://rdrr.io/bioc/minfi/man/preprocessSwan.html

    Args:
        rg_set: An object of class RGChannelSet.
    """
    return r_minfi.preprocessSWAN(rg_set, **kwargs)


def preprocess_quantile(obj: Any, **kwargs):
    """
    Stratified quantile normalization for Illumina amethylation arrays.
    This function implements stratified quantile normalization preprocessing
    for Illumina methylation microarrays. Probes
        are stratified by region (CpG island, shore, etc.)


    *ref docs: https://rdrr.io/bioc/minfi/man/preprocessQuantile.html

    Args:
        obj: An object of class RGChannelSet or [Genomic]MethylSet.

    """
    return r_minfi.preprocessQuantile(obj, **kwargs)


def mds_plot(obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs):
    """
    Multi-dimensional scaling (MDS) plots showing a 2-d projection of
    distances between samples.

    *ref docs: https://rdrr.io/bioc/minfi/man/mdsPlot.html


    Args:
        obj: An RGChannelSet, a MethylSet or a matrix. We either use the
        getBeta function to get Beta values (for the
        first two) or we assume the matrix contains Beta values.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure

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
    obj: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
     Density plots of methylation Beta values, primarily for QC.

     *ref docs: https://rdrr.io/bioc/minfi/man/densityPlot.html

    Args:
         obj: An RGChannelSet, a MethylSet or a matrix. We either use the
         getBeta function to get Beta values (for the
         first two) or we assume the matrix contains Beta values.
         save_path: where to save the generated plot
         width: width of saved figure
         height: height of saved figure
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
):
    """
    Plot raw vs normalized data in a density plot.

    Args:
        obj_0: raw data object
        obj_1: normalized data object
        sample_groups: Targets sample groups
        save_path: where to save the generated plot
        title_0: Title for figure 1
        title_1: Title for figure 2
        x_lab_0: x-axis label for figure 2
        x_lab_1: x-axis label for figure 2
        width: width of saved figure
        height: height of saved figure
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


def drop_loci_with_snps(obj: Any, **kwargs):
    """
    A convenience function for removing loci with SNPs based on their MAF.

    *ref docs: https://rdrr.io/bioc/minfi/man/getAnnotation.html

    Args:
        obj: A minfi object.
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
    **kwargs,
):
    """
    3D Scatter plot of PCA values.

    Args:
        data: data to compute PCA on, a matrix or dataFrame-like object
        sample_colors: list of colors used for each sample, should have the
            same length as the number of columns in data. Each sample belonging
            to the same group should have the same color.
        legend_groups: Sample groups
        legend_colors: Sample groups' colors
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
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
    **kwargs,
):
    """
    Plot single-position (single CpG) methylation values as a function of a
    categorical or continuous phenotype

    *ref docs: https://rdrr.io/bioc/minfi/man/plotCpg.html

    Args:
        dat: An RGChannelSet, a MethylSet or a matrix. We either use the
        getBeta (or getM for measure="M") function to
            get Beta values (or M-values) (for the first two) or we assume
            the matrix contains Beta values (or M-values)
        cpg: A character vector of the CpG position identifiers to be plotted.
        pheno: A vector of phenotype values.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_minfi.plotCpg(dat, cpg, pheno, **kwargs)
    dev_off()


def density_bean_plot(
    dat: Any, save_path: Path, width: int = 10, height: int = 10, **kwargs
):
    """
    Density ‘bean’ plots of methylation Beta values, primarily for QC.

    *ref docs: https://rdrr.io/bioc/minfi/man/densityBeanPlot.html

    Args:
        dat: An RGChannelSet, a MethylSet or a matrix. We either use the
        getBeta function to get Beta values (for the
            first two) or we assume the matrix contains Beta values.
        save_path: where to save the generated plot
        width: width of saved figure
        height: height of saved figure
    """
    pdf(str(save_path), width=width, height=height)
    r_minfi.densityBeanPlot(dat, **kwargs)
    dev_off()


def map_to_genome(meth_obj: Any, **kwargs):
    """
    Mapping Ilumina methylation array data to the genome using an annotation
    package. Depending on the genome, not all methylation loci may have a
    genomic position.

    *ref docs: https://rdrr.io/bioc/minfi/man/mapToGenome-methods.html

    Args:
        meth_obj: Either a MethylSet, a RGChannelSet or a RatioSet.
    """
    return r_minfi.mapToGenome(meth_obj, **kwargs)


def read_tcga(filename: Path, **kwargs):
    """
    Read in tab delimited file in the TCGA format

    *ref docs: https://rdrr.io/bioc/minfi/man/readTCGA.html

    Args:
        filename: The name of the file to be read from.

    """
    return r_minfi.readTCGA(str(filename), **kwargs)


def ratio_convert(meth_obj: Any, **kwargs):
    """
    Converting methylation data from methylation and unmethylation channels,
        to ratios (Beta and M-values).

    *ref docs: https://rdrr.io/bioc/minfi/man/ratioConvert-methods.html

    Args:
        meth_obj: Either a MethylSet, or a GenomicRatioSet.

    Returns:
        An object of class RatioSet or GenomicRatioSet.
    """
    return r_minfi.ratioConvert(meth_obj, **kwargs)
