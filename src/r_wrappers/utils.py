"""
Utility functions for R package integration via rpy2.

This module provides utility functions that help with the interaction between Python and R
through rpy2. It includes functions for data conversion, file operations, and gene ID mapping.

Attributes:
    r_annotation_dbi: The imported AnnotationDbi R package.
    r_pdist: The imported parallelDist R package.
    r_utils: The imported utils R package.
    r_genomic_ranges: The imported GenomicRanges R package.
"""

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as ro
from rpy2.rinterface_lib.sexp import NACharacterType
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from components.functional_analysis.orgdb import OrgDB

r_annotation_dbi = importr("AnnotationDbi")
r_pdist = importr("parallelDist")
r_utils = importr("utils")
r_genomic_ranges = importr("GenomicRanges")


def rpy2_df_to_pd_df(rpy2_df: Any) -> pd.DataFrame:
    """
    Converts a rpy2 DataFrame object to a pandas DataFrame object.

    Uses the rpy2 conversion utilities to transform R dataframes into pandas dataframes.

    Args:
        rpy2_df: An R dataframe to be converted to pandas. Will be coerced to
            a data.frame if it isn't one.

    Returns:
        A pandas DataFrame containing the converted data.

    Note:
        For more information see: https://rpy2.github.io/doc/latest/html/pandas.html
    """
    # 0. Ensure rpy2 object is (or is convertible to) an R dataframe
    rpy2_df = ro.r("as.data.frame")(rpy2_df)

    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(rpy2_df)

    return pd_from_r_df


def rpy2_df_to_pd_df_manual(rpy2_df: Any) -> pd.DataFrame:
    """
    Manually converts a rpy2 DataFrame object to a pandas DataFrame object.

    This is a somewhat more robust approach for dataframes that have complex
    values that may cause issues with the automatic conversion.

    Args:
        rpy2_df: An R dataframe to be converted to pandas. Will be coerced to
            a data.frame if it isn't one.

    Returns:
        A pandas DataFrame containing the converted data with NA_integer_ values
        replaced with numpy's np.nan.
    """
    # 0. Ensure rpy2 object is (or is convertible to) an R dataframe
    rpy2_df = ro.r("data.frame")(rpy2_df)

    df_dict = {
        col: [
            str(x) if isinstance(x, ro.Strvector) else tuple(list(x))
            for x in rpy2_df.rx2(col)
        ]
        for col in rpy2_df.colnames
    }

    return pd.DataFrame(df_dict, index=list(rpy2_df.rownames)).replace(
        "NA_integer_", np.nan
    )


def pd_df_to_rpy2_df(pd_df: pd.DataFrame) -> ro.DataFrame:
    """
    Converts a pandas DataFrame object to a rpy2 DataFrame object.

    Uses the rpy2 conversion utilities to transform pandas dataframes into R dataframes.

    Args:
        pd_df: A pandas DataFrame to be converted to R.

    Returns:
        An R DataFrame containing the converted data.

    Note:
        For more information see: https://rpy2.github.io/doc/latest/html/pandas.html
    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(pd_df)
    return r_from_pd_df


def assay_to_df(data: Any) -> pd.DataFrame:
    """
    Transforms an object that can be converted to an assay into a DataFrame.

    Args:
        data: An R object that has an assay method (e.g., SummarizedExperiment).

    Returns:
        A pandas DataFrame containing the assay data.
    """
    return rpy2_df_to_pd_df(ro.r("assay")(data))


def read_rds(load_path: Path) -> Any:
    """
    Reads a .RDS file and loads it into an R object.

    Args:
        load_path: Path to the .RDS file to be loaded.

    Returns:
        The R object stored in the .RDS file.

    Raises:
        ValueError: If the file extension is not .RDS or the file doesn't exist.
    """
    # 0. Ensure path has the right extension
    if load_path.suffix.upper() != ".RDS" and load_path.is_file():
        raise ValueError("The extension of the file provided must be .RDS")

    return ro.r.readRDS(str(load_path))


def save_rds(obj: Any, save_path: Path) -> None:
    """
    Saves a given R object into an .RDS file.

    Args:
        obj: R object to save.
        save_path: Path where to save the .RDS file.

    Raises:
        ValueError: If the file extension is not .RDS.
    """
    # 0. Ensure path has the right extension
    if save_path.suffix != ".RDS":
        raise ValueError("The extension of the file provided must be .RDS")

    ro.r.saveRDS(obj, file=str(save_path))


def save_csv(obj: Any, save_path: Path) -> None:
    """
    Saves a given R object into a .csv file.

    Args:
        obj: R object to save, typically a data.frame.
        save_path: Path where to save the .csv file.

    Raises:
        ValueError: If the file extension is not .csv.
    """
    # 0. Ensure path has the right extension
    if save_path.suffix != ".csv":
        raise ValueError("The extension of the file provided must be .csv")

    # 1. Save file
    r_utils.write_csv(obj, str(save_path))


def df_to_file(obj: Any, save_path: Path, sep: str = ",") -> None:
    """
    Saves a given R data frame into a file, with a given separator between columns.

    Args:
        obj: R object to save, typically a data.frame.
        save_path: Path where to save the file.
        sep: Separator character to use between columns. Defaults to comma.
    """
    r_utils.write_table(obj, str(save_path), sep=sep)


def sample_distance(data: Any) -> Any:
    """
    Computes the sample distances for given data and returns the distance matrix.

    Args:
        data: Can be a DESeqDataSet or a DESeqTransform (from rlog or vst transforms).

    Returns:
        An R distance matrix object containing pairwise distances between samples.
    """
    with localconverter(ro.default_converter):
        return r_pdist.parDist(ro.r("t")(ro.r("assay")(data)))


def annotate_deseq_result(
    result: rpy2.robjects.methods.RS4,
    org_db: OrgDB,
    from_type: str = "ENSEMBL",
    replace_na: bool = False,
) -> pd.DataFrame:
    """
    Annotates a DESeq result object, adding columns with additional gene information.

    Args:
        result: A DESeqResults object.
        org_db: Organism annotation database.
        from_type: Original ID naming scheme. Possible values: "ENSEMBL",
            "ENTREZID" and "SYMBOL".
        replace_na: Whether to replace NA values (due to failed mapping)
            with original gene ids.

    Returns:
        A pandas DataFrame with the DESeq results annotated with additional gene
        information (ENTREZID, SYMBOL, GENENAME, GENETYPE).
    """
    # 1. Results object to pandas dataframe
    result_df = rpy2_df_to_pd_df(result)

    # 2. Get gene annotations
    to_types = ["ENTREZID", "SYMBOL", "GENENAME", "GENETYPE"]
    try:
        feature_annotations = pd.concat(
            [
                map_gene_id(result_df.index, org_db, from_type, to_type=to_type)
                for to_type in to_types
            ],
            axis=1,
        )
    except Exception as e:
        logging.warn(e)
        feature_annotations = pd.DataFrame(
            index=result_df.index,
            columns=to_types,
        )

    # 3. [Optional] Replace nans with original unmapped IDs
    if replace_na:
        feature_annotations.fillna(result_df.index.to_series())

    # 4. If a column has all NaNs, then replace with original IDs
    for nan_col in feature_annotations.columns[feature_annotations.isna().all()]:
        feature_annotations[nan_col].fillna(result_df.index.to_series(), inplace=True)

    # 5. Add new annotation and return
    return pd.concat([result_df, feature_annotations], axis=1)


def filter_deseq_results(
    result: pd.DataFrame,
    p_filter: str = "pvalue",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 2.0,
) -> pd.DataFrame:
    """
    Filter DESeq results according to statistical metrics.

    Args:
        result: Pandas dataframe of DESeq results.
        p_filter: Column to filter by, usually "pvalue" or "padj".
        p_th: P-value threshold to select the most significant genes.
        lfc_level: Genes to include: "up" for up-regulated, "down" for
            down-regulated, and "all" for all DE genes.
        lfc_th: Log2 fold change threshold. Usually, results are considered to be
            biologically significant when absolute log2FoldChange > 2.

    Returns:
        A filtered pandas DataFrame containing only the genes that meet the specified criteria.
    """
    # 1. Filter by LFC level
    if lfc_level == "up":
        result = result[result["log2FoldChange"] > 0]
    elif lfc_level == "down":
        result = result[result["log2FoldChange"] < 0]

    # 3. Filter by LFC and significance thresholds
    result = result[
        (abs(result["log2FoldChange"]) > lfc_th) & (result[p_filter] < p_th)
    ]

    return result


def get_top_var_genes(
    data: rpy2.robjects.methods.RS4, top_n: int = 1000
) -> ro.IntVector:
    """
    Get the genes with the highest variance estimate in DESeqDataSet or DESeqTransform objects.

    Args:
        data: Data to extract the top var genes from (DESeqDataSet or DESeqTransform).
        top_n: Number of top genes to return.

    Returns:
        An R IntVector of gene indices with the highest variance.
    """
    return ro.r("head")(
        ro.r("order")(ro.r("rowVars")(ro.r("assay")(data))), top_n, decreasing=True
    )


def make_granges_from_dataframe(df: Any, **kwargs) -> Any:
    """
    Creates a GRanges object from a dataframe.

    Takes a data frame-like object as input and tries to automatically find
    the columns that describe genomic ranges. It returns them as a GRanges object.

    Args:
        df: A data frame or DataFrame object. If not, then the function first
            tries to turn df into a data frame with as.data.frame(df).
        **kwargs: Additional arguments to pass to makeGRangesFromDataFrame.

    Returns:
        A GRanges object containing genomic ranges from the dataframe.

    Note:
        For more details see: https://rdrr.io/bioc/GenomicRanges/man/makeGRangesFromDataFrame.html
    """
    df = pd_df_to_rpy2_df(df) if isinstance(df, pd.DataFrame) else df
    return r_genomic_ranges.makeGRangesFromDataFrame(df, **kwargs)


def clear_open_devices() -> None:
    """
    Clear open graphic devices, useful before creating new plots.

    This function ensures no previous R plotting devices are active,
    which can sometimes cause issues when generating new plots.
    """
    f = ro.r(
        """
            f <- function(){
                for (i in dev.list()) {
                    dev.off()
                }
            }
        """
    )
    f()


def map_gene_id(
    genes: Iterable[str],
    org_db: OrgDB,
    from_type: str = "ENSEMBL",
    to_type: str = "ENTREZID",
    multiple_values: str = "list",
) -> pd.Series:
    """
    Changes the ID naming scheme of the given gene set.

    Args:
        genes: Set of gene names. Should be a string vector.
        org_db: Organism annotation database.
        from_type: Original ID naming scheme. Possible values: "ENSEMBL",
            "ENTREZID" and "SYMBOL".
        to_type: Resulting ID naming scheme. Possible values: "ENSEMBL",
            "ENTREZID" and "SYMBOL".
        multiple_values: What to do when multiple values are mapped.
            Options are: "first", "list" (default), "filter", "asNA".

    Returns:
        A pandas Series containing the original gene ids as index (from_type) and
        the new gene ids (to_type) as values.

    Raises:
        ValueError: If from_type or to_type is not a valid column in the organism database.

    Note:
        For more details see: https://rdrr.io/bioc/ensembldb/man/EnsDb-AnnotationDbi.html
    """
    # 0. Check arguments
    allowed_tyes = ro.r("columns")(org_db.db)
    if from_type not in allowed_tyes:
        raise ValueError(f'from_type "{from_type}" not allowed')
    if to_type not in allowed_tyes:
        raise ValueError(f'to_type "{to_type}" not allowed')

    # 1. Annotate all genes
    ann_genes = list(
        r_annotation_dbi.mapIds(
            org_db.db,
            keys=ro.StrVector(list(map(str, genes))),
            column=to_type,
            keytype=from_type,
            multiVals=multiple_values,
        )
    )

    # 2.1. Process mapped results
    if multiple_values == "list":
        ann_genes = [
            "/".join(x) if not isinstance(x[0], NACharacterType) else np.nan
            for x in ann_genes
        ]
    else:
        ann_genes = [
            x if not isinstance(x, NACharacterType) else np.nan for x in ann_genes
        ]

    # 3. Build and return annotated genes, missing values marked as `np.nan`
    return pd.Series(ann_genes, index=genes, name=to_type)


def prepare_gene_list(
    genes: pd.DataFrame,
    org_db: Optional[OrgDB] = None,
    from_type: Optional[str] = None,
    to_type: Optional[str] = None,
    p_col: str = "padj",
    p_th: Optional[float] = None,
    lfc_col: str = "log2FoldChange",
    lfc_level: str = "all",
    lfc_th: Optional[float] = None,
    numeric_col: str = "log2FoldChance",
) -> ro.FloatVector:
    """
    Prepares a gene list for functional analysis with clusterProfiler.

    Loads a gene list from a dataframe and prepares it to match the expected
    format of clusterProfiler. If both "from_type" and "to_type" are
    provided, convert ID types from "from_type" to "to_type".

    Args:
        genes: A dataframe of at least two columns, containing gene IDs and
            a numeric column that can be used to rank them.
        org_db: Organism annotation database.
        from_type: Original ID naming scheme. Possible values: "ENSEMBL",
            "ENTREZID" and "SYMBOL".
        to_type: Resulting ID naming scheme. Possible values: "ENSEMBL",
            "ENTREZID" and "SYMBOL".
        p_col: Name of p-value column.
        p_th: Optionally filter by p_col.
        lfc_col: Name of LFC column.
        lfc_level: Genes to include: "up" for up-regulated, "down" for
            down-regulated, and "all" for all.
        lfc_th: Optionally filter by lfc_col.
        numeric_col: Column that should be used to retrieve the numeric vector
            for the gene list. This can be the fold change column, the stat
            column or any other the user decides in order to sort and, later if
            decided, threshold and filter the list.

    Returns:
        Float vector (R object) with sorted numeric values and names equal to
            gene ids.

    Note:
        The index of `genes` should contain the sample IDs of type `from_type`.
        Functions needing this resulting gene list require that all genes
        are in the same ID namespace, so genes not mapped are removed.
    """
    genes_list = deepcopy(genes)
    # 1. Annotate genes if appropriate arguments are provided
    if from_type and to_type and org_db:
        genes_list[to_type] = map_gene_id(
            genes=genes_list.index.to_list(),
            org_db=org_db,
            from_type=from_type,
            to_type=to_type,
            multiple_values="first",
        )

        genes_list = (
            genes_list.dropna(subset=[to_type])
            .drop_duplicates(subset=[to_type], keep=False)
            .set_index(to_type)
        )

    # 2. Filter results
    # 2.1. By p-value/p-adjusted
    if p_th:
        genes_list = genes_list[genes_list[p_col] < p_th]

    # 2.2. By LFC level
    if lfc_level == "up":
        genes_list = genes_list[genes_list[lfc_col] > 0]
    elif lfc_level == "down":
        genes_list = genes_list[genes_list[lfc_col] < 0]

    # 2.3. By log2 Fold Change
    if lfc_th:
        genes_list = genes_list[abs(genes_list[lfc_col]) > lfc_th]

    # 3. Sort by numeric column
    genes_list.sort_values(numeric_col, ascending=False, inplace=True)

    # 4. Build gene list and return
    x = ro.FloatVector(genes_list[numeric_col])
    x.names = ro.StrVector(genes_list.index)
    return x


def get_design_matrix(
    targets: pd.DataFrame,
    factors: Iterable[str],
    id_col: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Creates a design matrix for differential analysis.

    Args:
        targets: Pandas DataFrame containing sample annotations.
        factors: Factors used for differential analysis. Must be
            columns available in "targets". Duplicates will be ignored.
        id_col: Column containing sample ids.
        **kwargs: Additional arguments to pass to the model.matrix function.

    Returns:
        An R model matrix object suitable for differential expression analysis.

    Raises:
        AssertionError: If not all factors are columns in the targets DataFrame.
    """
    rpy2_targets = pd_df_to_rpy2_df(targets)

    # 0. Ensure that all factors are columns in targets
    assert len(targets.columns.intersection(factors)) == len(factors)

    # 1. Ensure all factors are characters, so that they can be converted to
    # R factors
    for factor in factors:
        rpy2_targets[rpy2_targets.colnames.index(factor)] = ro.r("as.character")(
            rpy2_targets[rpy2_targets.colnames.index(factor)]
        )

    # 2. Build design matrix and return
    fmla = ro.r(f"~ 0 + {'+'.join(factors)}")
    design_matrix = ro.r("model.matrix")(
        ro.r("terms")(fmla, keep_order=True), data=rpy2_targets, **kwargs
    )

    claned_colnames = []
    for colname in design_matrix.colnames:
        for factor in factors:
            colname = re.sub("^" + factor, "", colname)
        claned_colnames.append(colname)
    design_matrix.colnames = ro.StrVector(claned_colnames)

    design_matrix.rownames = (
        rpy2_targets.rx2(id_col) if id_col is not None else rpy2_targets.rownames
    )

    return design_matrix


def homogeinize_seqlevels_style(granges_obj: Any, annotations: Any) -> Any:
    """
    Standardizes sequence level styles between two GRanges objects.

    Changes seqlevels style of input granges accordingly so they match the style of the
    annotations. This is important when working with genome annotations from different sources.

    Args:
        granges_obj: GRanges object whose seqlevels will be changed.
        annotations: Annotation GRanges object with the desired seqlevel style.

    Returns:
        GRanges object with annotation-matching seqlevels.
    """
    return ro.r(
        """
        library(GenomicRanges)
        f <- function(x, annotations) {
            seqlevelsStyle(x) = seqlevelsStyle(annotations)
            seqlevels(x, pruning.mode="tidy") = seqlevels(annotations)
            seqinfo(x) = seqinfo(annotations)
            return(x)
        }
        """
    )(granges_obj, annotations)
