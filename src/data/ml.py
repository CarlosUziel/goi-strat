"""Machine learning preprocessing utilities for genomic and expression data.

This module provides functions for preparing gene expression, methylation, and gene set
data for machine learning applications, including filtering, normalization, and
extraction of relevant features.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import rpy2.robjects as ro
from sklearn import preprocessing

from components.functional_analysis.orgdb import OrgDB
from data.utils import filter_genes_wrt_annotation, get_overlapping_features
from r_wrappers.deseq2 import vst_transform
from r_wrappers.gsva import gsva
from r_wrappers.msigdb import get_msigb_gene_sets
from r_wrappers.utils import map_gene_id, pd_df_to_rpy2_df, rpy2_df_to_pd_df


def process_gene_count_data(
    counts_file: Path,
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    custom_genes_file: Optional[Path] = None,
    exclude_genes: Optional[Iterable[str]] = None,
) -> Tuple[
    pd.DataFrame, List[int], Iterable[str], pd.DataFrame, preprocessing.LabelEncoder
]:
    """Preprocess gene count data for machine learning tasks.

    Performs multiple preprocessing steps on gene expression data:
    1. Loads gene counts and annotations
    2. Filters to keep only common samples between counts and annotations
    3. Encodes class labels
    4. Applies gene filtering (optional custom genes or exclude specified genes)
    5. Maps gene IDs to ENTREZID and filters duplicates
    6. Removes features with overlapping ranges between classes
    7. Applies min-max scaling to normalize feature values

    Args:
        counts_file: CSV file containing expression data with genes as rows and
            samples as columns
        annot_df: DataFrame containing sample annotations
        contrast_factor: Column name in annot_df containing the class labels
        org_db: Organism annotation database object
        custom_genes_file: Optional file with specific genes to include (e.g., DEGs)
        exclude_genes: Optional list of gene IDs to exclude

    Returns:
        Tuple containing:
            - pd.DataFrame: Processed counts data (samples × genes)
            - List[int]: Encoded class labels
            - Iterable[str]: Gene IDs with non-overlapping ranges between classes
            - pd.DataFrame: Range information for each gene and class
            - LabelEncoder: The encoder used to transform class labels

    Raises:
        AssertionError: If counts or annotation dataframes are empty or
            if there aren't exactly two classes
    """
    # 1. Data
    # 1.1. Loading
    counts_df = pd.read_csv(counts_file, index_col=0).transpose()

    assert not counts_df.empty and not annot_df.empty, (
        "Counts or annotation dataframes are empty."
    )

    # 1.2. Only keep common samples between data and sample annotation
    common_idxs = list(set(counts_df.index).intersection(set(annot_df.index)))
    counts_df = counts_df.loc[common_idxs, :]
    annot_df = annot_df.loc[common_idxs, :]

    assert len(set(annot_df[contrast_factor])) == 2, (
        "Classes were lost after unifying count and annotation data. Please check input"
        " data."
    )

    # 1.3. Build class labels
    label_encoder = preprocessing.LabelEncoder()
    class_labels = label_encoder.fit_transform(annot_df[contrast_factor])

    # 2. Select genes
    if custom_genes_file is not None:
        custom_genes_df = pd.read_csv(custom_genes_file, index_col=0)

        genes = (
            custom_genes_df.index if not custom_genes_df.empty else counts_df.columns
        )

    else:
        genes = counts_df.columns

    if exclude_genes:
        genes = [gene for gene in genes if gene not in exclude_genes]

    # 2.1. Filter genes based on IDs and names
    genes = filter_genes_wrt_annotation(genes, org_db, "ENSEMBL")

    try:
        counts_df = counts_df.loc[:, genes]
    except KeyError:
        logging.warning(
            "Some of the input genes were not found in data_df and are thus ignored."
        )
        counts_df = counts_df.loc[:, counts_df.columns.intersection(set(genes))]

    # 2.2. Get ENTREZID genes IDs
    counts_df.columns = map_gene_id(counts_df.columns, org_db, "ENSEMBL", "ENTREZID")
    # get rid of non-uniquely mapped transcripts
    counts_df = counts_df.loc[:, ~counts_df.columns.str.contains("/", na=False)]
    # remove all transcripts that share ENTREZIDs IDs
    counts_df = counts_df.loc[:, counts_df.columns.dropna().drop_duplicates(keep=False)]

    # 3. Remove non-overlapping genes
    overlapping_genes, counts_df_ranges = get_overlapping_features(
        counts_df,
        [
            annot_df[annot_df[contrast_factor] == class_label].index
            for class_label in label_encoder.classes_
        ],
    )

    counts_df = counts_df.loc[:, overlapping_genes]

    # 4. Scale feature (gene) values between 0 and 1.
    counts_df = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(counts_df),
        index=counts_df.index,
        columns=counts_df.columns,
    )

    return counts_df, class_labels, overlapping_genes, counts_df_ranges, label_encoder


def get_gene_set_expression_data(
    counts: Union[Path, pd.DataFrame],
    annot_df: pd.DataFrame,
    org_db: OrgDB,
    msigdb_cat: str,
    contrast_factor: Optional[str] = None,
    custom_genes_file: Optional[Path] = None,
    exclude_genes: Optional[Iterable[str]] = None,
    gsva_processes: int = 8,
    remove_overlapping: bool = False,
) -> Tuple[
    pd.DataFrame,
    Optional[List[int]],
    Optional[Iterable[str]],
    Optional[pd.DataFrame],
    Optional[preprocessing.LabelEncoder],
]:
    """Transform gene-level expression data to gene set-level enrichment scores.

    Processes gene expression data to generate gene set enrichment scores using GSVA,
    with options for filtering and class-based feature selection.

    Args:
        counts: Path to CSV file or DataFrame with gene expression counts
            (genes as rows, samples as columns)
        annot_df: DataFrame containing sample annotations
        org_db: Organism annotation database object
        msigdb_cat: MSigDB category to use for gene sets (e.g., "H", "C1", "C2")
        contrast_factor: Optional column name in annot_df for class comparison
        custom_genes_file: Optional file with specific genes to include
        exclude_genes: Optional list of gene IDs to exclude
        gsva_processes: Number of processes to use for parallel GSVA computation
        remove_overlapping: Whether to remove gene sets with overlapping ranges between classes

    Returns:
        Tuple containing:
            - pd.DataFrame: Gene set enrichment scores (samples × gene sets)
            - List[int] or None: Encoded class labels (None if contrast_factor is None)
            - Iterable[str] or None: Gene set IDs with non-overlapping ranges
            - pd.DataFrame or None: Range information for each gene set by class
            - LabelEncoder or None: The encoder used to transform class labels

    Raises:
        AssertionError: If msigdb_cat is invalid, or if counts or annotation dataframes are empty,
            or if fewer than 2 classes exist when contrast_factor is specified

    Note:
        This function performs variance stabilizing transformation on the input counts
        before running GSVA to calculate gene set enrichment scores.
    """
    # 0. Setup
    assert msigdb_cat in (
        "H",
        *[f"C{i}" for i in range(1, 9)],
    ), f"{msigdb_cat} is not a valid category"

    # 1. Data
    # 1.1. Loading
    if isinstance(counts, Path):
        counts_df = pd.read_csv(counts, index_col=0)
    else:
        counts_df = deepcopy(counts)

    assert not counts_df.empty and not annot_df.empty, (
        "Counts or annotation dataframes are empty."
    )

    # 1.2. Only keep common samples between data and sample annotation
    common_idxs = counts_df.columns.intersection(annot_df.index)
    counts_df = counts_df.loc[:, common_idxs]
    annot_df = annot_df.loc[common_idxs, :]

    # 1.3. [Optional] Build class labels
    if contrast_factor is not None:
        assert len(set(annot_df[contrast_factor])) >= 2, (
            "There should be at least two classes after unifying count and annotation "
            "data. Please check input data."
        )

        label_encoder = preprocessing.LabelEncoder()
        class_labels = label_encoder.fit_transform(annot_df[contrast_factor])
    else:
        label_encoder, class_labels = (None, None)

    # 2. Select genes
    if custom_genes_file is not None:
        custom_genes_df = pd.read_csv(custom_genes_file, index_col=0)
        genes = custom_genes_df.index if not custom_genes_df.empty else counts_df.index
    else:
        genes = counts_df.index

    if exclude_genes:
        genes = [gene for gene in genes if gene not in exclude_genes]

    # 2.1. Filter genes based on IDs and names
    genes = filter_genes_wrt_annotation(genes, org_db, "ENSEMBL")

    try:
        counts_df = counts_df.loc[genes, :]
    except KeyError:
        logging.warning(
            "Some of the input genes were not found in data_df and are thus ignored."
        )
        counts_df = counts_df.loc[counts_df.index.intersection(set(genes)), :]

    # 2.2. Get ENTREZID genes and filter duplicates
    counts_df.index = map_gene_id(counts_df.index, org_db, "ENSEMBL", "ENTREZID")
    # get rid of non-uniquely mapped transcripts
    counts_df = counts_df.loc[~counts_df.index.str.contains("/", na=False)]
    # remove all transcripts that share ENTREZIDs IDs
    counts_df = counts_df.loc[counts_df.index.dropna().drop_duplicates(keep=False)]

    # 3. Get gene-set by samples matrix
    vst_df = rpy2_df_to_pd_df(
        vst_transform(
            ro.r("as.matrix")(
                pd_df_to_rpy2_df(counts_df.loc[counts_df.mean(axis=1) > 1])
            )
        )
    )
    gene_sets_df = gsva(
        vst_df,
        get_msigb_gene_sets(
            species=org_db.species,
            category=msigdb_cat,
            gene_id_col="entrez_gene",
        ),
        kcdf="Gaussian",
        **{"parallel.sz": gsva_processes},
    ).transpose()

    # 4. [Optional] Detect non-overlapping gene sets
    if contrast_factor is not None:
        overlapping_gene_sets, gene_sets_df_ranges = get_overlapping_features(
            gene_sets_df,
            [
                annot_df[annot_df[contrast_factor] == class_label].index
                for class_label in label_encoder.classes_
            ],
        )

        # 4.1. Remove non-overlapping gene sets
        if remove_overlapping:
            gene_sets_df = gene_sets_df.loc[:, overlapping_gene_sets]
    else:
        overlapping_gene_sets, gene_sets_df_ranges = (None, None)

    return (
        gene_sets_df,
        class_labels,
        overlapping_gene_sets,
        gene_sets_df_ranges,
        label_encoder,
    )


def process_probes_meth_data(
    meth_values_file: Path,
    annot_df: pd.DataFrame,
    contrast_factor: str,
    org_db: OrgDB,
    custom_meth_probes_file: Path,
    exclude_genes: Optional[Iterable[str]] = None,
) -> Tuple[
    pd.DataFrame, List[int], Iterable[str], pd.DataFrame, preprocessing.LabelEncoder
]:
    """Preprocess methylation probe data for machine learning tasks.

    Processes DNA methylation data (beta or M values) to select and filter
    methylation probes based on gene annotations and class-specific features.

    Args:
        meth_values_file: CSV file containing methylation data with probes as columns
            and samples as rows
        annot_df: DataFrame containing sample annotations
        contrast_factor: Column name in annot_df containing the class labels
        org_db: Organism annotation database object
        custom_meth_probes_file: CSV file with pre-selected differentially methylated
            probes that are annotated to gene regions
        exclude_genes: Optional list of ENTREZ IDs to exclude from the analysis

    Returns:
        Tuple containing:
            - pd.DataFrame: Processed methylation data (samples × probes)
            - List[int]: Encoded class labels
            - Iterable[str]: Probe IDs with non-overlapping ranges between classes
            - pd.DataFrame: Range information for each probe and class
            - LabelEncoder: The encoder used to transform class labels

    Raises:
        AssertionError: If methylation or annotation dataframes are empty

    Note:
        This function filters methylation probes by gene annotation and keeps only
        probes with non-overlapping value ranges between classes.
    """
    # 1. Data
    # 1.1. Loading
    meth_values_df = pd.read_csv(meth_values_file, index_col=0).transpose()

    assert not meth_values_df.empty and not annot_df.empty, (
        "Methylation values file or annotation file are empty."
    )

    # 1.2. Only keep common samples between data and sample annotation
    common_idxs = list(set(meth_values_df.index).intersection(set(annot_df.index)))
    meth_values_df = meth_values_df.loc[common_idxs, :]
    annot_df = annot_df.loc[common_idxs, :]

    # 1.3. Filter by class samples and build class labels
    label_encoder = preprocessing.LabelEncoder()
    class_labels = label_encoder.fit_transform(annot_df[contrast_factor])

    # 2. Select genes
    custom_meth_genes = pd.read_csv(custom_meth_probes_file, index_col=0).dropna(
        subset="annot.gene_id"
    )
    custom_meth_genes["annot.gene_id"] = (
        custom_meth_genes["annot.gene_id"].astype(int).astype(str)
    )

    valid_genes = [
        gene
        for gene in filter_genes_wrt_annotation(
            custom_meth_genes["annot.gene_id"].drop_duplicates(keep=False),
            org_db,
            "ENTREZID",
        )
        if gene not in (exclude_genes or [])
    ]
    probes = custom_meth_genes[
        custom_meth_genes["annot.gene_id"].isin(valid_genes)
    ].index

    try:
        meth_values_df = meth_values_df.loc[:, probes]
    except KeyError:
        logging.warning(
            "Some of the custom input probes were not found in data_df and are thus"
            " ignored."
        )
        meth_values_df = meth_values_df.loc[
            :, meth_values_df.columns.intersection(set(probes))
        ]

    # 3. Remove non-overlapping probes
    overlapping_probes, meth_values_df_ranges = get_overlapping_features(
        meth_values_df,
        [
            annot_df[annot_df[contrast_factor] == class_label].index
            for class_label in label_encoder.classes_
        ],
    )

    meth_values_df = meth_values_df.loc[:, overlapping_probes]

    return (
        meth_values_df,
        class_labels,
        overlapping_probes,
        meth_values_df_ranges,
        label_encoder,
    )


def process_gene_sets_data(
    data: pd.DataFrame,
    annot_df: pd.DataFrame,
    contrast_factor: str,
    custom_gene_sets_file: Optional[Path] = None,
    exclude_gene_sets: Optional[Iterable[str]] = None,
) -> Tuple[
    pd.DataFrame, List[int], Iterable[str], pd.DataFrame, preprocessing.LabelEncoder
]:
    """Preprocess gene set enrichment data for machine learning tasks.

    Processes gene set enrichment scores (typically from GSVA output) to prepare
    for machine learning analysis by filtering, selecting, and normalizing data.

    Args:
        data: DataFrame containing gene set enrichment scores with gene sets as rows
            and samples as columns
        annot_df: DataFrame containing sample annotations
        contrast_factor: Column name in annot_df containing the class labels
        custom_gene_sets_file: Optional CSV file with specific gene sets to include
        exclude_gene_sets: Optional list of gene set names to exclude

    Returns:
        Tuple containing:
            - pd.DataFrame: Processed gene set data (samples × gene sets)
            - List[int]: Encoded class labels
            - Iterable[str]: Gene set IDs with non-overlapping ranges between classes
            - pd.DataFrame: Range information for each gene set by class
            - LabelEncoder: The encoder used to transform class labels

    Raises:
        AssertionError: If data or annotation dataframes are empty or
            if there aren't exactly two classes

    Note:
        This function applies min-max scaling to normalize the gene set enrichment
        scores between 0 and 1 for better compatibility with machine learning algorithms.
    """
    # 1. Data
    # 1.1. Loading
    data = data.transpose()

    assert not data.empty and not annot_df.empty, (
        "Counts or annotation dataframes are empty."
    )

    # 1.2. Only keep common samples between data and sample annotation
    common_idxs = list(set(data.index).intersection(set(annot_df.index)))
    data_df = data.loc[common_idxs, :]
    annot_df = annot_df.loc[common_idxs, :]

    assert len(set(annot_df[contrast_factor])) == 2, (
        "Classes were lost after unifying count and annotation data. Please check input"
        " data."
    )

    # 1.3. Build class labels
    label_encoder = preprocessing.LabelEncoder()
    class_labels = label_encoder.fit_transform(annot_df[contrast_factor])

    # 2. Select genes
    if custom_gene_sets_file is not None:
        custom_gene_sets_df = pd.read_csv(custom_gene_sets_file, index_col=0)

        gene_sets = (
            custom_gene_sets_df.index
            if not custom_gene_sets_df.empty
            else data_df.columns
        )

    else:
        gene_sets = data_df.columns

    if exclude_gene_sets:
        gene_sets = [
            gene_set for gene_set in gene_sets if gene_set not in exclude_gene_sets
        ]

    try:
        data_df = data_df.loc[:, gene_sets]
    except KeyError:
        logging.warning(
            "Some of the input gene_sets were not found in data and are thus ignored."
        )
        data_df = data_df.loc[:, data_df.columns.intersection(set(gene_sets))]

    # 3. Remove non-overlapping gene_sets
    overlapping_gene_sets, data_df_ranges = get_overlapping_features(
        data_df,
        [
            annot_df[annot_df[contrast_factor] == class_label].index
            for class_label in label_encoder.classes_
        ],
    )

    data_df = data_df.loc[:, overlapping_gene_sets]

    # 4. Scale feature (gene) values between 0 and 1.
    data_df = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(data_df),
        index=data_df.index,
        columns=data_df.columns,
    )

    return data_df, class_labels, overlapping_gene_sets, data_df_ranges, label_encoder
