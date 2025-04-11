import logging
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

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
    annot_df: Path,
    contrast_factor: str,
    org_db: OrgDB,
    custom_genes_file: Path = None,
    exclude_genes: Iterable[str] = None,
) -> Tuple[pd.DataFrame, Iterable[int], Iterable[str], pd.DataFrame]:
    """Perform multiple pre-processing on gene expression data.

    Args:
        counts_file: A .csv file containing expression data of shape
            [n_genes, n_samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        org_db: Organism annotation database.
        custom_genes_path: A .csv file where the first column is a list of relevant
            ENSEMBL IDs genes, such as DEGs.
        exclude_genes: ENSEMBL ID genes to remove from data.

    Returns:
        Processed data.
    """
    # 1. Data
    # 1.1. Loading
    counts_df = pd.read_csv(counts_file, index_col=0).transpose()

    assert (
        not counts_df.empty and not annot_df.empty
    ), "Counts or annotation dataframes are empty."

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
    annot_df: Path,
    org_db: OrgDB,
    msigdb_cat: str,
    contrast_factor: Optional[str] = None,
    custom_genes_file: Path = None,
    exclude_genes: Iterable[str] = None,
    gsva_threads: int = 8,
    remove_overlapping: bool = False,
) -> Tuple[pd.DataFrame, Iterable[int], Iterable[str], pd.DataFrame]:
    """Transform a expression data matrix from a gene by sample matrix to a
    gene-set by sample matrix.

    Args:
        counts_file: A .csv file containing raw expression data of shape
            [n_genes, n_samples].
        annot_df: A pandas Dataframe containing samples annotations.
        org_db: Organism annotation database.
        msigdb_cat: Category of MSigDB to extract the gene sets from.
        contrast_factor: Column name containing the classes used for classification.
        custom_genes_path: A .csv file where the first column is a list of relevant
            ENSEMBL IDs genes, such as DEGs.
        exclude_genes: ENSEMBL ID genes to remove from data.

    Returns:
        Processed data.
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

    assert (
        not counts_df.empty and not annot_df.empty
    ), "Counts or annotation dataframes are empty."

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
        **{"parallel.sz": gsva_threads},
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
    annot_df: Path,
    contrast_factor: str,
    org_db: OrgDB,
    custom_meth_probes_file: Path,
    exclude_genes: Iterable[str] = None,
) -> Tuple[pd.DataFrame, Iterable[int], Iterable[str], pd.DataFrame]:
    """Perform multiple pre-processing on gene expression data.

    Args:
        meth_values_file: A .csv file containing either methylation B-values or M-values
            with shape [#probes, #samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        org_db: Organism annotation database.
        custom_meth_probes_file: A .csv file with pre-selected differentially methylated
            probes, annotated to gene regions.
        exclude_genes: ENTREZ ID genes to remove from data. Only has an effect if
            `custom_meth_probes_file` is provided.

    Returns:
        Processed data.
    """
    # 1. Data
    # 1.1. Loading
    meth_values_df = pd.read_csv(meth_values_file, index_col=0).transpose()

    assert (
        not meth_values_df.empty and not annot_df.empty
    ), "Methylation values file or annotation file are empty."

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
    annot_df: Path,
    contrast_factor: str,
    custom_gene_sets_file: Path = None,
    exclude_gene_sets: Iterable[str] = None,
) -> Tuple[pd.DataFrame, Iterable[int], pd.Series, pd.DataFrame]:
    """Perform multiple pre-processing on gene sets enrichment data.

    Args:
        counts_file: A .csv file containing GSVA enrichment scores data of shape
            [n_gene_sets, n_samples]
        annot_df: A pandas Dataframe containing samples annotations.
        contrast_factor: Column name containing the classes used for classification.
        custom_gene_sets_file: A .csv file where the first column represent gene set
            names.
        exclude_gene_sets: Gene set names to remove from the data.

    Returns:
        Processed data.
    """
    # 1. Data
    # 1.1. Loading
    data = data.transpose()

    assert (
        not data.empty and not annot_df.empty
    ), "Counts or annotation dataframes are empty."

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
