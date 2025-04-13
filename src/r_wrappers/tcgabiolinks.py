"""
Wrappers for R package TCGAbiolinks

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any, Iterable

import pandas as pd
import rpy2
from rpy2.robjects.packages import importr

from r_wrappers.utils import rpy2_df_to_pd_df_manual

r_source = importr("TCGAbiolinks")


def gdc_query(
    project: Iterable[str], data_category: str, **kwargs: Any
) -> rpy2.robjects.vectors.DataFrame:
    """Search for data in the Genomic Data Commons (GDC) database.

    This function uses the GDC API to search for both controlled and open-access
    data from The Cancer Genome Atlas (TCGA) and other projects.

    Args:
        project: TCGA project IDs to query, e.g., ["TCGA-GBM", "TCGA-LGG"].
        data_category: Type of data to query, e.g., "Transcriptome Profiling",
            "Copy Number Variation", "DNA Methylation", etc.
        **kwargs: Additional arguments to pass to the GDCquery function.
            Common parameters include:
            - data_type: Type of data, e.g., "Gene Expression Quantification".
            - workflow_type: Workflow used to analyze the data, e.g., "HTSeq - Counts".
            - legacy: Whether to use legacy data (default: FALSE).
            - access: Data access type, "open" or "controlled".
            - platform: Platform used to generate the data, e.g., "Illumina HiSeq".
            - file_type: File type to query, e.g., "rsem.genes.results".
            - experimental_strategy: Experimental strategy, e.g., "RNA-Seq".
            - sample_type: Sample types to include, e.g., ["Primary Tumor", "Solid Tissue Normal"].
            - barcode: Specific sample barcodes to query.

    Returns:
        rpy2.robjects.vectors.DataFrame: A DataFrame containing the query results.
        This can be passed directly to `gdc_download` to download the data.

    References:
        https://rdrr.io/bioc/TCGAbiolinks/man/GDCquery.html
    """
    return r_source.GDCquery(project=project, data_category=data_category, **kwargs)


def get_query_results(
    query: rpy2.robjects.vectors.DataFrame, **kwargs: Any
) -> pd.DataFrame:
    """Extract results table from a GDC query.

    This function retrieves the results table from a GDC query object, with options
    to select specific columns and limit the number of rows returned.

    Args:
        query: A query object returned by the `gdc_query` function.
        **kwargs: Additional arguments to pass to the getResults function.
            Common parameters include:
            - cols: Columns to select from the results table.
            - rows: Number of rows to return.
            - print: Whether to print the head of the results (default: TRUE).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the query results.

    References:
        https://rdrr.io/bioc/TCGAbiolinks/man/getResults.html
    """
    return rpy2_df_to_pd_df_manual(r_source.getResults(query, **kwargs))


def gdc_download(query: Any, **kwargs: Any) -> None:
    """Download data from the GDC database.

    This function uses the GDC API or GDC transfer tool to download data from
    the Genomic Data Commons. The data will be saved in a folder structure:
    project/data.category.

    Args:
        query: A query object returned by the `gdc_query` function.
        **kwargs: Additional arguments to pass to the GDCdownload function.
            Common parameters include:
            - method: Download method ("api" or "client").
            - directory: Directory to save the downloaded data.
            - files_per_chunk: Number of files to download per chunk.
            - chunks_per_download: Number of chunks to download.
            - progress: Whether to show a progress bar (default: TRUE).

    Returns:
        None: The function downloads files to disk but does not return any value.

    References:
        https://rdrr.io/bioc/TCGAbiolinks/man/GDCdownload.html
    """
    r_source.GDCdownload(query=query, **kwargs)


def gdc_prepare(query: Any, **kwargs: Any) -> rpy2.robjects.methods.RS4:
    """Prepare downloaded GDC data for analysis.

    This function reads the data downloaded by `gdc_download` and prepares it
    into an R object suitable for analysis. The type of the returned object
    depends on the data type that was downloaded.

    Args:
        query: A query object returned by the `gdc_query` function.
        **kwargs: Additional arguments to pass to the GDCprepare function.
            Common parameters include:
            - directory: Directory where the data was downloaded.
            - save: Whether to save the prepared data as an RData file.
            - save_filename: Name of the file to save the data.
            - summarizedExperiment: Whether to return a SummarizedExperiment object.
            - remove_duplicate_keys: Whether to remove duplicate samples.
            - dictionary: Gene ID conversion dictionary.

    Returns:
        rpy2.robjects.methods.RS4: An R object containing the prepared data.
        The exact type depends on the type of data downloaded:
        - RNA-Seq: SummarizedExperiment or RangedSummarizedExperiment
        - Methylation: GenomicRatioSet
        - ATAC-Seq: RangedSummarizedExperiment
        - miRNA-Seq: SummarizedExperiment or data.frame
        - Copy number: RaggedExperiment

    References:
        https://rdrr.io/bioc/TCGAbiolinks/man/GDCprepare.html
    """
    return r_source.GDCprepare(query=query, **kwargs)
