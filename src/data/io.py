import gzip
import logging
import shutil
from itertools import product
from pathlib import Path

import pandas as pd

from data.utils import parallelize_star


def unzip_gz(gz_file: Path, target_dir: Path) -> None:
    """Extracts a gzip file to a target directory.

    Given a gzip file, extracts its contents to the specified target directory,
    creating the directory if it doesn't exist.

    Args:
        gz_file: Path to the gzip file to extract
        target_dir: Directory where the extracted file will be placed

    Returns:
        None

    Note:
        The extracted file will have the same name as the gzip file without the .gz extension
    """
    # 1. Ensure target dir exists
    target_dir.mkdir(exist_ok=True, parents=True)

    # 2. Extract each file in target directory
    target_dir = target_dir or gz_file.parent
    dest_path = target_dir.joinpath(Path(gz_file).stem)

    with gzip.open(str(gz_file), "rb") as s_file, dest_path.open("wb") as d_file:
        shutil.copyfileobj(s_file, d_file)


def copy_file(file_path: Path, new_file_path: Path) -> None:
    """Copy a file from one path to another.

    Ensures that the parent directories of the destination path exist
    before copying the file.

    Args:
        file_path: Source path of the file to copy
        new_file_path: Destination path where the file will be copied to

    Returns:
        None

    Raises:
        shutil.SameFileError: If source and destination are the same file
    """
    new_file_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        shutil.copy(file_path, new_file_path)
    except shutil.SameFileError as e:
        logging.warning(e)


def subset_star_counts(counts_file: Path, subset_col: int = 1) -> None:
    """Filter a STAR gene counts file by selecting only ENSG entries and relevant columns.

    Reads a STAR counts file, filters rows to keep only those containing "ENSG" in the
    index (Ensembl gene IDs), and keeps only the specified column.

    Args:
        counts_file: Path to the STAR gene counts file
        subset_col: Column index (0-based) to keep in the filtered output, defaults to 1

    Returns:
        None: The input file is modified in-place

    Note:
        The function is idempotent - if the file has already been filtered,
        it will return without changes
    """
    df = pd.read_csv(counts_file, sep="\t", comment="#", index_col=0, header=None)

    if len(df.columns) == 1:
        logging.warning(
            "File only contains index plus one column, this star counts file"
            f" ({counts_file.name}) has already been filtered."
        )
        return

    df = df.loc[[idx for idx in df.index if "ENSG" in idx], subset_col].sort_index()
    df.to_csv(counts_file, sep="\t", header=False)


def clean_star_counts(star_path: Path, star_counts_path: Path) -> None:
    """Process STAR RNA-seq output files by copying and preprocessing count data.

    Copies files with the name "ReadsPerGene.out.tab" from sample directories within star_path
    to star_counts_path, renaming them based on the parent directory (sample ID).
    Then filters the copied files to keep only relevant data.

    Args:
        star_path: Directory containing sample subdirectories with STAR output files
        star_counts_path: Destination directory for the processed counts files

    Returns:
        None

    Note:
        This function assumes the parent directory name of each ReadsPerGene.out.tab
        file corresponds to the sample ID.
    """
    star_counts_path.mkdir(exist_ok=True, parents=True)

    _ = parallelize_star(
        copy_file,
        [
            (counts_file, star_counts_path.joinpath(f"{counts_file.parent.stem}.tsv"))
            for counts_file in star_path.glob("*/ReadsPerGene.out.tab")
        ],
        method="fork",
    )

    _ = parallelize_star(
        subset_star_counts,
        list(product(star_counts_path.glob("*.tsv"), [1])),
        method="fork",
    )


def rename_genes(counts_file: Path) -> None:
    """Remove decimal parts from Ensembl gene IDs in a counts file.

    Reads a tab-separated counts file, removes decimal parts from Ensembl gene IDs
    (e.g., ENSG00000123456.1 becomes ENSG00000123456), and writes the modified data
    back to the original file.

    Args:
        counts_file: Path to the tab-separated counts file

    Returns:
        None: The input file is modified in-place
    """
    df = pd.read_csv(counts_file, sep="\t", header=None, index_col=0)
    df.index = [idx.split(".")[0] for idx in df.index]
    df.to_csv(counts_file, sep="\t", header=False)


def intersect_raw_counts(counts_path: Path, pattern: str = "*.tsv") -> None:
    """Filter gene count files to include only genes common to all files.

    Processes multiple count files to ensure they all contain the same set of genes
    by finding the intersection of genes across all files and updating each file
    to contain only those genes.

    Args:
        counts_path: Directory containing the count files to process
        pattern: File pattern to match when looking for count files, defaults to "*.tsv"

    Returns:
        None: The input files are modified in-place

    Raises:
        AssertionError: If fewer than 2 files are found matching the pattern

    Note:
        The function reads each file into memory, processes them, and then writes
        back to the original files. For large datasets, this may be memory-intensive.
    """
    # 0. Get list of files for each type
    counts_files = list(counts_path.glob(pattern))
    assert len(counts_files) > 1, f"No files found under {counts_path}"

    # 1. Load content of each file
    counts_data = pd.DataFrame(
        {
            counts_file: {
                line.split("\t")[0]: line.split("\t")[1]
                for line in counts_file.read_text().splitlines()
            }
            for counts_file in counts_files
        }
    )

    # 2. Remove any genes with missing values (meaning that are not shared among all
    # files)
    counts_data.dropna(inplace=True)

    # 3. Write results
    for c in counts_data.columns:
        counts_data[c].to_csv(c, sep="\t", header=False)
