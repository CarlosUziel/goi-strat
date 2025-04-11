import gzip
import logging
import shutil
from itertools import product
from pathlib import Path

import pandas as pd

from data.utils import parallelize_star


def unzip_gz(gz_file: Path, target_dir: Path):
    """
    Given a root directory, unzip all ".gz" files contained in any level
    of subdirectories matching pattern to the given target directory. If
    target_dir is None, files are unzipped in the same directory where they
    are located.

    Args:
        gz_file: Path to the gzip file to extract
        target_dir: new directory where files are moved to, defaults to parent
          directory of gz_file if no path is provided.

    Note:
        When using pattern matching, "**" must be present in order for the
        search to be recursive. Example: ``**/*.gz``
    """
    # 1. Ensure target dir exists
    target_dir.mkdir(exist_ok=True, parents=True)

    # 2. Extract each file in target directory
    target_dir = target_dir or gz_file.parent
    dest_path = target_dir.joinpath(Path(gz_file).stem)

    with gzip.open(str(gz_file), "rb") as s_file, dest_path.open("wb") as d_file:
        shutil.copyfileobj(s_file, d_file)


def copy_file(file_path: Path, new_file_path: Path):
    """
    Copy a file from one path to another, ensuring that the parents of the new
    path exist.
    """
    new_file_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        shutil.copy(file_path, new_file_path)
    except shutil.SameFileError as e:
        logging.warning(e)


def subset_star_counts(counts_file: Path, subset_col: int = 1):
    """
    Filter a STAR gene counts file by removing unnecessary rows and selecting the
    relevant column.
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


def clean_star_counts(star_path: Path, star_counts_path: Path):
    """
    Copy and rename star counts files after mapping. It is assumed that gene counts
    files are inside directories named after the sample ID.
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


def rename_genes(counts_file: Path):
    """
    Given a counts file in .tsv format, which contain gene names
    in each line, rename them to remove the decimal part of the name.

    Args:
        counts_file: tab-separated counts file.
    """
    df = pd.read_csv(counts_file, sep="\t", header=None, index_col=0)
    df.index = [idx.split(".")[0] for idx in df.index]
    df.to_csv(counts_file, sep="\t", header=False)


def intersect_raw_counts(counts_path: Path, pattern: str = "*.tsv"):
    """
    Replaces the genes contained in the raw count files in `counts_path`
    so that all files contain the intersection set of genes. In other words,
    makes sure that files only contain those genes available in all files.

    Args:
        counts_path: Directory where the files to change can be located.
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

    # 2. Remove any genes with missing values (meaning that are not share among all
    # files)
    counts_data.dropna(inplace=True)

    # 3. Write results
    for c in counts_data.columns:
        counts_data[c].to_csv(c, sep="\t", header=False)
