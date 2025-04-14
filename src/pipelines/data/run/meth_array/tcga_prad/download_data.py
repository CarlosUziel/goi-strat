"""
Download and process methylation array data from TCGA-PRAD dataset.

This script automates the retrieval and initial processing of DNA methylation microarray data
from The Cancer Genome Atlas Prostate Adenocarcinoma (TCGA-PRAD) project. It focuses on
Illumina Human Methylation 450K array data, which provides genome-wide methylation profiling
at single-nucleotide resolution.

The script performs the following operations:
1. Sets up the necessary directory structure for storing the methylation array data
2. Downloads .idat files (raw intensity data) from TCGA using GDC API queries
3. Organizes the methylation data files for downstream processing
4. Creates annotation files that match samples with their clinical information
5. Filters samples to retain only those that also have RNA-seq data for integrative analyses

The resulting organized dataset serves as input for various methylation analysis pipelines
including differential methylation analysis and epigenetic feature annotation.

Usage:
    python download_data.py [--root-dir ROOT_DIR] [--threads NUM_THREADS]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --threads: Number of threads for parallel processing (default: CPU count - 2)
"""

import argparse
import logging
import multiprocessing
import warnings
from pathlib import Path

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import tcga_prad_meth_array

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    type=str,
    help="Root directory",
    nargs="?",
    default="/mnt/d/phd_data",
)
parser.add_argument(
    "--threads",
    type=int,
    help="Number of threads for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
DATA_ROOT: Path = STORAGE.joinpath("TCGA-PRAD_MethArray")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
IDAT_PATH: Path = DATA_PATH.joinpath("idat")
IDAT_PATH.mkdir(exist_ok=True, parents=True)

tcga_prad_meth_array(data_path=DATA_PATH, idat_path=IDAT_PATH)
