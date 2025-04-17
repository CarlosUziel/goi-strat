"""
Script to download RNA-seq data from the TCGA-PRAD (Prostate Adenocarcinoma) dataset.

This script automates the process of downloading RNA-seq data from The Cancer
Genome Atlas Prostate Adenocarcinoma (TCGA-PRAD) project using the GDC API.
It focuses on primary tumor and normal tissue samples.

The script performs the following steps:
1. Sets up the necessary directories for data storage
2. Downloads gene expression count data from TCGA using the GDC API
3. Processes the downloaded files to extract and organize the STAR counts
4. Saves the processed data in a structured format for downstream analysis

The downloaded data includes both primary tumor and normal tissue samples,
which are essential for differential expression analysis in prostate cancer studies.

Usage:
    python download_data.py [--root-dir ROOT_DIR] [--processes NUM_PROCESSES]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --processes: Number of processes for parallel processing (default: CPU count - 2)
"""

import argparse
import logging
import multiprocessing
import warnings
from pathlib import Path

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import tcga_rna_seq

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
    "--processes",
    type=int,
    help="Number of processes for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
DATA_ROOT: Path = STORAGE.joinpath("TCGA-PRAD_SU2C_RNASeq")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
COUNTS_PATH: Path = DATA_PATH.joinpath("star_counts")

tcga_rna_seq(data_path=DATA_PATH, counts_path=COUNTS_PATH)
