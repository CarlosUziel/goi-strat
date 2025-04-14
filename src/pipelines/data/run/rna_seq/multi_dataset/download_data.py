"""
Script to download RNA-seq data from multiple TCGA datasets.

This script automates the process of downloading RNA-seq data from The Cancer
Genome Atlas (TCGA) for multiple cancer types. It focuses on the largest
datasets by sample size for comprehensive analysis.

The script:
1. Sets up the necessary directories for each TCGA dataset
2. Downloads gene expression count data from TCGA using the GDC API
3. Processes the downloaded files to extract and organize the STAR counts
4. Saves the processed data in a structured format for downstream analysis

The downloaded datasets include:
- TCGA-BRCA: Breast Cancer (>1000 RNA-seq samples)
- TCGA-LUAD: Lung Adenocarcinoma (517 RNA-seq samples)
- TCGA-THCA: Thyroid Cancer (505 RNA-seq samples)
- TCGA-UCEC: Uterine Corpus Endometrial Carcinoma (557 RNA-seq samples)
- TCGA-LUSC: Lung Squamous Cell Carcinoma (501 RNA-seq samples)
- TCGA-KIRC: Kidney Renal Clear Cell Carcinoma (533 RNA-seq samples)
- TCGA-HNSC: Head and Neck Squamous Cell Carcinoma (512 RNA-seq samples)
- TCGA-LGG: Lower Grade Glioma (516 RNA-seq samples)

Usage:
    python download_data.py [--root-dir ROOT_DIR] [--threads NUM_THREADS]

Arguments:
    --root-dir: Root directory for data storage (default: /mnt/d/phd_data)
    --threads: Number of threads for parallel processing (default: CPU count - 2)
"""

import argparse
import datetime
import logging
import multiprocessing
import warnings
from pathlib import Path
from typing import Iterable

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
    "--threads",
    type=int,
    help="Number of threads for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
# top TCGA projects by sample size
TCGA_NAMES: Iterable[str] = [
    "TCGA-BRCA",  # >1000 RNA-seq samples
    "TCGA-LUAD",  # 517 RNA-seq samples
    "TCGA-THCA",  # 505 RNA-seq samples
    "TCGA-UCEC",  # 557 RNA-seq samples
    "TCGA-LUSC",  # 501 RNA-seq samples
    "TCGA-KIRC",  # 533 RNA-seq samples
    "TCGA-HNSC",  # 512 RNA-seq samples
    "TCGA-LGG",  # 516 RNA-seq samples
]

# A lot of data needs to be downloaded per project, don't parallelize
n_projects = len(TCGA_NAMES)
for i, tcga_dataset in enumerate(TCGA_NAMES):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}][{i}/{n_projects - 1}] Downloading {tcga_dataset}...")

    data_path = STORAGE.joinpath(tcga_dataset).joinpath("data")
    data_path.mkdir(parents=True, exist_ok=True)
    counts_path = data_path.joinpath("star_counts")

    tcga_rna_seq(
        project_name=tcga_dataset, data_path=data_path, counts_path=counts_path
    )
