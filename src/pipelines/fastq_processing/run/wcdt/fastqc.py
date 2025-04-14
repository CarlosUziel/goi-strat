"""
FastQC quality control script for WCDT datasets.

This script runs FastQC on the trimmed FASTQ files from the WCDT-MCRPC_MethArray dataset.
It is configured to run on a SLURM cluster with specific resource allocations and
analyzes the quality of trimmed reads (files processed by trim_galore) using FastQC.

FastQC performs quality control checks on raw sequence data coming from high-throughput
sequencing pipelines. It provides a modular set of analyses which you can use to give a
quick impression of whether your data has any problems of which you should be aware
before doing any further analysis.

The script sets up logging configuration, defines paths and parameters for FastQC
execution, and then calls the run_fastqc utility function to process the files.

Usage:
    python fastqc.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    FASTQ_PATH: Path to trimmed FASTQ files (output from trim_galore)
    FASTQC_PATH: Path where FastQC results will be stored
    FASTQC_KWARGS: FastQC command line parameters (e.g., thread count)
    SLURM_KWARGS: SLURM job submission parameters
    PATTERN: Pattern to match trimmed FASTQ files

Notes:
    This script is specifically configured for the WCDT dataset and the Skylake
    partition of a specific SLURM cluster. Parameters may need adjustment for
    different environments or datasets.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_fastqc

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
FASTQ_PATH: Path = STORAGE.joinpath("trim_galore")
FASTQC_PATH: Path = STORAGE.joinpath("fastqc_clean")
FASTQC_KWARGS: Dict[str, Any] = {
    "--threads": 96,
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "FASTQC",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*val*.fq.gz"

run_fastqc(
    fastq_path=FASTQ_PATH,
    fastqc_path=FASTQC_PATH,
    fastqc_kwargs=FASTQC_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
