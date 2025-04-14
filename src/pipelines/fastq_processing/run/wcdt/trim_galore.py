"""
Trim Galore quality and adapter trimming script for WCDT datasets.

This script processes FASTQ files from the WCDT-MCRPC_MethArray dataset to trim adapter
sequences and low-quality bases using Trim Galore. It is configured to run on a SLURM
cluster with specific resource allocations.

Trim Galore is a wrapper around Cutadapt and FastQC that consistently applies quality and
adapter trimming to FastQ files, with additional functionality for specific sequencing
applications like RRBS and bisulfite sequencing. This script processes paired-end FASTQ
files, trimming adapters and removing low-quality sequences.

The script sets up logging configuration, defines paths and parameters for Trim Galore
execution, and then calls the run_trim_galore utility function to process the files.

Usage:
    python trim_galore.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    FASTQ_PATH: Path to raw FASTQ files to be trimmed
    TRIM_GALORE_PATH: Path where trimmed FASTQ files will be stored
    TRIM_GALORE_KWARGS: Trim Galore command line parameters
        --cores: Number of CPU cores to use
        --quality: Phred quality score threshold for trimming
        --stringency: Minimum overlap between adapter and read
        --length: Minimum read length to keep after trimming
        --clip_R1: Trim bases from 5' end of read 1
        --clip_R2: Trim bases from 5' end of read 2
        --three_prime_clip_R1: Trim bases from 3' end of read 1
        --three_prime_clip_R2: Trim bases from 3' end of read 2
        --dont_gzip: Do not compress output files (they will be compressed later)
        --rrbs: Apply RRBS-specific trimming
        --non_directional: Data was treated in non-directional manner
    SLURM_KWARGS: SLURM job submission parameters
    PATTERN: Pattern to match FASTQ files

Notes:
    This script is specifically configured for the WCDT dataset and the Skylake
    partition of a specific SLURM cluster. Parameters may need adjustment for
    different environments or datasets.

    The parameters are optimized for bisulfite sequencing data with RRBS protocol
    in non-directional mode.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_trim_galore

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
FASTQ_PATH: Path = STORAGE.joinpath("fastq_raw")
TRIM_GALORE_PATH: Path = STORAGE.joinpath("trim_galore")
TRIM_GALORE_KWARGS: Dict[str, Any] = {
    "--cores": 62,
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "TRIM_GALORE",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*.fastq.gz"

run_trim_galore(
    fastq_path=FASTQ_PATH,
    trim_galore_path=TRIM_GALORE_PATH,
    trim_galore_kwargs=TRIM_GALORE_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
