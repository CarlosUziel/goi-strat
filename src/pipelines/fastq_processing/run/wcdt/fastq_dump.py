"""
SRA to FASTQ conversion script for WCDT datasets.

This script converts SRA (Sequence Read Archive) files from the WCDT-MCRPC_MethArray
dataset to FASTQ format using fasterq-dump. It is configured to run on a SLURM cluster
with specific resource allocations.

fasterq-dump is an optimized version of the fastq-dump tool from the SRA Toolkit that
extracts sequence data from SRA-formatted files and converts them into FASTQ format.
This script processes the SRA files and compresses the resulting FASTQ files using pigz.

The script sets up logging configuration, defines paths and parameters for fasterq-dump
execution, and then calls the run_fasterq_dump utility function to process the files.

Usage:
    python fastq_dump.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    SRA_PATH: Path to SRA files to be converted
    NGC_FILEPATH: Path to the NGC file containing authentication credentials
                  for accessing protected data
    FASTQ_PATH: Path where extracted FASTQ files will be stored
    FASTERQ_DUMP_KWARGS: fasterq-dump command line parameters
        --split-files: Generate separate files for paired-end reads
        --include-technical: Include technical reads
    PIGZ_KWARGS: pigz compression command line parameters
        -p: Number of threads to use for compression
    SLURM_KWARGS: SLURM job submission parameters
    PATTERN: Pattern to match SRA files

Notes:
    This script is specifically configured for the WCDT dataset and the Skylake
    partition of a specific SLURM cluster. Parameters may need adjustment for
    different environments or datasets.

    The NGC file is required for accessing controlled-access data from repositories
    like dbGaP.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_fasterq_dump

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
SRA_PATH: Path = STORAGE.joinpath("sra")
NGC_FILEPATH: Path = SRA_PATH.joinpath("ncbi").joinpath("prj_22525.ngc")
FASTQ_PATH: Path = STORAGE.joinpath("fastq_raw")

FASTERQ_DUMP_KWARGS: Dict[str, Any] = {
    "--threads": 96,
    "--mem": "1G",
    "--split-files": "",
}
PIGZ_KWARGS: Dict[str, Any] = {
    "-p": 96,
    "-9": "",
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "fasterq_dump",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}

run_fasterq_dump(
    sra_path=SRA_PATH,
    fastq_path=FASTQ_PATH,
    ngc_filepath=NGC_FILEPATH,
    fasterq_dump_kwargs=FASTERQ_DUMP_KWARGS,
    pigz_kwargs=PIGZ_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
)
