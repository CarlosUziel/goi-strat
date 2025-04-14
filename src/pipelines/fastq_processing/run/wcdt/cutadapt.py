"""
Cutadapt adapter trimming script for WCDT datasets.

This script processes FASTQ files from the WCDT-MCRPC_MethArray dataset to trim adapter
sequences using Cutadapt. It is configured to run on a SLURM cluster with specific
resource allocations.

Cutadapt is a tool that finds and removes adapter sequences, primers, poly-A tails and
other types of unwanted sequence from high-throughput sequencing reads. This script
processes FASTQ files, trimming adapters according to the sequences specified in the
adapter files.

The script sets up logging configuration, defines paths and parameters for Cutadapt
execution, and then calls the run_cutadapt utility function to process the files.

Usage:
    python cutadapt.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    FASTQ_PATH: Path to raw FASTQ files to be trimmed
    FWD_ADAPTER_FILE: File containing forward adapter sequences
    RV_ADAPTER_FILE: File containing reverse adapter sequences
    CUTADAPT_PATH: Path where trimmed FASTQ files will be stored
    CUTADAPT_KWARGS: Cutadapt command line parameters
        --minimum-length: Minimum length of reads to keep after trimming
    SLURM_KWARGS: SLURM job submission parameters
    PATTERN: Pattern to match FASTQ files

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

from pipelines.fastq_processing.utils import run_cutadapt

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
FASTQ_PATH: Path = STORAGE.joinpath("fastq_raw")
CUTADAPT_PATH: Path = STORAGE.joinpath("cutadapt")
FWD_ADAPTER_FILE: Path = (
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath("adapter_seqs")
    .joinpath("fwd_adapters.fasta")
)
RV_ADAPTER_FILE: Path = (
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath("adapter_seqs")
    .joinpath("rv_adapters.fasta")
)
CUTADAPT_KWARGS: Dict[str, Any] = {
    "--cores": 96,
    "--minimum-length": 20,
    "--overlap": 20,
    "--nextseq-trim": 10,
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "CUTADAPT",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*.fastq.gz"

run_cutadapt(
    fastq_path=FASTQ_PATH,
    cutadapt_path=CUTADAPT_PATH,
    fwd_adapter_file=FWD_ADAPTER_FILE,
    rv_adapter_file=RV_ADAPTER_FILE,
    cutadapt_kwargs=CUTADAPT_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
