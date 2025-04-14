"""
Bismark mapping script for WCDT datasets.

This script maps trimmed bisulfite-treated sequencing reads from the WCDT-MCRPC_MethArray
dataset to a reference genome using Bismark. It is configured to run on a SLURM cluster
with specific resource allocations.

Bismark is a specialized aligner for bisulfite sequencing data that performs both read
mapping and methylation calling. This script uses the trimmed reads (output from trim_galore)
as input for the mapping process.

The script sets up logging configuration, defines paths and parameters for Bismark
execution, and then calls the run_bismark_mapping utility function to process the files.

Usage:
    python bismark_mapping.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    FASTQ_PATH: Path to trimmed FASTQ files (output from trim_galore)
    GENOME_PATH: Path to the reference genome directory for Homo sapiens (GRCh38)
    BISMARK_PATH: Path where Bismark mapping results will be stored
    BISMARK_KWARGS: Bismark command line parameters
        --parallel: Number of alignment instances to run in parallel
        --gzip: Compress output files
        --un: Output unmapped reads
        --ambiguous: Output reads that map to multiple locations
        --nucleotide_coverage: Generate nucleotide coverage report
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

from pipelines.fastq_processing.utils import run_bismark_mapping

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
FASTQ_PATH: Path = STORAGE.joinpath("trim_galore")
GENOME_PATH: Path = STORAGE.joinpath(
    "/gpfs/data/fs71358/cperez/storage/genomes/Homo_sapiens/GRCh38/ENSEMBL"
)
BISMARK_PATH: Path = STORAGE.joinpath("bismark")
BISMARK_KWARGS: Dict[str, Any] = {
    "--parallel": 10,
    "--gzip": "",
    "--un": "",
    "--ambiguous": "",
    "--nucleotide_coverage": "",
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "BISMARK_MAPPING",
    "--nodes": 1,
    "--partition": "skylake_0384",
    "--qos": "skylake_0384",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*val*.fq.gz"

run_bismark_mapping(
    fastq_path=FASTQ_PATH,
    genome_path=GENOME_PATH,
    bismark_path=BISMARK_PATH,
    bismark_kwargs=BISMARK_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
