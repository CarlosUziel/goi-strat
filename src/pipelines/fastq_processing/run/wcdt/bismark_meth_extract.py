"""
Bismark methylation extraction script for WCDT datasets.

This script extracts methylation information from Bismark-aligned BAM files from the
WCDT-MCRPC_MethArray dataset. It processes the alignment files to identify methylated
and unmethylated cytosines, generating various output formats for downstream
methylation analysis.

This step typically follows the bismark_mapping.py process and represents the final
stage in the Bismark workflow. The extracted methylation data can be used for
differential methylation analysis, visualization, and other epigenetic analyses.

The script sets up logging configuration, defines paths and parameters for the Bismark
methylation extractor, and then calls the run_bismark_meth_extract utility function
to process the alignment files.

Usage:
    python bismark_meth_extract.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    GENOME_PATH: Path to the reference genome directory for Homo sapiens (GRCh38)
    BISMARK_PATH: Path to Bismark alignment results and where methylation data will be stored
    BISMARK_KWARGS: Bismark methylation extraction command line parameters
        --parallel: Number of instances to run in parallel
        --buffer_size: Memory buffer size for sorting
        -s: Single-end mode
        --bedGraph: Output methylation data in bedGraph format
        --merge_non_CpG: Merge all non-CpG contexts
        --report: Generate extraction report
        --counts: Report conversion counts
        --cytosine_report: Generate cytosine report
        --gzip: Compress output files
        --gazillion: Process very large files
    SLURM_KWARGS: SLURM job submission parameters
    PATTERN: Pattern to match Bismark BAM files

Notes:
    This script is specifically configured for the WCDT dataset and the Skylake
    partition of a specific SLURM cluster. Parameters may need adjustment for
    different environments or datasets.

    Methylation extraction can be memory-intensive, especially for large genomes
    and high-coverage data. The script is configured to use a large buffer size
    and multiple computing nodes to handle this workload.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_bismark_meth_extract

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
GENOME_PATH: Path = STORAGE.joinpath(
    "/gpfs/data/fs71358/cperez/storage/genomes/Homo_sapiens/GRCh38/ENSEMBL"
)
BISMARK_PATH: Path = STORAGE.joinpath("bismark")
BISMARK_KWARGS: Dict[str, Any] = {
    "--parallel": 64,
    "--buffer_size": "160G",
    "-s": "",
    "--bedGraph": "",
    "--merge_non_CpG": "",
    "--report": "",
    "--counts": "",
    "--cytosine_report": "",
    "--gzip": "",
    "--gazillion": "",
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "BISMARK_METH_EXTRACT",
    "--nodes": 2,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*bam"

run_bismark_meth_extract(
    genome_path=GENOME_PATH,
    bismark_path=BISMARK_PATH,
    bismark_kwargs=BISMARK_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
