"""
Bismark genome preparation script for WCDT datasets.

This script prepares a reference genome for Bismark alignment of bisulfite-treated
sequencing reads. It creates the bisulfite-converted reference genomes and indexes
needed for the alignment process. This is a prerequisite step that must be completed
before running the bismark_mapping.py script.

Bismark genome preparation converts the reference genome to both C→T and G→A versions
to allow alignment of bisulfite-converted reads. The process generates index files
that are later used by Bismark during the mapping step.

The script sets up logging configuration, defines paths and parameters for the Bismark
genome preparation, and then calls the run_bismark_genome utility function to process
the reference genome.

Usage:
    python bismark_genome_prep.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    GENOME_PATH: Path to the reference genome directory for Homo sapiens (GRCh38)
    BISMARK_GENOME_KWARGS: Bismark genome preparation command line parameters
        --parallel: Number of instances to run in parallel
    SLURM_KWARGS: SLURM job submission parameters

Notes:
    This script is specifically configured for the WCDT dataset and the Skylake
    partition of a specific SLURM cluster. Parameters may need adjustment for
    different environments or datasets.

    Genome preparation can be computationally intensive and memory-demanding,
    especially for large genomes like human. Ensure adequate resources are
    allocated for this process.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_bismark_genome

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
GENOME_PATH: Path = STORAGE.joinpath(
    "/gpfs/data/fs71358/cperez/storage/genomes/Homo_sapiens/GRCh38/ENSEMBL"
)
BISMARK_GENOME_KWARGS: Dict[str, Any] = {
    "--parallel": 96,
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "BISMARK_GENOME",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}

run_bismark_genome(
    genome_path=GENOME_PATH,
    bismark_genome_kwargs=BISMARK_GENOME_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
)
