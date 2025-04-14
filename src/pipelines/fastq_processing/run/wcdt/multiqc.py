"""
MultiQC aggregated quality control report script for WCDT datasets.

This script runs MultiQC to aggregate quality control metrics from various analysis steps
of the WCDT-MCRPC_MethArray dataset. It is configured to run on a SLURM cluster with
specific resource allocations.

MultiQC is a tool that searches a given directory for analysis logs and compiles an HTML
report that summarizes all QC metrics from different bioinformatics analyses (FastQC,
Trim Galore, Bismark, etc.) into a single comprehensive report. This allows for easy
visualization and comparison of quality metrics across samples and processing steps.

The script sets up logging configuration, defines paths and parameters for MultiQC
execution, and then calls the run_multiqc utility function to process the files.

Usage:
    python multiqc.py

Configuration:
    STORAGE: Base storage path for the WCDT-MCRPC_MethArray dataset
    MULTIQC_PATH: Path where MultiQC results will be stored
    ANALYSES_PATHS: List of paths containing analysis results to be included in the report,
                   including FastQC results and Bismark alignment results
    MULTIQC_KWARGS: MultiQC command line parameters
        --filename: Name of the output report file
        --title: Title to display in the report
        --interactive: Enable interactive plots in the report
        --no-data-dir: Do not create a data directory alongside the report
    SLURM_KWARGS: SLURM job submission parameters

Notes:
    This script is specifically configured for the WCDT dataset and the Skylake
    partition of a specific SLURM cluster. Parameters may need adjustment for
    different environments or datasets.

    MultiQC aggregates quality reports from multiple locations, making it easier to
    compare QC metrics across different samples and processing steps in a single report.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_multiqc

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
MULTIQC_PATH: Path = STORAGE.joinpath("multiqc_clean")
ANALYSES_PATH: Iterable[Path] = [
    STORAGE.joinpath("fastqc_clean"),
    # STORAGE.joinpath("bismark"),
]
MULTIQC_KWARGS: Dict[str, Any] = {
    "--interactive": "",
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "MULTIQC",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}

run_multiqc(
    multiqc_path=MULTIQC_PATH,
    analyses_paths=ANALYSES_PATH,
    multiqc_kwargs=MULTIQC_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
)
