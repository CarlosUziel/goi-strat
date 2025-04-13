"""Run a series of Python scripts in batches, submitted as SLURM jobs.

This script configures and submits a batch of bioinformatics analysis scripts to a SLURM
cluster. It defines batches of related Python scripts (grouped by analysis type) and submits
each batch as a separate SLURM job.

The script configures:
1. Common parameters for all scripts (paths, thread counts)
2. SLURM cluster directives (nodes, partitions, QoS settings)
3. Batch definitions - groups of scripts that should run together

Each batch is submitted using the submit_batches utility function.

Example:
    This script can be executed directly:

    $ python batched_scripts.py

Note:
    All paths are configured for the specific WCDT (West Coast Dream Team) dataset
    analysis on a specific SLURM cluster environment.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, Union

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from slurm.utils import submit_batches

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


ROOT_PATH: Path = Path("/gpfs/data/fs71358/cperez")
STORAGE_ROOT: Path = ROOT_PATH.joinpath("storage")
SRC_ROOT: Path = ROOT_PATH.joinpath("goi-strat").joinpath("src")
COMMON_KWARGS: Dict[str, str] = {"--root-dir": STORAGE_ROOT, "--threads": 96}
SLURM_KWARGS: Dict[str, Union[str, int]] = {
    "--nodes": 2,
    "--partition": "skylake_0384",
    "--qos": "skylake_0384",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
LOGS_PATH: Path = ROOT_PATH.joinpath("logs").joinpath("goi-strat")

GOI_BATCHES: Dict[str, Iterable[Path]] = {
    "methylkit": (
        SRC_ROOT.joinpath("pipelines")
        .joinpath("differential_methylation")
        .joinpath("run")
        .joinpath("wcdt")
        .joinpath("goi_methylkit.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("functional_analysis")
        .joinpath("run")
        .joinpath("rrbs")
        .joinpath("wcdt")
        .joinpath("goi_methylkit.py"),
    ),
    "biseq": (
        SRC_ROOT.joinpath("pipelines")
        .joinpath("differential_methylation")
        .joinpath("run")
        .joinpath("wcdt")
        .joinpath("goi_biseq.py"),
        SRC_ROOT.joinpath("pipelines")
        .joinpath("functional_analysis")
        .joinpath("run")
        .joinpath("rrbs")
        .joinpath("wcdt")
        .joinpath("goi_biseq.py"),
    ),
}


submit_batches(
    batches=GOI_BATCHES,
    src_path=SRC_ROOT,
    logs_path=LOGS_PATH,
    common_kwargs=COMMON_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
)
