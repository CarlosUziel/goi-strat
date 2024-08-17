"""
    Run a series of python scripts in batches, submited as SLURM jobs.
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
