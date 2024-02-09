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

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT_MCRPC_MethArray")
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
