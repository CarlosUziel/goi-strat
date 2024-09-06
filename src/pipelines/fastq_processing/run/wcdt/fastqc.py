import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_fastqc

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT-MCRPC_MethArray")
FASTQ_PATH: Path = STORAGE.joinpath("trim_galore")
FASTQC_PATH: Path = STORAGE.joinpath("fastqc_clean")
FASTQC_KWARGS: Dict[str, Any] = {
    "--threads": 96,
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "FASTQC",
    "--nodes": 1,
    "--partition": "skylake_0096",
    "--qos": "skylake_0096",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*val*.fq.gz"

run_fastqc(
    fastq_path=FASTQ_PATH,
    fastqc_path=FASTQC_PATH,
    fastqc_kwargs=FASTQC_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
