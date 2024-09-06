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
