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
