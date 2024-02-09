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

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT_MCRPC_MethArray")
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
