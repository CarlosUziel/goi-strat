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

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT_MCRPC_MethArray")
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
