import logging
import warnings
from pathlib import Path
from typing import Any, Dict

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.fastq_processing.utils import run_bismark_mapping

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

STORAGE: Path = Path("/gpfs/data/fs71358/cperez/storage/WCDT_MCRPC_MethArray")
FASTQ_PATH: Path = STORAGE.joinpath("trim_galore")
GENOME_PATH: Path = STORAGE.joinpath(
    "/gpfs/data/fs71358/cperez/storage/genomes/Homo_sapiens/GRCh38/ENSEMBL"
)
BISMARK_PATH: Path = STORAGE.joinpath("bismark")
BISMARK_KWARGS: Dict[str, Any] = {
    "--parallel": 10,
    "--gzip": "",
    "--un": "",
    "--ambiguous": "",
    "--nucleotide_coverage": "",
}
SLURM_KWARGS: Dict[str, Any] = {
    "--job-name": "BISMARK_MAPPING",
    "--nodes": 1,
    "--partition": "skylake_0384",
    "--qos": "skylake_0384",
    "--ntasks-per-node": 48,
    "--ntasks-per-core": 2,
}
PATTERN: str = "**/*val*.fq.gz"

run_bismark_mapping(
    fastq_path=FASTQ_PATH,
    genome_path=GENOME_PATH,
    bismark_path=BISMARK_PATH,
    bismark_kwargs=BISMARK_KWARGS,
    slurm_kwargs=SLURM_KWARGS,
    pattern=PATTERN,
)
