import argparse
import datetime
import logging
import multiprocessing
import warnings
from pathlib import Path
from typing import Iterable

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import tcga_rna_seq

_ = traceback.install()
rpy2_logger.setLevel(logging.ERROR)
logging.basicConfig(force=True)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root-dir",
    type=str,
    help="Root directory",
    nargs="?",
    default="/mnt/d/phd_data",
)
parser.add_argument(
    "--threads",
    type=int,
    help="Number of threads for parallel processing",
    nargs="?",
    default=multiprocessing.cpu_count() - 2,
)

user_args = vars(parser.parse_args())
STORAGE: Path = Path(user_args["root_dir"])
# top TCGA projects by sample size
TCGA_NAMES: Iterable[str] = [
    "TCGA-BRCA",  # >1000 RNA-seq samples
    "TCGA-LUAD",  # 517 RNA-seq samples
    "TCGA-THCA",  # 505 RNA-seq samples
    "TCGA-UCEC",  # 557 RNA-seq samples
    "TCGA-LUSC",  # 501 RNA-seq samples
    "TCGA-KIRC",  # 533 RNA-seq samples
    "TCGA-HNSC",  # 512 RNA-seq samples
    "TCGA-LGG",  # 516 RNA-seq samples
]

# A lot of data needs to be downloaded per project, don't parallelize
n_projects = len(TCGA_NAMES)
for i, tcga_dataset in enumerate(TCGA_NAMES):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}][{i}/{n_projects - 1}] Downloading {tcga_dataset}...")

    data_path = STORAGE.joinpath(tcga_dataset).joinpath("data")
    data_path.mkdir(parents=True, exist_ok=True)
    counts_path = data_path.joinpath("star_counts")

    tcga_rna_seq(
        project_name=tcga_dataset, data_path=data_path, counts_path=counts_path
    )
