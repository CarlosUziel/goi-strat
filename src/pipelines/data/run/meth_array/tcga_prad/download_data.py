import argparse
import logging
import multiprocessing
import warnings
from pathlib import Path

from rich import traceback
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from pipelines.data.utils import tcga_prad_meth_array

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
DATA_ROOT: Path = STORAGE.joinpath("TCGA-PRAD_MethArray")
DATA_PATH: Path = DATA_ROOT.joinpath("data")
IDAT_PATH: Path = DATA_PATH.joinpath("idat")
IDAT_PATH.mkdir(exist_ok=True, parents=True)

tcga_prad_meth_array(data_path=DATA_PATH, idat_path=IDAT_PATH)
