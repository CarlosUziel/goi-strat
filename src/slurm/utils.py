from datetime import date
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable

from tqdm.rich import tqdm

from slurm.slurm_job_submitter import SlurmJobSubmitter


def submit_batches(
    batches: Dict[str, Iterable[Path]],
    src_path: Path,
    logs_path: Path,
    common_kwargs: Dict[str, str],
    slurm_kwargs: Dict[str, str],
):
    """
    Given an iterable of batches of python scripts, submit each to a slurm node.

    Args:
        batches: An iterable of python scripts batches.
        src_path: Python path and parent of scripts.
        logs_path: Where to store logs.
        common_kwargs: Common scripts kwargs.
        slurm_kwargs: A dictionary of SLURM cluster batch job options.
    """
    # 0. Setup
    logs_root = logs_path.joinpath(date.today().strftime("%Y_%m_%d"))

    # 1. Process each batch of scripts
    for batch_name, batch_scripts in tqdm(batches.items()):
        batch_logs_path = logs_root.joinpath(batch_name)
        batch_logs_path.mkdir(exist_ok=True, parents=True)

        slurm_kwargs["--job-name"] = batch_name
        slurm_kwargs["--error"] = str(batch_logs_path.joinpath("error.log"))
        slurm_kwargs["--output"] = str(batch_logs_path.joinpath("output.log"))

        # 3.2. Build command
        command = "\n".join(
            (
                f"PYTHONPATH={str(src_path)}",
                *[
                    " ".join(
                        list(
                            map(
                                str,
                                (
                                    "python",
                                    src_path.joinpath(script),
                                    *list(chain(*common_kwargs.items())),
                                ),
                            )
                        )
                    )
                    for script in batch_scripts
                ],
            )
        )

        # 3.3. Launch SLURM job
        log_file = batch_logs_path.joinpath("sbatch.log")
        SlurmJobSubmitter(metadata=slurm_kwargs, tmp_dir=batch_logs_path).submit(
            command=command, log_id=1, log_path=log_file
        )
