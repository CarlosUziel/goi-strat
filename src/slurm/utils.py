from datetime import date
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Union, Any

from tqdm.rich import tqdm

from slurm.slurm_job_submitter import SlurmJobSubmitter


def submit_batches(
    batches: Dict[str, Iterable[Path]],
    src_path: Path,
    logs_path: Path,
    common_kwargs: Dict[str, str],
    slurm_kwargs: Dict[str, Union[str, Any]],
) -> None:
    """Submit batches of Python scripts to a SLURM cluster.

    This function organizes Python scripts into batches and submits each batch as a 
    separate SLURM job. It creates appropriate log directories with date-based 
    organization and configures each job with proper SLURM directives.

    Args:
        batches: Dictionary mapping batch names to iterables of Python script paths.
            Each batch will be submitted as a separate SLURM job.
        src_path: Path to the source code directory, used as the PYTHONPATH and
            parent directory for script paths.
        logs_path: Directory path where log files should be stored.
        common_kwargs: Dictionary of common command-line arguments to pass to all
            Python scripts in all batches.
        slurm_kwargs: Dictionary of SLURM directives to use for job submission
            (e.g., {'--ntasks-per-node': '48', '--partition': 'skylake_0384'}).

    Returns:
        None

    Note:
        For each batch, a new directory is created under logs_path with the current
        date and batch name. SLURM output and error logs are stored in this directory.
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
