from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import BaseModel

from data.utils import run_cmd


class SlurmJobSubmitter(BaseModel):
    """Submit jobs to a SLURM cluster system. :no-index:

    This class provides functionality for submitting jobs to a SLURM cluster system by
    creating temporary script files with appropriate SBATCH directives.

    Args:
        metadata: Contains SLURM metadata to run a job. Only the keyword
            arguments (long format) and their values are needed; the rest is
            formatted automatically. Each key-value pair will be formatted
            as follows:

                {'--ntasks-per-node': '28'}  --> #SBATCH --ntasks-per-node=28

        tmp_dir: Where to temporarily store the script to be run by SLURM.
            The script file is created in this directory.

    Attributes:
        metadata: Dictionary containing SLURM directives as key-value pairs.
        tmp_dir: Path to directory for storing temporary SLURM script files.
    """

    metadata: Dict[str, str]
    tmp_dir: Path = Path("/tmp")

    def submit(
        self, command: str, log_id: Union[str, int], log_path: Optional[Path] = None
    ) -> None:
        """Submit job to SLURM cluster.

        Creates a temporary script file with appropriate SBATCH directives
        and submits it to the SLURM scheduler.

        Args:
            command: Command to run, typically a call to Python or bash script.
            log_id: Identifier used to uniquely identify logs that run with the same
                metadata. Added to the script filename.
            log_path: Optional path to store stdout and stderr logs from the
                sbatch command itself (not the job's output).

        Returns:
            None
        """
        # 0. Create temporary script file
        script_file = self.tmp_dir.joinpath(
            Path(f"{self.metadata['--job-name']}_{log_id}.slrm")
        )

        # 1. Prepare metadata header to file
        metadata_header = "#!/bin/bash\n" + "\n".join(
            [f"#SBATCH {k}={v}" for k, v in self.metadata.items()]
        )

        # 2. Write metadata header plus command
        script_file.write_text(metadata_header + "\n\n" + command.strip())

        # 3. Submit sbatch job
        run_cmd(cmd=["sbatch", str(script_file)], log_path=log_path)

        # 4. Remove temporary script
        # script_file.unlink()
