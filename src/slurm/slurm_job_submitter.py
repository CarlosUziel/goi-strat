from pathlib import Path
from typing import Dict, Union

from pydantic import BaseModel

from data.utils import run_cmd


class SlurmJobSubmitter(BaseModel):
    """
    Submit jobs to a slurm system

    Args:
        metadata: Contains slurm metadata to run a job. Only the keyword
            argument (long format) and the value is needed, the rest is
            formatted automatically. Each key-value pair will be formatted
            as follows:

                {'--ntasks-per-node': '28'}  --> #SBATCH --ntasks-per-node=28

        tmp_dir: Where to temporarily store the script to be run by slurm.
            It is automatically deleted after the job has been submitted.
    """

    metadata: Dict[str, str]
    tmp_dir: Path = Path("/tmp")

    def submit(self, command: str, log_id: Union[str, int], log_path: Path = None):
        """
            Submit job to slurm.

        Args:
            command: command to run, typically a call to python or bash script.
            log_id: used to uniquely identify logs that run with the same
                metadata
            log_path: Optionally, add a path to store stdout and stderr.
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
