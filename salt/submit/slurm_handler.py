import logging
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)


class SlurmHandler:
    """A class to submit batch jobs to a Slurm scheduler.

    Attributes
    ----------
    batch_path : Path
        Path where the batch file which is created will be stored.
    log_path : Path
        Path where the batch log files will be stored.
    base_dir : Path
        Directory in which batch job will execute its command.

    Methods
    -------
    activate_testmode():
        Activate test mode: check config files in dry runs, no jobs submitted.
    deactivate_testmode():
        Deactivate test mode, enable submitting jobs.
    send_job(command: str, tag: str = "slurm_job"):
        Submit job by creating and executing Slurm batch file
    """

    def __init__(self, batch_path: str, log_path: str, basedir: str) -> None:
        self.batch_path = Path(batch_path)
        self.log_path = Path(log_path)
        self.base_dir = Path(basedir) if basedir else Path.cwd()
        self._tag = "salt_job"
        # Keywords to be used in Slurm configuration
        self._slurm_options_dict: dict[str, Any] = {}
        self._test_mode = False

    def activate_testmode(self) -> None:
        logging.debug("Activated test mode: not submitting any jobs.")
        self._test_mode = True

    def deactivate_testmode(self) -> None:
        logging.debug("Deactivated test mode: submitting jobs.")
        self._test_mode = False

    def send_job(self, command: str, tag: str = "salt_job") -> None:
        self._tag = tag
        batchfile = self._make_batch_file(command)
        if self._test_mode:
            logging.debug(f"Created batch file {batchfile}")
        else:
            subprocess.call(f"sbatch {batchfile}", shell=True)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        self._slurm_options_dict[key] = value

    def _make_batch_file(self, command: str) -> Path:
        batch_file = self.batch_path / f"sbatch_{self._tag}.sh"
        with batch_file.open("w") as bf:
            bf.write(f"""#!/bin/sh
# {self._tag} batch run script\n""")
            for key, value in self._slurm_options_dict.items():
                if value is None:
                    bf.write(f"#SBATCH --{key}\n")
                else:
                    bf.write(f"#SBATCH --{key}={value}\n")
            bf.write(f"""BASEDIR={self.base_dir};pwd; ls -l\n""")
            bf.write(f"""{command}""")
        batch_file.chmod(0o755)
        logging.debug(f"Made batch file {batch_file}")
        return batch_file
