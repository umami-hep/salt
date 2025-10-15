import logging
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Condor Handler")


class CondorHandler:
    """A class to submit batch jobs to a HTCondor scheduler.

    Parameters
    ----------
    batch_path : str
        Path where the batch config files which are created will be stored.
    log_path : str
        Path where the batch log files will be stored.
    base_dir : str
        Directory in which batch job will execute its command.
    """

    def __init__(self, batch_path: str, log_path: str, base_dir: str) -> None:
        self.batch_path = Path(batch_path)
        self.log_path = Path(log_path)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._tag = "salt_job"
        # Keywords to be used in HTCondor configuration file
        self._condor_options_dict = {
            "universe": "Universe",
            "jobflavour": "+JobFlavour",
            "project": "+MyProject",
            "runtime": "+RequestRuntime",
            "memory": "Request_Memory",
            "cpu": "Request_CPUs",
            "gpu": "Request_GPUs",
            "requirements": "Requirements",
            "container": "+MySingularityImage",
            "containerargs": "+MySingularityArgs",
        }
        self._condor_options: dict[str, Any] = {}
        self._test_mode = False

    def activate_testmode(self) -> None:
        """Activate the testmode for submission."""
        logger.debug("Activated test mode: not submitting any jobs.")
        self._test_mode = True

    def deactivate_testmode(self) -> None:
        """Deactivate the testmode for submission."""
        logger.debug("Deactivated test mode: submitting jobs.")
        self._test_mode = False

    def send_job(self, command: str, tag: str = "salt_job") -> None:
        """Send the job to the HTCondor.

        Parameters
        ----------
        command : str
            Command that is run
        tag : str, optional
            Tag for the job, by default "salt_job"
        """
        self._tag = tag
        bashfile = self._make_bash_file(command)
        jobfile = self._make_job_file(bashfile)
        if self._test_mode:
            logger.debug(f"Created job file {jobfile}")
        else:
            subprocess.call(f"condor_submit {jobfile}", shell=True)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value for a key in the condor options dict.

        Parameters
        ----------
        key : str
            Key for the value that is changed
        value : Any
            New value for the key
        """
        self._condor_options[key] = value

    def _make_bash_file(self, command: str) -> Path:
        """Make the batch file which is run.

        Parameters
        ----------
        command : str
            Command that will be run

        Returns
        -------
        Path
            Path to the output batch file
        """
        run_file = self.batch_path / f"batch_{self._tag}.sh"
        with run_file.open("w") as fr:
            fr.write(
                f"""#!/bin/sh
# {self._tag} batch run script
#$ -cwd
#$ -j y
#$ -l cvmfs
BASEDIR={self.base_dir}
pwd; ls -l
{command}
ls -l
"""
            )
        run_file.chmod(0o755)
        logger.debug(f"Made run file {run_file}")
        return run_file

    def _make_job_file(self, run_file: Path) -> Path:
        """Make the job options file for condor.

        Parameters
        ----------
        run_file : Path
            Path to the file that is executed

        Returns
        -------
        Path
            Path to the job options file
        """
        batch_file = self.batch_path / f"batch_{self._tag}.job"
        with batch_file.open("w") as fs:
            for key, value in self._condor_options.items():
                if key in self._condor_options_dict:
                    fs.write(f"{self._condor_options_dict[key]}={value}\n")
            fs.write(
                f"""Executable          = {run_file}
Output              = {self.log_path}/stdout_{self._tag}_$(ClusterId).txt
Error               = {self.log_path}/stderr_{self._tag}_$(ClusterId).txt
Log                 = {self.log_path}/batch_{self._tag}_$(ClusterId).log

queue
"""
            )
        logger.debug(f"Made job file {batch_file}")
        return batch_file
