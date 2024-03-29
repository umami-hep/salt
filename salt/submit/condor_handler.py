import logging
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)


class CondorHandler:
    """A class to submit batch jobs to a HTCondor scheduler.

    Attributes
    ----------
    batch_path : Path
        Path where the batch config files which are created will be stored.
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
    send_job(command: str, tag: str = "htcondor_job"):
        Submit job by creating config files (bash file and HTCondor submission file)
        and executing condor_submit.
    """

    def __init__(self, batch_path: str, log_path: str, basedir: str) -> None:
        self.batch_path = Path(batch_path)
        self.log_path = Path(log_path)
        self.base_dir = Path(basedir) if basedir else Path.cwd()
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
        logging.debug("Activated test mode: not submitting any jobs.")
        self._test_mode = True

    def deactivate_testmode(self) -> None:
        logging.debug("Deactivated test mode: submitting jobs.")
        self._test_mode = False

    def send_job(self, command: str, tag: str = "salt_job") -> None:
        self._tag = tag
        bashfile = self._make_bash_file(command)
        jobfile = self._make_job_file(bashfile)
        if self._test_mode:
            logging.debug(f"Created job file {jobfile}")
        else:
            subprocess.call(f"condor_submit {jobfile}", shell=True)

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        self._condor_options[key] = value

    def _make_bash_file(self, command: str) -> Path:
        run_file = self.batch_path / f"batch_{self._tag}.sh"
        with run_file.open("w") as fr:
            fr.write(f"""#!/bin/sh
# {self._tag} batch run script
#$ -cwd
#$ -j y
#$ -l cvmfs
BASEDIR={self.base_dir}
pwd; ls -l
{command}
ls -l
""")
        run_file.chmod(0o755)
        logging.debug(f"Made run file {run_file}")
        return run_file

    def _make_job_file(self, run_file: Path) -> Path:
        batch_file = self.batch_path / f"batch_{self._tag}.job"
        with batch_file.open("w") as fs:
            for key, value in self._condor_options.items():
                if key in self._condor_options_dict:
                    fs.write(f"{self._condor_options_dict[key]}={value}\n")
            fs.write(f"""Executable          = {run_file}
Output              = {self.log_path}/stdout_{self._tag}_$(ClusterId).txt
Error               = {self.log_path}/stderr_{self._tag}_$(ClusterId).txt
Log                 = {self.log_path}/batch_{self._tag}_$(ClusterId).log

queue
""")
        logging.debug(f"Made job file {batch_file}")
        return batch_file
