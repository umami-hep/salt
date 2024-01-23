import argparse
from pathlib import Path

from condor_handler import CondorHandler

# Set up argument parser
parser = argparse.ArgumentParser(description="Submit batch jobs to HTCondor.")
parser.add_argument("-c", "--config", required=True, type=Path, help="Configuration file for job.")
parser.add_argument("-t", "--tag", default="salt_job", help="Tag for job to be submitted.")
parser.add_argument(
    "-e",
    "--environment",
    default="conda",
    choices=["conda", "singularity", "local"],
    help="Environment for job to be submitted.",
)
args = parser.parse_args()

# Define directories
batch_dir = Path.cwd() / "condor"
batch_path = batch_dir / "batch"
log_path = batch_dir / "batch_logs"
for directory in [batch_path, log_path]:
    directory.mkdir(parents=True, exist_ok=True)

# Set up HTCondor options
job_basedir = Path(__file__).resolve().parent.parent.parent
handler = CondorHandler(str(batch_path), str(log_path), str(job_basedir))
handler["universe"] = "vanilla"
handler["cpu"] = 1
handler["gpu"] = 2
handler["memory"] = 80_000  # 80 GiB - in MiB
handler["runtime"] = 82800  # 23h - in seconds
handler["requirements"] = 'OpSysAndVer == "CentOS7"'

# Run in singularity container?
if args.environment == "singularity":
    handler["container"] = (
        '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/'
        'atlas-flavor-tagging-tools/algorithms/salt:latest"'
    )
    # check host names to determine where to mount storage
    storage_dir = Path("/tmp")  # noqa: S108
    if Path("/etc/hostname").read_text().startswith("lxplus"):
        storage_dir = Path("/eos")
    handler["containerargs"] = f'"--nv --bind {storage_dir}"'

# Construct and submit the job command
command = "cd ${BASEDIR} && export OMP_NUM_THREADS=1 && "
if args.environment == "conda":
    command += (
        "source conda/bin/activate && conda activate salt && "
        'echo "Activated environment ${CONDA_DEFAULT_ENV}" && '
    )
command += (
    'echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" && '
    "cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1 && "
    "cd ${BASEDIR}/salt && pwd && "
    f"salt fit --config {args.config.resolve()}"
)
handler.send_job(command, args.tag)
