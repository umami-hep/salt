import argparse
from datetime import datetime
from pathlib import Path

from slurm_handler import SlurmHandler

# Set up argument parser
parser = argparse.ArgumentParser(description="Submit batch jobs to Slurm.")
parser.add_argument("-c", "--config", required=True, type=Path, help="Configuration file for job.")
parser.add_argument("-t", "--tag", default="salt_job", help="Tag for job to be submitted.")
parser.add_argument("-p", "--partition", default=None, type=str, help="Partition to submit job.")
parser.add_argument(
    "-cn", "--constraint", default=None, type=str, help="Constraint on requested resources."
)
parser.add_argument("-a", "--account", default=None, type=str, help="Slurm account name.")
parser.add_argument(
    "-e",
    "--environment",
    default="conda",
    choices=["conda", "singularity", "local"],
    help="Environment for job to be submitted.",
)
parser.add_argument("-q", "--qos", default=None, type=str, help="Quality Of Service for job")
parser.add_argument("-n", "--nodes", default=1, type=int, help="Nodes to split training across")
parser.add_argument("-g", "--gpus_per_node", default=1, type=int, help="GPUs for each node")
parser.add_argument(
    "-gt",
    "--gpu_type",
    default="",
    type=str,
    help="GPU type e.g. v100, leave empty for no preference",
)
parser.add_argument("-cpt", "--cpus_per_task", default=10, type=int, help="CPUs for each task")
parser.add_argument("-m", "--memory", default="100G", type=str, help="Memory per node")
parser.add_argument("-ex", "--exclusive", action="store_true")
parser.add_argument("-ti", "--time", default=None, type=str, help="Job time limit e.g. '24:00:00'")
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument(
    "-b",
    "--bind",
    nargs="+",
    help="List of binds for singularity (e.g. /path/to/upp/output:/inputs)",
)
parser.add_argument("-r", "--requeue", action="store_true")
parser.add_argument(
    "-s",
    "--signal",
    default="SIGUSR1@90",
    type=str,
    help="Signal from Slurm to trigger Lightning to prepare for requeue",
)
parser.add_argument(
    "-sls",
    "--salt_log_suffix",
    default=None,
    help="Appended to model name to create Salt log directory",
)
args, extra_args = parser.parse_known_args()

# Manually parse extra args not explicitly handled for passing to Salt and Slurm
extra_slurm_args = {}
extra_config_args = {}
for extra_arg in extra_args:
    dest_argument = extra_arg.lstrip("-")
    dest = dest_argument.split(".")[0]
    argument = ".".join(dest_argument.split(".")[1:])
    if "=" not in argument:
        name = argument
        value = None
    else:
        name, value = argument.split("=")
    if dest == "slurm":
        extra_slurm_args[name] = value
    elif dest == "config":
        extra_config_args[name] = value
    else:
        raise ValueError(f'Extra argument provided but unknown destination "{dest}"')

if args.bind and args.environment != "singularity":
    parser.error("--bind option is only allowed with --environment singularity")

# Define directories
batch_dir = Path.cwd() / "slurm"
batch_path = batch_dir / "batch"
log_path = batch_dir / "batch_logs"
for directory in [batch_path, log_path]:
    directory.mkdir(parents=True, exist_ok=True)

# Variables that need to be harmonized between Slurm and salt
nodes = args.nodes
gpus_per_node = args.gpus_per_node
cpus_per_task = args.cpus_per_task

gpu_type = args.gpu_type
gres = f"gpu:{gpu_type}:{gpus_per_node}" if gpu_type else f"gpu:{gpus_per_node}"

# Set up Slurm options
job_basedir = Path(__file__).resolve().parent.parent.parent
handler = SlurmHandler(str(batch_path), str(log_path), str(job_basedir))
handler["job-name"] = args.tag
if args.partition is not None:
    handler["partition"] = args.partition
if args.constraint is not None:
    handler["constraint"] = args.constraint
if args.account is not None:
    handler["account"] = args.account
if args.qos is not None:
    handler["qos"] = args.qos
handler["nodes"] = nodes
handler["gres"] = gres
handler["ntasks-per-node"] = gpus_per_node
if args.memory is not None:
    handler["mem"] = args.memory  # memory, 100 GiB - in MiB
if args.exclusive:
    handler["exclusive"] = None  # Exclusive access to nodes
handler["cpus-per-task"] = cpus_per_task  # Don't use this if you have exclusive access to the node
handler["export"] = "ALL"
handler["output"] = f"{log_path}/slurm-%j.out"
handler["error"] = f"{log_path}/slurm-%j.err"
if args.time is not None:
    handler["time"] = args.time  # Time limit of job, default is system specified
if args.requeue:
    handler["requeue"] = None
    handler["signal"] = args.signal
# Fill extra Slurm args
for arg, value in extra_slurm_args.items():
    handler[arg] = value


log_suffix = args.salt_log_suffix
if args.requeue and not log_suffix:
    log_suffix = datetime.now().strftime("%Y%m%d-T%H%M%S")

# Construct and submit the job command
command = "cd ${BASEDIR} && export OMP_NUM_THREADS=1\n"
if args.environment == "conda":
    command += (
        "source conda/bin/activate && conda activate salt\n"
        'echo "Activated environment ${CONDA_DEFAULT_ENV}"\n'
    )
elif args.environment == "singularity":
    command += "srun singularity exec -e --nv \\\n"
    command += " \\\n".join([f"--bind {b}" for b in args.bind]) + " \\\n"
    command += (
        "--home ${BASEDIR} \\\n"
        "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/algorithms/salt:latest/ \\\n"
        'sh -c "'
    )
command += (
    "echo 'CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}' &&\n"
    "cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1 &&\n"
    "cd ${BASEDIR}/salt && pwd &&\n"
    + ("srun " if args.environment == "conda" else "")
    + f"salt fit --config {args.config.resolve()} \\\n"
    f"      --trainer.devices={gpus_per_node} \\\n"
    f"      --trainer.num_nodes={nodes} \\\n"
    f"      --data.num_workers={cpus_per_task} \\\n"
)

if args.requeue:
    command += f"      --overwrite_config \\\n      --log_suffix={log_suffix} \\\n"

# Fill extra args to Salt config
for arg, value in extra_config_args.items():
    if value is not None:
        command += f"      --{arg}={value} \\\n"
    else:
        command += f"      --{arg} \\\n"

if args.force:
    command += "      --force"
if args.environment == "singularity":
    command += '"'

# To inspect batch script before running
# handler.activate_testmode() # noqa: ERA001

# Submit the job
handler.send_job(command, args.tag)
