#!/bin/bash

# Job name
#SBATCH --job-name=salt

# choose the GPU queue
#SBATCH -p GPU

# requesting one node
#SBATCH --nodes=1
# Only if you really need it!
# #SBATCH --exclusive

# keep environment variables
#SBATCH --export=ALL

# requesting 4 V100 GPU
# (remove the "v100:" if you don't care what GPU)
#SBATCH --gres=gpu:a100:1

# note! this needs to match --trainer.devices!
#SBATCH --ntasks-per-node=1

# number of cpus per task
# don't use this if you have exclusive access to the node
#SBATCH --cpus-per-task=10

# request enough memory
#SBATCH --mem=100G

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/share/rcifdata/svanstroud/submit/out/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=/share/rcifdata/svanstroud/submit/out/slurm-%j.%x.err

# speedup (not sure if this does anything)
export OMP_NUM_THREADS=1

# print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"

# move to workdir
cd /share/rcifdata/svanstroud/salt/salt
echo "Moved dir, now in: ${PWD}"

# activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate salt
echo "Activated environment ${CONDA_DEFAULT_ENV}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# run the training
echo "Running training script..."
srun salt fit \
    --config configs/GN1.yaml \
