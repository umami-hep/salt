#!/bin/bash

# Job name
#SBATCH --job-name=salt

# choose the GPU queue
#SBATCH -p GPU

# requesting one node
#SBATCH -nodes=1
#SBATCH --exclusive

# keep environment variables
#SBATCH --export=ALL

# requesting 4 V100 GPU
# (remove the "v100:" if you don't care what GPU)
#SBATCH --gres=gpu:v100:4

# note! this needs to match --trainer.devices!
#SBATCH --ntasks-per-node=4

# request enough memory
#SBATCH --mem=200G

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/share/rcifdata/svanstroud/submit/out/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=/share/rcifdata/svanstroud/submit/out/slurm-%j.%x.err

# speedup
export OMP_NUM_THREADS=1

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
srun main fit \
    --config configs/gnn.yaml \
    --trainer.devices 4 \
