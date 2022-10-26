#!/bin/bash

# Job name
#SBATCH --job-name=salt

# choose the GPU queue
#SBATCH -p GPU

#requesting one node
#SBATCH -N1
#SBATCH --exclusive

# keep environment variables
#SBATCH --export=ALL

#requesting cpus
#SBATCH -n24

#requesting 4 V100 GPU
#SBATCH --gres=gpu:v100:4

# request enough memory
#SBATCH --mem=200G

# speedup trick
export OMP_NUM_THREADS=1

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
train \
    --config configs/simple.yaml \
    --trainer.devices 4 \
