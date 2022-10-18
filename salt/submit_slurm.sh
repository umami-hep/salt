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

# mail on failures
#SBATCH --mail-user=sam.van.stroud@cerh.ch
#SBATCH --mail-type=FAIL

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/share/rcifdata/svanstroud/submit/out/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=/share/rcifdata/svanstroud/submit/out/slurm-%j.%x.err

# speedup trick
export OMP_NUM_THREADS=1

cd /share/rcifdata/svanstroud/salt/salt
echo "Moved dir, now in: ${PWD}"

source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
conda activate salt
echo "Activated environment ${CONDA_DEFAULT_ENV}"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Running train.py fit"
python train.py fit \
    --config configs/simple.yaml \
    --trainer.devices 4 \
