#!/bin/bash
#SBATCH --job-name=dsrl_can_baseline
#SBATCH --account=pi_tkf6
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=28G
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/baseline_can_seed%a_%j.out
#SBATCH --error=logs/slurm/baseline_can_seed%a_%j.err

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate dsrl

export MUJOCO_PY_MUJOCO_PATH=/home/cz493/project_pi_tkf6/cz493/.mujoco/mujoco210
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia

cd /nfs/roberts/project/pi_tkf6/cz493/dsrl

python train_dsrl.py \
    --config-path=cfg/robomimic \
    --config-name=dsrl_can.yaml \
    seed=${SLURM_ARRAY_TASK_ID}
