#!/bin/bash
#SBATCH --job-name=kl_dsrl_square_500k
#SBATCH --account=pi_tkf6
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/slurm/kl_dsrl_square_500k_%A_%a.out
#SBATCH --error=logs/slurm/kl_dsrl_square_500k_%A_%a.err

# Train each β (seed-1) from scratch to TOTAL steps; the final replay buffer is saved
# alongside final.zip so the run can be resumed later without losing data.

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate dsrl
module purge

export MUJOCO_PY_MUJOCO_PATH=/home/cz493/project_pi_tkf6/cz493/.mujoco/mujoco210
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia

cd /nfs/roberts/project/pi_tkf6/cz493/dsrl

SEED=1
TOTAL=500000
BETAS=(0 0.1 0.5 1 2)
TAGS=(0 0p1 0p5 1 2)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

python train_kl_dsrl.py \
    --config-path=cfg/robomimic \
    --config-name=kl_dsrl_square.yaml \
    seed=${SEED} \
    train.kl_coef=${BETA} \
    total_timesteps=${TOTAL} \
    +save_final_replay_buffer=True \
    name=robomimic_square_kl_dsrl_beta${TAG}_seed${SEED}_${TOTAL}
