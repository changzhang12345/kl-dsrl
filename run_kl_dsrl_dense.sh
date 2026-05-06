#!/bin/bash
#SBATCH --job-name=kl_dsrl_square_dense
#SBATCH --account=pi_tkf6
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/kl_dsrl_square_dense_%j.out
#SBATCH --error=logs/slurm/kl_dsrl_square_dense_%j.err

source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
module purge
conda activate dsrl

export MUJOCO_PY_MUJOCO_PATH=/home/cz493/project_pi_tkf6/cz493/.mujoco/mujoco210
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia

cd /nfs/roberts/project/pi_tkf6/cz493/dsrl

SEED=1
BETA=0.5
BETA_TAG=0p5

python train_kl_dsrl.py \
    --config-path=cfg/robomimic \
    --config-name=kl_dsrl_square_dense.yaml \
    seed=${SEED} \
    train.kl_coef=${BETA} \
    name=robomimic_square_kl_dsrl_dense_beta${BETA_TAG}_seed${SEED}
