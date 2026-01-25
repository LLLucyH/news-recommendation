#!/bin/bash
#SBATCH --ntasks-per-node=2
#SBATCH -N 1
#SBATCH --gres=gpu:volta:2
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


srun python -u train_baseline.py "$@"


