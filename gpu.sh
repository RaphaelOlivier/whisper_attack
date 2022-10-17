#!/bin/bash
#
#SBATCH --job-name=cw
#SBATCH --output=cw_res.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-2-5
#SBATCH --gres=gpu:3 
#SBATCH --mem=100g 
#SBATCH --time=72:00:00

srun hostname
srun ./cw.sh
