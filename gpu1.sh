#!/bin/bash
#
#SBATCH --job-name=lang1
#SBATCH --output=1lang_res.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=gpu2
#SBATCH --nodelist=compute-2-13
#SBATCH --gres=gpu:1
#SBATCH --mem=100g 
#SBATCH --time=72:00:00

srun hostname
srun ./lang1.sh
