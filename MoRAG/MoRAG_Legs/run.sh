#!/bin/bash
#SBATCH -w gnode077
#BATCH -A research
#SBATCH -c 39
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=6-00:00:00
###SBATCH --mail-type=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sai.shashank@research.iiit.ac.in

source ~/.venv/TMR/bin/activate
HYDRA_FULL_ERROR=1 python train.py

