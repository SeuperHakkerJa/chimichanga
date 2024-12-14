#!/bin/bash
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH -n 12

######SBATCH -p gpu  --gres=gpu:1

#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH --output=taco-exp/slurm-%j.out

module load cuda/11.8.0-lpttyok
module load mesa 
module load gcc 

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate taco


# python train.py agent=taco task=quadruped_run save_video=True exp_name=test_run
python train.py agent=taco task=cheetah_run save_video=True exp_name=cheetah_test_2 curl=False