#!/bin/bash
#SBATCH --time=03:15:00 
#SBATCH --mem-per-cpu=1024
#SBATCH --job-name=cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint=gpu_32gb
#SBATCH --error=/work/[groupname]/[username]/job.%J.err
#SBATCH --output=/work/[groupname]/[username]/job.%J.out


# Probably 24 GB of RAM
# Probably 32 GB of VRAM

#Load Modules

module load cuda
module load tensorflow-gpu/py311/2.15
module load anaconda

conda activate layer_skip

python train_ee_cnn.py --max-documents 100 --output-dir checkpoints/ee_cnn



