#!/bin/bash
#SBATCH --ntasks-per-node=1 
#SBATCH --nodes=1
#SBATCH --output=conv2d_gpu.out 
#SBATCH --error=conv2d_gpu.err 
#SBATCH --gres=gpu:k40:1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB


./conv2d_gpu

