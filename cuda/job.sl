#!/bin/bash
#SBATCH --ntasks-per-node=1 
#SBATCH --nodes=1
#SBATCH --time=00:01:00 
#SBATCH --output=conv2d_gpu.out 
#SBATCH --error=conv2d_gpu.err 
#SBATCH --gres=gpu:k40:1

./conv2d_gpu

