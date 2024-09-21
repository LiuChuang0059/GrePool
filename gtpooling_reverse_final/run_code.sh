#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# config=$1

# echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
# source ~/.bashrc
# conda activate graph-aug

# echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES

# echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES



for ratio in 0.5 0.75
do
	python main.py --configs configs/code2/gnn-transformer/JK=cat/pooling=cls+norm_input.yml\
					   --num_encoder_layers 4 \
					   --gnn_num_layer 4\
			           --epochs 30\
			           --batch_size 16\
			           --nhead 4 \
			           --runs 5\
			           --num_workers 4\
			           --token_ratio $ratio
done
