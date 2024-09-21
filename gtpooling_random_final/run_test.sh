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

# echo "python main.py --configs $config --num_workers 0 --devices $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES


# NCI1
for data in ogbg-hiv
do
    for ratio in 0.9 #0.8 0.7 0.75 0.6 0.5 0.4 0.3 0.2 0.1
    do
        python main.py --configs configs/OGB/ogb.yml\
                       --dataset $data\
                       --num_encoder_layers 4 \
                       --gnn_num_layer 4\
                       --epochs 2\
                       --batch_size 32\
                       --nhead 4\
                       --token_ratio $ratio\
                       --dropout_attn 0.1\
                       --runs 1
    done
done









