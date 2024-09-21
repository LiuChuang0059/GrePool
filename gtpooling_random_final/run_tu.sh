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
for data in NCI1; do
  for ratio in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    for n_encoders in 2 4 6 8; do
      for rate in 0.0 0.1 0.01 1.0 10.0; do
        python main.py --configs configs/TU/tu.yml --dataset $data --num_encoder_layers $n_encoders --gnn_num_layer 4 --epochs 100 --batch_size 128 --nhead 4 --num_workers 8 --token_ratio $ratio --uniform_rate $rate
      done
    done
  done
done

# DD IMDB-BINARY
for data in DD IMDB-BINARY; do
  for ratio in 0.75 0.5; do #0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
    for n_encoders in 2 4 6 8; do
      for rate in 0.0 0.1 0.01 1.0 10.0; do
        python main.py --configs configs/TU/tu.yml --dataset $data --num_encoder_layers $n_encoders --gnn_num_layer 4 --epochs 100 --batch_size 64 --nhead 4 --num_workers 4 --token_ratio $ratio --uniform_rate $rate
      done
    done
  done
done
