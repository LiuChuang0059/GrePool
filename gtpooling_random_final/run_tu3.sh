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

if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

# run TU dataset 3
# ENZYMES Mutagenicity FRANKENSTEIN

for attn_drop in 0.0; do
  for ratio in 0.5 0.9 0.7 0.3 0.1; do
    for n_encoders in 4 6 8; do
      for rate in 0.0; do
        for data in ENZYMES Mutagenicity FRANKENSTEIN; do
          python main.py --configs configs/TU/tu.yml --dataset $data --device $cuda \
            --num_encoder_layers $n_encoders --gnn_num_layer 4 --epochs 100 \
            --batch_size 128 --nhead 4 --num_workers 8 \
            --token_ratio $ratio --uniform_rate $rate --dropout_attn $attn_drop
        done
      done
    done
  done
done
