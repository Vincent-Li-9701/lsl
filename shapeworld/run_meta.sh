#!/bin/bash

if [ "$#" -eq 1 ]; then
    name="--name $1"
else
    name=""
fi

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python -u lsl/train.py --cuda \
    --batch_size 72 \
    --seed $RANDOM \
    --lr 5e-6 \
    --warmup_ratio 0.05 \
    --initializer_range 0.02 \
    --epochs 451 \
    --backbone lxmert \
    --optimizer bertadam \
    --wandb $name \
    exp/meta > classification_head_small_support_mean_linear.out
