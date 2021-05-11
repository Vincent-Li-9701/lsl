#!/bin/bash

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=9 python -u lsl/train.py --cuda \
    --batch_size 72 \
    --seed $RANDOM \
    --lr 5e-6 \
    --warmup_ratio 0.10 \
    --initializer_range 0.02 \
    --epochs 1401 \
    --backbone lxmert \
    --optimizer bertadam \
    exp/meta #> classification_head_fast_long_support_mean.out
