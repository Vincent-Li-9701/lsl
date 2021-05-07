#!/bin/bash

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=4 python -u lsl/train.py --cuda \
    --batch_size 72 \
    --seed $RANDOM \
    --lr 1e-5 \
    --warmup_ratio 0.05 \
    --initializer_range 0.02 \
    --epochs 501 \
    --backbone lxmert \
    --optimizer bertadam \
    exp/meta #> vision_output_feat_fast_long.out
