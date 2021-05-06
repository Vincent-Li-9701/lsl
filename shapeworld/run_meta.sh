#!/bin/bash

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python -u lsl/train.py --cuda \
    --batch_size 48 \
    --seed $RANDOM \
    --lr 1e-6 \
    --warmup_ratio 0.05 \
    --initializer_range 0.02 \
    --epochs 250 \
    --backbone lxmert \
    --optimizer bertadam \
    exp/meta #> vision_output_full_pretrianed.out
