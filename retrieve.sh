#!/usr/bin/env bash

set -e

CUDA_VISIBLE_DEVICES=0 python -u retrieval.py \
  --anchors csv/openi_test.csv \
  --candidates csv/openi_test.csv \
  --clip-ckpt model/llm2clip4cxr.bin \
  --text-base lukeingawesome/llm2vec4cxr \
  --csv-img-key img_path \
  --csv-caption-key caption \
  --precision bf16 \
  --pooling-mode latent_attention \
  --text-max-len 512 \
  --batch 16 \
  --similarity clip \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --save retrieval_mimic.csv \
  --exit-no-segfault \
  --adapter-name llm2clip
