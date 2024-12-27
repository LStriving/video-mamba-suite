#!/bin/bash
config=$1
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)

# 打印output_folder的值
echo "Ckpt folder: $output_folder"

# train
python train2stage.py \
    ${config} \
    --resume resume

# eval
python eval2stage.py \
    --config2 ${config} \
    --ckpt2 ${output_folder} \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --heatmap \
    --re-extract
