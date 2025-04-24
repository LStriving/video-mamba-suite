#!/bin/bash
config=$1
config2=$2
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
# get vw　("vw: 0.7," -> "0.7")
vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
vw=${vw:0:3}

# 打印output_folder的值
echo "Ckpt folder: $output_folder, vw: $vw"

python train2tower.py \
    $config \
    $config2 \
    --tower_name DINOAttnEarlyFusion \
    --output dino \
    --resume resume

base_name=$(basename $output_folder)
mkdir -p outputs/${base_name}

echo "Redirecting output to outputs/${base_name}/eval_${vw}.log"

nohup python eval2tower.py \
    --config2 $config \
    --config3 $config2 \
    --re-extract \
    --ckpt2 $output_folder \
    --cache_dir tmp/threshold0.23 \
    --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
    --heatmap \
    --heatmap_sigma 4 \
    --heatmap_branch none \
    --heatmap_size 56 \
    --image_size 128 \
    --tower_name DINOAttnEarlyFusion > outputs/${base_name}/eval_${vw}.log