#!/bin/bash

config2=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_noact.yaml

# python train2tower.py \
#     $config \
#     $config2 \
#     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576 \
#     --output load_heatmap \
#     --tower_name DINOAttnEarlyFusion \
#     --resume resume \
#     --enable_branch_eval

vws=(0.0 0.5 0.6 0.8 0.9 1.0)
for vw in ${vws[@]}
do
    config=configs/2stage/2tower/crossattn/heatmap_vw/mamba_swallow_i3d_secondstage_2tower_dino_l3_ep30_acu4_loadheatmap_vw${vw}.yaml
    output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
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
done
# echo all commands to output file