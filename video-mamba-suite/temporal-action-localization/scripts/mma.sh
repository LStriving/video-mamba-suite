#!/bin/bash
config=configs/mmad/sg_mamba/best.yaml
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
vw=${vw:0:3}
config2=configs/mmad/sg_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep30_sigma1_hid576.yaml

echo "Ckpt folder: $output_folder, vw: $vw"
python train2tower.py \
    $config \
    $config2 \
    --output no_heatmap_pretrained \
    --tower_name CrossMambaEarlyFusion \
    --resume resume \
    --enable_branch_eval
    # --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma1_hid576 \

base_name=$(basename $output_folder)
mkdir -p outputs/${base_name}

# echo "Redirecting output to outputs/${base_name}/eval_${vw}.log"

# nohup python eval2tower.py \
#     --config2 $config \
#     --config3 $config2 \
#     --re-extract \
#     --ckpt2 $output_folder \
#     --cache_dir tmp/threshold0.23 \
#     --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
#     --heatmap \
#     --heatmap_sigma 4 \
#     --heatmap_branch none \
#     --heatmap_size 56 \
#     --image_size 128 \
#     --tower_name CrossMambaEarlyFusion > outputs/${base_name}/eval_${vw}.log

# echo all commands to output file