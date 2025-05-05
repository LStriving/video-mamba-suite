#!/bin/bash
config=configs/2stage/2tower/crossmamba/vw_channelagg/mamba_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_vw0.6.yaml
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
vw=${vw:0:3}
config2=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_noact.yaml

echo "Ckpt folder: $output_folder, vw: $vw"
# python train2tower.py \
#     $config \
#     $config2 \
#     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576 \
#     --output load_heatmap \
#     --tower_name CrossMambaEarlyFusion \
#     --resume resume \
#     --enable_branch_eval

base_name=$(basename $output_folder)
mkdir -p outputs/${base_name}

# echo "Redirecting output to outputs/${base_name}/oracle_eval_${vw}.log"

python eval2tower.py \
    --seg_duration 5 \
    --config configs/2stage/mamba_swallow2_0427_i3d_train_stage1_traintest.yaml \
    --config2 $config \
    --config3 $config2 \
    --re-extract \
    --ckpt2 $output_folder \
    --cache_dir tmp/threshold0.23_swallow2 \
    --heatmap_dir tmp/swallow2_perfect_raw_heatmap_sigma4 \
    --heatmap \
    --heatmap_sigma 4 \
    --heatmap_branch none \
    --heatmap_size 56 \
    --image_size 128 \
    --only_perfect \
    --seg_duration 5 \
    --video_root data/swallow/external_videos2_processed/videos \
    --flow_dir data/swallow/external_videos2_processed/flowframes \
    --infer_perfect_stage1 --perfect_stage1 ./tmp/swallow2_0427_i3d_rgb128_flow128_perfect/ \
    --train_set \
    --only_perfect \
    --tower_name CrossMambaEarlyFusion 
    # > outputs/${base_name}/oracle_eval_${vw}.log

# echo all commands to output file