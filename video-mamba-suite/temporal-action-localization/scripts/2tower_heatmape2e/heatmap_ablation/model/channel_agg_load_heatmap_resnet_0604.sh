#!/bin/bash

bash scripts/heatmap_e2e/mamba/resnet_0604.sh


config=configs/2stage/2tower/crossmamba/heatmap_model/mamba_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg_resnet_vw0.7.yaml
vws=(0.5 0.6 0.7 0.8 0.9 1)
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
config2=configs/2stage/heatmap/e2e/extract_model/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_resnet_0604.yaml
base_name=$(basename $output_folder)
mkdir -p outputs/${base_name}
echo "Ckpt folder: $output_folder, vw: $vw"
python train2tower.py \
    $config \
    $config2 \
    --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_resnet_0604 \
    --output load_heatmap \
    --tower_name CrossMambaEarlyFusion \
    --resume resume
    #     # --enable_branch_eval


mkdir -p tmp/raw_heatmap_sigma4_p0.23_resnet_0604
cp tmp/raw_heatmap_sigma4_p0.23_resnet/*.json tmp/raw_heatmap_sigma4_p0.23_resnet_0604/
cp tmp/raw_heatmap_sigma4_p0.23_resnet/*.pkl tmp/raw_heatmap_sigma4_p0.23_resnet_0604/

for vw in "${vws[@]}"
do
    echo "Redirecting output to outputs/${base_name}/eval_${vw}_0604.log"

    nohup python eval2tower.py \
        --config2 $config \
        --config3 $config2 \
        --re-extract \
        --ckpt2 $output_folder \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23_resnet_0604 \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --kalman True \
        --vw  $vw \
        --tower_name CrossMambaEarlyFusion > outputs/${base_name}/eval_${vw}_0604.log
done
bash scripts/2tower_heatmape2e/heatmap_ablation/model/channel_agg_load_heatmap_resnet_0604_nokalman.sh