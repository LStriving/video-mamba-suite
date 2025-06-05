#!/bin/bash
# bash scripts/extract_heatmap/resnet_0604_nokalman.sh
# bash scripts/heatmap_e2e/mamba/resnet_0604_nokalman.sh


config=configs/2stage/2tower/crossmamba/heatmap_model/mamba_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg_resnet_vw0.7.yaml
vws=(0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79)
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
config2=configs/2stage/heatmap/e2e/extract_model/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_resnet_0604_nokalman.yaml
base_name=$(basename $output_folder)
mkdir -p outputs/${base_name}
# echo "Ckpt folder: $output_folder, vw: $vw"
# python train2tower.py \
#     $config \
#     $config2 \
#     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_resnet_0604_nokalman \
#     --output load_heatmap \
#     --tower_name CrossMambaEarlyFusion \
#     --resume resume
    #     # --enable_branch_eval

# if not exist, create dir
if [ ! -d "tmp/raw_heatmap_sigma4_p0.23_resnet_0604_nokalman" ]; then
    mkdir -p tmp/raw_heatmap_sigma4_p0.23_resnet_0604_nokalman
    cp tmp/raw_heatmap_sigma4_p0.23_resnet/*.json tmp/raw_heatmap_sigma4_p0.23_resnet_0604_nokalman/
    cp tmp/raw_heatmap_sigma4_p0.23_resnet/*.pkl tmp/raw_heatmap_sigma4_p0.23_resnet_0604_nokalman/
fi

for vw in "${vws[@]}"
do
    echo "Redirecting output to outputs/${base_name}/eval_${vw}_0604_nokalman.log"

    nohup python eval2tower.py \
        --config2 $config \
        --config3 $config2 \
        --re-extract \
        --ckpt2 $output_folder \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23_resnet_0604_nokalman \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --kalman False \
        --vw  $vw \
        --tower_name CrossMambaEarlyFusion > outputs/${base_name}/eval_${vw}_0604_nokalman.log
    
    tail outputs/${base_name}/eval_${vw}_0604_nokalman.log
done
