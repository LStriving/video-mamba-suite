#!/bin/bash
vws=(0.915 0.925)
for vw in "${vws[@]}"
do
    config=configs/2stage/2tower/crossmamba/nokalman/mamba_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg_nokalman_vw$vw.yaml
    output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
    vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
    vw=${vw:0:5}
    config2=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_nokalman.yaml

    # echo "Ckpt folder: $output_folder, vw: $vw"
    # python train2tower.py \
    #     $config \
    #     $config2 \
    #     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_nokalman \
    #     --output load_heatmap \
    #     --tower_name CrossMambaEarlyFusion \
    #     --resume resume
    #     # --enable_branch_eval

    base_name=$(basename $output_folder)
    mkdir -p outputs/${base_name}
    echo "Redirecting output to outputs/${base_name}/eval_${vw}.log"

    nohup python eval2tower.py \
        --config2 $config \
        --config3 $config2 \
        --re-extract \
        --ckpt2 $output_folder \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23_nokalman \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --kalman False \
        --tower_name CrossMambaEarlyFusion > outputs/${base_name}/eval_${vw}.log
done
