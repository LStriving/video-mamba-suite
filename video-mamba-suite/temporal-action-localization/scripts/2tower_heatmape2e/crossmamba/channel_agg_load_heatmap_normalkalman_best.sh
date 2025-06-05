#!/bin/bash
config=configs/2stage/2tower/crossmamba/normalkalman/mamba_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg_normalkalman_vw0.7.yaml
config2=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_normalkalman.yaml
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
base_name=$(basename $output_folder)
mkdir -p outputs/${base_name}
vws=(0.7)
for vw in "${vws[@]}"
do

    # echo "Ckpt folder: $output_folder, vw: $vw"
    # python train2tower.py \
    #     $config \
    #     $config2 \
    #     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_normalkalman \
    #     --output load_heatmap \
    #     --tower_name CrossMambaEarlyFusion \
    #     --resume resume
    #     # --enable_branch_eval

    echo "Redirecting output to outputs/${base_name}/eval_${vw}_best.log"

        # --ckpt2 ${output_folder} \
    nohup python eval2tower.py \
        --config2 $config \
        --config3 $config2 \
        --re-extract \
        --ckpt2 ${output_folder}_best \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23_normalkalman \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --kalman True \
        --normal_kalman \
        --vw $vw \
        --eval_single_cls \
        --tower_name CrossMambaEarlyFusion > outputs/${base_name}/eval_${vw}_best.log
done
