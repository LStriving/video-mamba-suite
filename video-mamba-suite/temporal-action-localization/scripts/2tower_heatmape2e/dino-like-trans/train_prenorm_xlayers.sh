#!/bin/bash

layers=(2 4 5)
vws=(0.0 0.5 0.7 0.8 0.9 1.0)

for layer in ${layers[@]}
do
    python train2tower.py \
    configs/2stage/2tower/crossattn/mamba_swallow_i3d_secondstage_2tower_dino_layer${layer}.yaml \
    configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4.yaml \
    --tower_name DINOAttnEarlyFusion \
    --output dino

    
    mkdir -p outputs/2tower_heatmape2e_DINOcrossattn${layer}layers
    for vw in ${vws[@]}
    do
        nohup python eval2tower.py \
        --config2 configs/2stage/2tower/crossattn/layer${layer}_vw/mamba_swallow_i3d_secondstage_2tower_vw${vw}.yaml \
        --config3 configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4.yaml \
        --re-extract \
        --ckpt2 ckpts/link2/2tower_${layer}layers_dinolikeattn \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --tower_nam DINOAttnEarlyFusion > outputs/2tower_heatmape2e_DINOcrossattn${layer}layers/eval_${vw}.log
    done
done

