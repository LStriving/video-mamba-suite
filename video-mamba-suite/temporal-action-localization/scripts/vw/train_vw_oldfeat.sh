#!/bin/bash
# exp 2
# vw

# using the same pretrained i3d-rgb model to extract both rgb and heatmap features and set heatmap size to 56
vws=(0.7)
for vw in ${vws[@]}
do
    nohup python train2tower.py \
    ./configs/2stage/vw/mamba_swallow_i3d_secondstage_2tower_10ep_vw${vw}.yaml \
    configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml \
    --tower_nam LogitsAvg \
    --output oldfeat > outputs/train_oldfeat_${vw}.log &
done

