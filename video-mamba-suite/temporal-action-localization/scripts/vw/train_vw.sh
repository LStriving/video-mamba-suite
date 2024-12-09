#!/bin/bash
# exp 2
# vw
vws=(0.3)
for vw in ${vws[@]}
do
    nohup python train2tower.py ./configs/2stage/vw/mamba_swallow_i3d_secondstage_2tower_10ep_vw${vw}.yaml configs/2stage/heatmap/mamba_swallow_heatmap_secondstage_newfeat.yaml --tower_nam LogitsAvg > outputs/train_${vw}.log &
done
