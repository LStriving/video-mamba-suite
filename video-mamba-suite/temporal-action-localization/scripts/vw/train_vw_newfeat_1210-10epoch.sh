#!/bin/bash
# exp 2
# vw
vws=(0.6)
for vw in ${vws[@]}
do
    nohup python train2tower.py \
        ./configs/2stage/vw/mamba_swallow_i3d_secondstage_2tower_10ep_vw${vw}.yaml \
        configs/2stage/heatmap/mamba_swallow_heatmap_secondstage_newfeat_1210-10epoch.yaml \
        --tower_nam LogitsAvg \
        --output 1210_10epoch > outputs/train_newfeat1210-10epoch_${vw}.log &
done

