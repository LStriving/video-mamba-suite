#!/bin/bash
# exp 2
# vw
vws=(0.3)
for vw in ${vws[@]}
do
    nohup python train2tower_2opt.py ./configs/2stage/vw/mamba_swallow_i3d_secondstage_2tower_10ep_vw${vw}.yaml configs/2stage/heatmap/mamba_swallow_heatmap_secondstage_newfeat.yaml --tower_name LogitsAvg_sepbranch > outputs/train2opt_${vw}.log &
done
