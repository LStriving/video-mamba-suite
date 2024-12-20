#!/bin/bash
# exp 2
# vw
vws=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for vw in ${vws[@]}
do
    nohup python eval2tower.py \
    --config2 ./configs/2stage/vw/mamba_swallow_i3d_secondstage_2tower_10ep_vw${vw}.yaml \
    --config3 configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml \
    --re-extract \
    --ckpt2 ckpts/2tower/fuse_logits0.7 \
    --heatmap_i3d pretrained/pretrained_swallow_i3d.pth \
    --cache_dir tmp/mulmodal_new \
    --heatmap_dir tmp/heatmap_orirgbi3d \
    --heatmap \
    --heatmap_branch rgb \
    --heatmap_size 56 \
    --tower_nam LogitsAvg > outputs/eval_oldfeat_${vw}.log
done
