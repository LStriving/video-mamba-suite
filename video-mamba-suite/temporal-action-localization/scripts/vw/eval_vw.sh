#!/bin/bash
# exp 2
# vw
vws=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for vw in ${vws[@]}
do
    nohup python eval2tower.py \
    --config2 ./configs/2stage/vw/mamba_swallow_i3d_secondstage_2tower_10ep_vw${vw}.yaml \
    --config3 configs/2stage/heatmap/mamba_swallow_heatmap_secondstage_newfeat.yaml \
    --re-extract \
    --ckpt2 ckpts/2tower/fuse_logits0.3 \
    --flow_i3d /mnt/cephfs/home/zhoukai/Codes/vfss/vfss_tal/log/lr0_05_bs8_i3d_flow_bce_224_rot30_prob0_8/best_ckpt.pt \
    --cache_dir tmp/mulmodal \
    --heatmap_dir tmp/heatmap_pretrained \
    --heatmap \
    --tower_nam LogitsAvg > outputs/eval_${vw}.log
done