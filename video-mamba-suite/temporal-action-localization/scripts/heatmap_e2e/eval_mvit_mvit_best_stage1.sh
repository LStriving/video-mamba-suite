#!/bin/bash

python eval2stage.py \
    --ckpt ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_006_0.81519.pth.tar \
    --config2 configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3.yaml \
    --ckpt2 ckpts/link/e2e_heatmap_stage2_mvitNmvit \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/06stage1/raw_heatmap \
    --heatmap \
    --re-extract