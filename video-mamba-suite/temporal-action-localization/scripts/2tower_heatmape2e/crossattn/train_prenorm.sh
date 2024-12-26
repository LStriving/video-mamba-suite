#!/bin/bash
python train2tower.py \
    configs/2stage/2tower/crossattn/mamba_swallow_i3d_secondstage_2tower_detach.yaml \
    configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4.yaml \
    --tower_name CrossAttnEarlyFusion-PreNorm \
    --output test