#!/bin/bash
python train2tower.py \
    configs/2stage/2tower/crossattn/mamba_swallow_i3d_secondstage_2tower_ep30_acu4.yaml \
    configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.yaml \
    --tower_name DINOAttnEarlyFusion \
    --output dino