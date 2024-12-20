#!/bin/bash

cd /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization

CUDA_VISIBLE_DEVICES=1,2,3,4 python eval2stage.py \
    --config2 configs/2stage/heatmap/e2e/heatmap_secondstage_mvitV2_mvitV2_p2l3_ep30.yaml \
    --ckpt2 ckpts/link/e2e_heatmap_stage2_mvitV2NmvitV2_ep30 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap \
    --heatmap \
    --re-extract