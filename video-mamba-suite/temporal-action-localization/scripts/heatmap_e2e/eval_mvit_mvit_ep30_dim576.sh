#!/bin/bash

python eval2stage.py \
    --config2 configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4_hid576.yaml \
    --ckpt2 ckpts/link/e2e_heatmap_stage2_mvitNmvit_ep30_sigma4_hid576 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --heatmap \
    --re-extract