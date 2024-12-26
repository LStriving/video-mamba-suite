#!/bin/bash

python eval2stage.py \
    --config2 configs/2stage/heatmap/e2e/mamba/heatmap_secondstage_vmamba_mamba_p2l3_ep30_sigma4.yaml \
    --ckpt2 ckpts/link2/e2e_heatmap_stage2_mambaNmamba_ep30_sigma4 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --heatmap \
    --re-extract