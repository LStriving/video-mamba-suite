#!/bin/bash

python eval2stage.py \
    --config2 configs/2stage/heatmap/e2e/mamba/heatmap_secondstage_Mmamba_Mmamba_p2l3_ep30_sigma4_hid576_drop0_convpool.yaml \
    --ckpt2 ckpts/link2/e2e_heatmap_stage2_MmambaNMmamba_ep30_sigma4_hid576_drop0.0_conv \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --heatmap \
    --re-extract > outputs/heatmap_e2e/sigma4_eval_Mmamba_Mmamba_ep30_hid576_convp.log

