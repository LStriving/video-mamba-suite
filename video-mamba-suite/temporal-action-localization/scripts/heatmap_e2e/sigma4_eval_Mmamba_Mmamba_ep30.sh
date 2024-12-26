#!/bin/bash

python eval2stage.py \
    --config2 configs/2stage/heatmap/e2e/mamba/heatmap_secondstage_Mmamba_Mmamba_p2l3_ep30_sigma4.yaml \
    --ckpt2 ckpts/link2/e2e_heatmap_stage2_MmambaNMmamba_ep30_sigma4 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4_p0.23 \
    --heatmap \
    --re-extract > outputs/heatmap_e2e/sigma4_eval_Mmamba_Mmamba_ep30.txt

# python eval2stage.py \
#     --config2 configs/2stage/heatmap/e2e/mamba/heatmap_secondstage_Mmamba_Mmamba_p2l3_ep30_sigma4_hid576.yaml \
#     --ckpt2 ckpts/link2/e2e_heatmap_stage2_MmambaNMmamba_ep30_sigma4_hid576 \
#     --heatmap_size 56 \
#     --heatmap_branch none \
#     --cache_dir tmp/raw_heatmap_sigma4_p0.23 \
#     --heatmap \
#     --re-extract > outputs/heatmap_e2e/sigma4_eval_Mmamba_Mmamba_ep30_hid576.txt