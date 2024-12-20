#!/bin/bash

python eval2stage.py \
    --config2 configs/2stage/heatmap/e2e/heatmap_secondstage_resnet_fix_mvit_p2l3_eval.yaml \
    --ckpt2 ckpts/e2e_heatmap_stage2 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap \
    --resnet_ateval \
    --heatmap \
    --re-extract

