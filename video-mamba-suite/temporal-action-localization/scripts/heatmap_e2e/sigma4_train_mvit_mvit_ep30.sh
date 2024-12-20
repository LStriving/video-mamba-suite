#!/bin/bash

python train2stage.py \
    configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4.yaml \
    --output hid1024 \
    --resume resume