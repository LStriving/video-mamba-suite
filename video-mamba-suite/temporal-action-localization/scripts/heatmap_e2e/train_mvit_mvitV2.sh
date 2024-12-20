#!/bin/bash

python train2stage.py \
    configs/2stage/heatmap/e2e/heatmap_secondstage_mvitV2_mvitV2_p2l3_ep30.yaml \
    --output hid1024 \
    --resume resume