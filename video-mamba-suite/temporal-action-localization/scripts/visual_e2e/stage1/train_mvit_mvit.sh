#!/bin/bash

python train2stage.py \
    configs/2stage/stage1/mamba_swallow_rgb_train_stage1_traintest_e2e.yaml \
    --output hid1024 \
    --resume resume