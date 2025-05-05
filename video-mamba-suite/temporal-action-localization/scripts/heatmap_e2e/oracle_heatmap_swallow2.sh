#!/bin/bash
# try to get perfect stage1 heatmap feature (A script just to get for the perfect stage1 heatmap feature not the final result)

# python eval2stage.py \
#     --config configs/2stage/mamba_swallow2_0427_i3d_train_stage1_traintest.yaml \
#     --config2 configs/2stage/mamba_swallow2_0427_i3d_secondstage.yaml \
#     --ckpt ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar \
#     --ckpt2 ckpts/ckpt_swallow_stage2/ \
#     --re-extract --cache_dir ./tmp/multi_class_swallow2_0427/ \
#     --image_size 128 \
#     --flow_i3d /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/pretrained/flow_imagenet.pt \
#     --infer_perfect_stage1 --perfect_stage1 ./tmp/swallow2_0427_i3d_rgb128_flow128_perfect/ \
#     --only_perfect \
#     --train_set --seg_duration 5 \
#     --video_root data/swallow/external_videos2_processed/videos \
#     --flow_dir data/swallow/external_videos2_processed/flowframes

python eval2stage.py \
    --config configs/2stage/mamba_swallow2_0427_i3d_train_stage1_traintest.yaml \
    --config2 configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_noact.yaml \
    --ckpt ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar \
    --ckpt2 xxx \
    --cache_dir tmp/multi_class_swallow2/ \
    --re-extract \
    --heatmap_size 56 \
    --seg_duration 5 \
    --heatmap \
    --heatmap_branch none \
    --infer_perfect_stage1 --perfect_stage1 tmp/swallow2_perfect_raw_heatmap_sigma4 \
    --video_root data/swallow/external_videos2_processed/videos \
    --flow_dir data/swallow/external_videos2_processed/flowframes \
    --only_perfect