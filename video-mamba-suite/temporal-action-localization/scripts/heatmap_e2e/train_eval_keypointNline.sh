#!/bin/bash
# config=configs/2stage/heatmap/e2e/input/keypoint_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.yaml
# output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)

# # 打印output_folder的值
# echo "Ckpt folder: $output_folder"

# # train
# # python train2stage.py \
# #     ${config} \
# #     --resume resume

# # mkdir
# mkdir -p outputs/heatmap_e2e
# echo "Redirect to outputs/heatmap_e2e/keypoint_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.log"
# # eval
# python eval2stage_tmp.py \
#     --config2 ${config} \
#     --ckpt2 ${output_folder} \
#     --heatmap_size 56 \
#     --heatmap_branch none \
#     --cache_dir tmp/raw_heatmap_sigma4_keypoint \
#     --heatmap \
#     --heatmap_type keypoint \
#     --re-extract > outputs/heatmap_e2e/keypoint_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.log

# tail outputs/heatmap_e2e/keypoint_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.log
######## line


config=configs/2stage/heatmap/e2e/input/linediv2_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.yaml
output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)

# 打印output_folder的值
echo "Ckpt folder: $output_folder"

# train
# python train2stage.py \
#     ${config} \
#     --resume resume

echo "Redirect to outputs/heatmap_e2e/line2_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.log"
# eval
# python eval2stage.py \
#     --config2 ${config} \
#     --ckpt2 ${output_folder} \
#     --heatmap_size 56 \
#     --heatmap_branch none \
#     --cache_dir tmp/raw_heatmap_sigma4 \
#     --heatmap \
#     --re-extract

nohup python eval2stage_tmp.py \
    --config2 ${config} \
    --ckpt2 ${output_folder} \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4_line_div2 \
    --heatmap \
    --heatmap_type line \
    --re-extract > outputs/heatmap_e2e/line2_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.log
tail outputs/heatmap_e2e/line2_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576.log