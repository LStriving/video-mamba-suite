#!/bin/bash
sigmas=(6 8 1 2 3 5)
for sigma in "${sigmas[@]}"
do
config_name=configs/2stage/heatmap/e2e/sigma/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma${sigma}_hid576.yaml

python train2stage.py \
    ${config_name} \
    --resume training

output_log=outputs/resnet_heatmap.log
echo "Redirecting to ${output_log}"
mkdir  -p tmp/raw_heatmap_sigma${sigma}_p0.23
cp -r tmp/raw_heatmap_sigma4_p0.23/*.json tmp/raw_heatmap_sigma${sigma}_p0.23/
cp -r tmp/raw_heatmap_sigma4_p0.23/*.pkl tmp/raw_heatmap_sigma${sigma}_p0.23/
CUBLAS_WORKSPACE_CONFIG=:4096:8 python eval2stage.py \
    --config2 ${config_name} \
    --ckpt2 ./ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma${sigma}_hid576 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma${sigma}_p0.23 \
    --heatmap \
    --re-extract \
    --heatmap_sigma ${sigma} \
    --kalman True > ${output_log}