config_name=configs/2stage/heatmap/e2e/extract_model/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_resnet_0604_nokalman.yaml

python train2stage.py \
    ${config_name} \
    --resume training

output_log=outputs/resnet_heatmap_0604_nokalman.log
echo "Redirecting to ${output_log}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python eval2stage.py \
    --config2 ${config_name} \
    --ckpt2 ./ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_resnet_0604_nokalman \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4_p0.23_resnet_0604_nokalman \
    --heatmap \
    --re-extract \
    --kalman False > ${output_log}