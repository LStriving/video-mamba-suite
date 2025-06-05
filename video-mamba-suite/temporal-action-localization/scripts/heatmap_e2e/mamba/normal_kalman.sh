config_name=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_normalkalman.yaml

python train2stage.py \
    ${config_name} \
    --resume training

output_log=outputs/normal_kalman_heatmap.log
echo "Redirecting to ${output_log}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python eval2stage.py \
    --config2 ${config_name} \
    --ckpt2 ./ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_normalkalman \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4_p0.23_normalkalman \
    --heatmap \
    --re-extract \
    --kalman True \
    --normal_kalman > ${output_log}