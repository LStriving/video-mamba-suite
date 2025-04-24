config_name=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep80_sigma4_hid576_nokalman.yaml

python train2stage.py \
    ${config_name} \
    --resume training

python eval2stage.py \
    --config2 ${config_name} \
    --ckpt2 ./ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep80_sigma4_hid576_nokalman \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4_p0.23_nokalman \
    --heatmap \
    --re-extract \
    --kalman False