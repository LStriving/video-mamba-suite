actions=(1 2 3 4 5 6 7)
config2=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_noact.yaml
for action in ${actions[@]}
do
    config=configs/best_single/best_$action.yaml
    output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
    action_name=$(grep -oP 'desired_actions: \[\K[^]]*' $config)
    echo $action_name
    base_name=$(basename $output_folder)
    mkdir -p outputs/$base_name
    echo "Redirecting output to outputs/$base_name/$action_name.log"
    nohup python eval2tower.py \
        --config2 $config \
        --config3 $config2 \
        --re-extract \
        --ckpt2 $output_folder \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --dump_result \
        --tower_name CrossMambaEarlyFusion > outputs/$base_name/$action_name.log
done