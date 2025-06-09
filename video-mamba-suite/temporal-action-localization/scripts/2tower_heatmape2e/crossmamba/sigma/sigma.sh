sigmas=(6 8 1 2 3 5)
vws=(0.7 0.8 0.9)
for sigma in "${sigmas[@]}"
do
    config=configs/2stage/2tower/crossmamba/sigma/mamba_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg_sigma${sigma}.yaml
    output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
    vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
    config2=configs/2stage/heatmap/e2e/sigma/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma${sigma}_hid576.yaml

    # echo "Ckpt folder: $output_folder, vw: $vw"
    # python train2tower.py \
    #     $config \
    #     $config2 \
    #     --backbone_2 ./ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma${sigma}_hid576 \
    #     --output load_heatmap \
    #     --tower_name CrossMambaEarlyFusion \
    #     --resume resume

    base_name=$(basename $output_folder)
    mkdir -p outputs/sigma/${base_name}
    
    for vw in "${vws[@]}"
    do
        # vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
        # vw=${vw:0:3}
        output_log=outputs/sigma/${base_name}/eval_${vw}_sigma${sigma}.log
        echo "Redirecting output to $output_log"

        nohup python eval2tower.py \
            --config2 $config \
            --config3 $config2 \
            --re-extract \
            --ckpt2 $output_folder \
            --cache_dir tmp/threshold0.23 \
            --heatmap_dir tmp/raw_heatmap_sigma${sigma}_p0.23 \
            --heatmap \
            --heatmap_sigma ${sigma} \
            --heatmap_branch none \
            --heatmap_size 56 \
            --image_size 128 \
            --vw $vw \
            --tower_name CrossMambaEarlyFusion > $output_log
        echo "Processing sigma ${sigma} with vw ${vw}..." >> outputs/sigma/2tower_crossmamba_sigma.log
        tail $output_log >> outputs/sigma/2tower_crossmamba_sigma.log
        echo "Finished processing sigma ${sigma}, output saved to $output_log"
    done
done
echo "All sigma evaluations completed. Check outputs/sigma/2tower_crossmamba_sigma.log for details."