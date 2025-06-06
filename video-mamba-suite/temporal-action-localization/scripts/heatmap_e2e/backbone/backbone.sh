#!/bin/bash
# backbones=(actionformer tridet)
backbones=(actionformer)

for backbone in "${backbones[@]}"
do
    config_name=configs/2stage/heatmap/e2e/diff_backbone/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_${backbone}.yaml

    # python train2stage.py \
    #     ${config_name} \
    #     --resume training
    # mkdir -p outputs/backbone
    # output_log=outputs/backbone/${backbone}_heatmap.log
    # echo "Redirecting to ${output_log}"

    # CUBLAS_WORKSPACE_CONFIG=:4096:8 python eval2stage.py \
    #     --config2 ${config_name} \
    #     --ckpt2 ./ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_${backbone} \
    #     --heatmap_size 56 \
    #     --heatmap_branch none \
    #     --cache_dir tmp/raw_heatmap_sigma4_p0.23 \
    #     --heatmap \
    #     --re-extract > ${output_log}
    
    # tail ${output_log}

    # # dual branch
    # config=configs/2stage/2tower/crossmamba/${backbone}_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg_logitsavg.yaml
    # output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
    # vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
    # vw=${vw:0:3}
    

    # echo "Ckpt folder: $output_folder, vw: $vw"
    # python train2tower.py \
    #     $config \
    #     $config_name \
    #     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_${backbone} \
    #     --output load_heatmap \
    #     --tower_name LogitsAvg \
    #     --resume resume \
    #     # --enable_branch_eval

    # base_name=$(basename $output_folder)
    # mkdir -p outputs/backbone/${base_name}
    # output_log=outputs/backbone/${base_name}/eval_${vw}.log
    # echo "Redirecting output to ${output_log}"
    # nohup python eval2tower.py \
    #     --config2 $config \
    #     --config3 $config_name \
    #     --re-extract \
    #     --ckpt2 $output_folder \
    #     --cache_dir tmp/threshold0.23 \
    #     --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
    #     --heatmap \
    #     --heatmap_sigma 4 \
    #     --heatmap_branch none \
    #     --heatmap_size 56 \
    #     --image_size 128 \
    #     --tower_name LogitsAvg > ${output_log}
    # tail ${output_log}

    # # ==========================================================================================================================
    # # ccm
    config=configs/2stage/2tower/crossmamba/${backbone}_swallow_i3d_secondstage_2tower_crossmamba_l3_ep30_acu4_loadheatmap_channelagg.yaml
    output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
    # vw=$(grep 'vw:' "$config" | awk -F ':' '{print $2}' | xargs)
    # vw=${vw:0:3}
    

    # echo "Ckpt folder: $output_folder, vw: $vw"
    # python train2tower.py \
    #     $config \
    #     $config_name \
    #     --backbone_2 ckpts/link2/e2e_heatmap_stage2_video_mamba_l3_ep45_sigma4_hid576_${backbone} \
    #     --output load_heatmap \
    #     --tower_name CrossMambaEarlyFusion \
    #     --resume resume \
        # --enable_branch_eval

    base_name=$(basename $output_folder)
    # mkdir -p outputs/backbone/${base_name}
    vw=0.6
    output_log=outputs/backbone/${base_name}/eval_${vw}.log

    echo "Redirecting output to ${output_log}"

    nohup python eval2tower.py \
        --config2 $config \
        --config3 $config_name \
        --re-extract \
        --ckpt2 $output_folder \
        --cache_dir tmp/threshold0.23 \
        --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
        --heatmap \
        --heatmap_sigma 4 \
        --heatmap_branch none \
        --heatmap_size 56 \
        --image_size 128 \
        --vw 0.6 \
        --eval_single_cls \
        --tower_name CrossMambaEarlyFusion >  ${output_log}
    tail ${output_log}
    echo "Done with backbone: ${backbone}"
    echo "View the results in ${output_log}"
done
