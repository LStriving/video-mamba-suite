#!/bin/bash
init_values=(1e0 1e-2 1e-3 1e-4)

for init_value in ${init_values[@]}
do
    python train2tower.py \
    configs/2stage/2tower/crossattn/init_value/mamba_swallow_i3d_secondstage_2tower_dino_layer3_${init_value}.yaml \
    configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4.yaml \
    --tower_name DINOAttnEarlyFusion \
    --output dino  \
    --resume resume
    
    mkdir -p outputs/2tower_heatmape2e_DINOcrossattn_initv_${init_value}

    nohup python eval2tower.py \
    --config2 configs/2stage/2tower/crossattn/init_value/mamba_swallow_i3d_secondstage_2tower_dino_layer3_${init_value}.yaml \
    --config3 configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4.yaml \
    --re-extract \
    --ckpt2 ckpts/link2/2tower_3layers_dinolikeattn_${init_value} \
    --cache_dir tmp/threshold0.23 \
    --heatmap_dir tmp/raw_heatmap_sigma4_p0.23 \
    --heatmap \
    --heatmap_sigma 4 \
    --heatmap_branch none \
    --heatmap_size 56 \
    --image_size 128 \
    --tower_nam DINOAttnEarlyFusion > outputs/2tower_heatmape2e_DINOcrossattn_initv_${init_value}/eval_0.7.log

done