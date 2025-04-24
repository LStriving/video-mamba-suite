config_name=configs/2stage/heatmap/e2e/mamba/heatmap_secondstage_2Mmamba_Mmamba_p2l3_ep30_sigma4_hid576_gamma0.1_drop0.1_maxpool.yaml
#!/bin/bash
# python train2stage.py \
#     ${config_name} 

python eval2stage.py \
    --config2 ${config_name} \
    --ckpt2 ckpts/link2/e2e_heatmap_stage2_2MmambaNMmamba_ep30_sigma4_gamma0.1_drop.1_maxpool_hid576 \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --heatmap \
    --re-extract