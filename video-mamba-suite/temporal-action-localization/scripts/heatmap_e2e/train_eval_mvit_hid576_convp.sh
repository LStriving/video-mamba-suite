config_name=configs/2stage/heatmap/e2e/heatmap_secondstage_mvit_mvit_p2l3_ep30_sigma4_hid576_con.yaml
#!/bin/bash
python train2stage.py \
    ${config_name} \
    --resume resume

python eval2stage.py \
    --config2 ${config_name} \
    --ckpt2 ckpts/link2/e2e_heatmap_stage2_mvitNmvit_ep30_sigma4_hid576_conv \
    --heatmap_size 56 \
    --heatmap_branch none \
    --cache_dir tmp/raw_heatmap_sigma4 \
    --heatmap \
    --re-extract