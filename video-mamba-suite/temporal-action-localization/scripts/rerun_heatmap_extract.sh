
# bash scripts/extract_heatmap/resnet.sh;
# bash scripts/extract_heatmap/vitpose.sh;
bash scripts/heatmap_e2e/mamba/vitpose.sh;
bash scripts/heatmap_e2e/mamba/resnet.sh;
bash scripts/2tower_heatmape2e/heatmap_ablation/model/channel_agg_load_heatmap_resnet.sh;
bash scripts/2tower_heatmape2e/heatmap_ablation/model/channel_agg_load_heatmap_vitpose.sh;