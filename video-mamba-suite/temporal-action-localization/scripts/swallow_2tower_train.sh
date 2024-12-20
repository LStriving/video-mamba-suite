# exp 1 
#   from ckpt ori lr
python train2tower.py ./configs/2stage/2tower/mamba_swallow_i3d_secondstage_2tower_10ep_fromckpt_lr\*10.yaml configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml --backbone_1 ckpts/ckpt_swallow_stage2 --backbone_2 ckpts/ckpt_swallow_heatmap_stage2/ --tower_name Convfusion
#   from scratch ori lr
python train2tower.py ./configs/2stage/2tower/mamba_swallow_i3d_secondstage_2tower_10ep.yaml configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml  --tower_name Convfusion

# eval
#
nohup python eval2tower.py --config2 configs/2stage/2tower/mamba_swallow_i3d_secondstage_2tower_10ep_fromckpt_lr\*10.yaml --config3 ./configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml --ckpt2 ckpts/2tower_10ep_initfromckpt_lr1e-4 --re-extract --cache_dir ./tmp/multi_class/ --heatmap_dir tmp/heatmap/ > outputs/convfusion_ckpt_orilr_result.log 

nohup python eval2tower.py --config2 configs/2stage/2tower/mamba_swallow_i3d_secondstage_2tower_10ep.yaml --config3 configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml --ckpt2 ckpts/ckpt_swallow_2tower_10ep/ --re-extract --heatmap_dir ./tmp/heatmap/ --cache_dir ./tmp/multi_class/ > outputs/convfusion_scratch_result.log 