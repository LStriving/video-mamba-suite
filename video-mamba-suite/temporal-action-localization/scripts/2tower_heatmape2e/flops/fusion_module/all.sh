dir=scripts/2tower_heatmape2e/flops/fusion_module
bash $dir/best.sh
bash $dir/self_mamba.sh
bash $dir/valina_crossattn_alllayer2gamma.sh
bash $dir/wo_ccm.sh