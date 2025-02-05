actions=(1 2 3 4 5 6 7)
config2=configs/2stage/2tower/mamba_swallow_i3d_secondstage_2tower_10ep.yaml
output_folder=$(grep 'output_folder:' "$config2" | awk -F ':' '{print $2}' | xargs)
base_name=$(basename $output_folder)
# reproduce the results first
echo "Reproducing the results"
echo "Redirecting output to outputs/$base_name/conv_fusion.log"
mkdir -p outputs/$base_name
# if output file not exists, run
# if [ ! -f outputs/$base_name/conv_fusion.log ]; then
    nohup python eval2tower.py \
    --config2 $config2 \
    --config3 configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml \
    --ckpt2 ckpts/2tower/ckpt_swallow_2tower_10ep_fromckpt_orilr/ \
    --re-extract --heatmap_dir ./tmp/heatmap0.23_sigma0.6/ \
    --heatmap \
    --heatmap_branch rgb \
    --tower_name Convfusion \
    --result_path outputs/$base_name \
    --cache_dir ./tmp/threshold0.23 > outputs/$base_name/conv_fusion.log
# fi

# for action in ${actions[@]}
# do
#     config=configs/2stage/2tower/convfusion_single/mamba_swallow_i3d_secondstage_2tower_10ep_$action.yaml
#     action_name=$(grep -oP 'desired_actions: \[\K[^]]*' $config)
#     echo "Evaling: "$action_name
#     echo "Redirecting output to outputs/$base_name/$action_name.log"
#     # if output file exists, skip
#     if [ -f outputs/$base_name/$action_name.log ]; then
#         echo "File exists, skipping"
#         continue
#     fi
#     nohup nohup python eval2tower.py \
#     --config2 $config \
#     --config3 configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml \
#     --ckpt2 ckpts/2tower/ckpt_swallow_2tower_10ep_fromckpt_orilr/ \
#     --heatmap_branch rgb \
#     --re-extract --heatmap_dir ./tmp/heatmap0.23_sigma0.6/ \
#     --tower_name Convfusion \
#     --cache_dir ./tmp/threshold0.23 > outputs/$base_name/$action_name.log
# done