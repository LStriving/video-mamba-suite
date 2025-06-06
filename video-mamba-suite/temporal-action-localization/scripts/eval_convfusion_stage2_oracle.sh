# actions=(1 2 3 4 5 6 7)
config2=configs/2stage/2tower/mamba_swallow_i3d_secondstage_2tower_10ep.yaml
output_folder=$(grep 'output_folder:' "$config2" | awk -F ':' '{print $2}' | xargs)
base_name=$(basename $output_folder)
# reproduce the results first
echo "Redirecting output to outputs/$base_name/swallow2_oralce_conv_fusion.log"
mkdir -p outputs/$base_name
heatmap_dir=./tmp/swallow2_perfect_heatmap_sigma0.6_i3d
perfect_stage1_dir=./tmp/swallow2_0427_i3d_rgb128_flow128_perfect
if [ ! -d $heatmap_dir ]; then
    mkdir -p $heatmap_dir
    cp $perfect_stage1_dir/tmp.json $heatmap_dir/
    cp $perfect_stage1_dir/epoch_024_0.82621_perfect.pkl $heatmap_dir/epoch_024_0.82621.pkl
fi
# if output file not exists, run
# if [ ! -f outputs/$base_name/conv_fusion.log ]; then
nohup \
python eval2tower.py \
    --config configs/2stage/mamba_swallow2_0427_i3d_train_stage1_traintest.yaml \
    --config2 $config2 \
    --config3 configs/2stage/heatmap/mamba_swallow_heatmap_secondstage.yaml \
    --ckpt2 ckpts/2tower/ckpt_swallow_2tower_10ep_fromckpt_orilr/ \
    --re-extract --heatmap_dir $heatmap_dir \
    --perfect_stage1 $perfect_stage1_dir \
    --tower_name Convfusion \
    --video_root data/swallow/external_videos2_processed/videos \
    --flow_dir data/swallow/external_videos2_processed/flowframes \
    --seg_duration 5 \
    --heatmap_size 56 \
    --image_size 128 \
    --heatmap \
    --heatmap_sigma 0.6 \
    --heatmap_branch rgb \
    --train_set \
    --only_perfect \
    --infer_perfect_stage1 \
    --selected_index -1 \
    --cache_dir $heatmap_dir \
    > outputs/$base_name/swallow2_oralce_conv_fusion.log

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