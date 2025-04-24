actions=(1 2 3 4 5 6 7)
# config2=configs/2stage/heatmap/e2e/mamba/video_mamba/heatmap_secondstage_videomamba_l3_avgtoken_ep45_sigma4_hid576_noact.yaml
for action in ${actions[@]}
do
    config=configs/tmp_actionmamba_e2e_single/$action.yaml
    output_folder=$(grep 'output_folder:' "$config" | awk -F ':' '{print $2}' | xargs)
    action_name=$(grep -oP 'desired_actions: \[\K[^]]*' $config)
    echo $action_name
    base_name=$(basename $output_folder)
    mkdir -p outputs/$base_name
    echo "Redirecting output to outputs/$base_name/$action_name.log"
    nohup python eval.py \
        --config $config \
        --ckpt 'ckpts/ckpt_swallow/mamba_swallow_i3d_e2e_2025-04-08 13:09:37/epoch_006_0.59017.pth.tar' > outputs/$base_name/$action_name.log
done