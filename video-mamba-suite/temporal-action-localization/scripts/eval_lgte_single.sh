actions=(1 2 3 4 5 6 7)
config2=configs/2stage/mamba_swallow_i3d_secondstage_lgte.yaml
output_folder=$(grep 'output_folder:' "$config2" | awk -F ':' '{print $2}' | xargs)
base_name=$(basename $output_folder)
# reproduce the results first
echo "Reproducing the results"
echo "Redirecting output to outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log"
mkdir -p outputs/$base_name
# if output file not exists, run
if [ ! -f outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log ]; then
    nohup python eval2stage.py --config2 $config2 --re-extract \
        --ckpt2 ./ckpts/ckpt_swallow_stage2_lgte/ckpt_swallow_stage2_lgte \
        --cache_dir tmp/threshold0.23 --image_size 128 > outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log
fi

for action in ${actions[@]}
do
    config=configs/2stage/single_action_lgte/lgte_$action.yaml
    action_name=$(grep -oP 'desired_actions: \[\K[^]]*' $config)
    echo "Evaling: "$action_name
    echo "Redirecting output to outputs/$base_name/$action_name.log"
    # if output file exists, skip
    if [ -f outputs/$base_name/$action_name.log ]; then
        echo "File exists, skipping"
        continue
    fi
    nohup python eval2stage.py \
        --config2 $config \
        --re-extract \
        --ckpt2 ./ckpts/ckpt_swallow_stage2_lgte/ckpt_swallow_stage2_lgte \
        --cache_dir tmp/threshold0.23 \
        --image_size 128 \
        --dump_result > outputs/$base_name/$action_name.log
done