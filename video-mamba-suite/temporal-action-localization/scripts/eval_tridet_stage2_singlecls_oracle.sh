config=configs/2stage/mamba_swallow2_0427_i3d_train_stage1_traintest.yaml
# config2=configs/2stage/actionformer_swallow_i3d_secondstage.yaml
config2=configs/2stage/tridet_swallow_i3d_stage2.yaml
output_folder=$(grep 'output_folder:' "$config2" | awk -F ':' '{print $2}' | xargs)
base_name=$(basename $output_folder)
# reproduce the results first
# echo "Reproducing the results"
# echo "Redirecting output to outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log"
# mkdir -p outputs/$base_name
# if output file not exists, run
# if [ ! -f outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log ]; then
echo outputs/tridet_stage2_singlecls_externaltest.log
        # --ckpt2 'link2/actionformer_ckpt_swallow_stage' \
        # --ckpt2 'ckpts/link2/actionformer_ckpt_swallow_stage_max192' \
nohup python eval2stage.py --config $config --config2 $config2 --re-extract \
        --ckpt2 'ckpts/link2/tridet_swallow_second_stage' \
        --last_epoch \
        --only_perfect --train_set --infer_perfect_stage1 --perfect_stage1 ./tmp/swallow2_0427_i3d_rgb128_flow128_perfect/ \
        --seg_duration 5 \
        --cache_dir tmp/threshold0.23_swallow2 --image_size 128 > outputs/tridet_stage2_singlecls_externaltest.log
