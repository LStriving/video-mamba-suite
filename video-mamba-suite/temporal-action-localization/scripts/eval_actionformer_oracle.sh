config=configs/2stage/mamba_swallow2_0427_i3d_train_stage1_traintest.yaml
config2=configs/actionformer_swallow_i3d_e2e.yaml
output_folder=$(grep 'output_folder:' "$config2" | awk -F ':' '{print $2}' | xargs)
base_name=$(basename $output_folder)
# reproduce the results first
# echo "Reproducing the results"
# echo "Redirecting output to outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log"
# mkdir -p outputs/$base_name
# if output file not exists, run
# if [ ! -f outputs/$base_name/mamba_swallow_i3d_secondstage_lgte.log ]; then
nohup python eval2stage.py --config $config --config2 $config2 --re-extract \
        --ckpt2 'ckpts/actionformer_e2e_swallow/actionformer_swallow_i3d_e2e_2025-01-25 23:06:50/epoch_007_0.59890.pth.tar' \
        --only_perfect --train_set --infer_perfect_stage1 --perfect_stage1 ./tmp/swallow2_0427_i3d_rgb128_flow128_perfect/ \
        --seg_duration 5 \
        --cache_dir tmp/threshold0.23_swallow2 --image_size 128 > outputs/actionformer_externaltest.log
echo outputs/actionformer_externaltest.log
