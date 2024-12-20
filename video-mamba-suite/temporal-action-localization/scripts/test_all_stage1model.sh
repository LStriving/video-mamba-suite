ckpt_root=ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001

# get all ckpt files under the ckpt_root
ckpt_files=$(find $ckpt_root -name "*.pth.tar")

# test all ckpt files
for ckpt_file in $ckpt_files
do
    basename=$(basename $ckpt_file)
    echo "Testing $ckpt_file"
    python eval.py --config configs/2stage/mamba_swallow_i3d_train_stage1_traintest.yaml --ckpt $ckpt_file > outputs/stage1_test/$basename.log
done