#!/bin/bash
threshold=(0.23 0.25)
devices=(0 1 2 3 4 5 6)  # 假设你有 7 个可用的 GPU
iter=0

mkdir -p outputs/diff_threshold
for i in "${threshold[@]}"
do
    if [ $iter -ge 7 ]; then
        break  # 如果已经运行了7个实验，则退出循环
    fi
    echo "Threshold: $i"
    mkdir -p tmp/threshold$i
    python eval2stage.py \
        --config2 configs/2stage/mamba_swallow_i3d_secondstage.yaml \
        --ckpt ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar \
        --ckpt2 ckpts/ckpt_swallow_stage2/ \
        --cache_dir tmp/threshold$i \
        --re-extract \
        --confidence $i \
        --test_first_stage > outputs/diff_threshold/$i.txt
    iter=$((iter+1))
done
wait  # 等待所有后台进程完成

