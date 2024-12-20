#!/bin/bash

# 16
for i in {0..15}
do
    python tools/resize_videos.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/result/datas \
    --output_dir /mnt/cephfs/dataset/swallow_videos_date1214_size128 \
    --size 128 128 \
    --file_ext .avi \
    --resume \
    --filter_file /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/tmp/stage1_video${i}.txt &
done