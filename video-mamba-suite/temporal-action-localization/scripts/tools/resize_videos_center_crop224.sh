#!/bin/bash

# 16
for i in {0..15}
do
    python tools/resize_videos.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/result/datas \
    --output_dir /mnt/cephfs/dataset/swallow_videos_date1216_centercrop_size224 \
    --size 224 224 \
    --file_ext .avi \
    --resume \
    --center_crop \
    --filter_file /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/tmp/stage1_video${i}.txt &
done