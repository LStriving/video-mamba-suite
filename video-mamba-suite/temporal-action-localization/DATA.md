# Data Preparation

```bash
VIDEO_DIR=/mnt/cephfs/home/yangweihao/tp/swallow_videos/videos_jing_8/
cd /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization
python tools/generate_data.py --video_dir ${VIDEO_DIR} --json_path /mnt/cephfs/home/yangweihao/tp/swallow_videos/videos_jing_8/valid_entries.json --output_dir data/swallow/external_videos2_processed/
```

```bash
python tools/convert_generate_json.py --input data/swallow/external_videos2_processed/annotations.json --output data/swallow/external_videos2_processed/converted_annotations.json
```


```bash
cd /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg
python tools/extract_flow_frame.py --video_path /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/external_videos2_processed/videos --save_path /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/external_videos2_processed/flowframes
```



```bash
cd /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg
python tmp/splittxt.py --input_file /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/external_videos2_processed/converted_annotations.json --output_prefix tmp/stage1_swallow2_0427_8all --split_num 8 --output_dir / --ext .npy
```

```bash
workspaceFolder=/mnt/cephfs/home/liyirui/project/swallow_a2net_vswg
video_dir=/mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/external_videos2_processed/videos
cmd='python tools/get_features.py 
    --img_size 128 
    --output_dir ./feat/swallow2_0427/stage1_128/no_interplote 
    --flow_dir /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/external_videos2_processed/flowframes 
    --videos_dir /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/external_videos2_processed/videos
    --flow_i3d ./pretrained/flow_imagenet.pt --rgb_i3d ./pretrained/pretrained_swallow_i3d.pth 
    --cuda  --batch_size 500  --resume  --no_interplote '

# 8 gpus
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES="${i}" ${cmd} --video_list "tmp/stage1_swallow2_0427_8all${i}.txt" &
done
wait


```bash
# eval
cd /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization
python eval.py --config /mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/configs/mamba_swallow2_0427_i3d_e2e.yaml --ckpt 'ckpts/ckpt_swallow/mamba_swallow_i3d_e2e_2025-04-08 13:09:37' --train_set
```