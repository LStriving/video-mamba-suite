import os
import skvideo.io
import numpy as np
import torch
import cv2
import json
from tqdm import tqdm

def resize_video(video_path, output_path, size=(256, 256)):
    # Load video
    videodata = skvideo.io.vread(video_path)
    # Resize video
    resized_videodata = np.zeros((videodata.shape[0], size[0], size[1], 3), dtype=np.uint8)
    for i, frame in enumerate(videodata):
        resized_videodata[i] = cv2.resize(frame, size)
    # Save video
    skvideo.io.vwrite(output_path, resized_videodata)

def centercrop_resize_video(video_path, output_path, size=(224, 224)):
    if size[0] == 224:
        resized_len = 226
    elif size[0] == 128:
        resized_len = 130
    else:
        raise ValueError(f"Unsupported size {size}.")
    # Load video
    videodata = skvideo.io.vread(video_path)
    # Resize video
    resized_videodata = np.zeros((videodata.shape[0], size[0], size[1], 3), dtype=np.uint8)
    for i, frame in enumerate(videodata):
        h, w = frame.shape[:2]
        if w < resized_len or h < resized_len:
            d = resized_len - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        
        resized_videodata[i] = center_crop(frame, size)
    # Save video
    skvideo.io.vwrite(output_path, resized_videodata)

def center_crop(image, output_size):
    h, w = image.shape[:2]
    new_h, new_w = output_size
    i = int(np.round((h - new_h) / 2.))
    j = int(np.round((w - new_w) / 2.))
    image = image[i:i+new_h, j:j+new_w]
    return image

def get_file_list(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            file_list = json.load(f)
        file_list = [file_list.keys()]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            file_list = f.readlines()
        file_list = [file.strip() for file in file_list]
    return file_list


def resize_videos(input_dir, output_dir, size=(256, 256), file_ext='.avi', filter_file=None, resume=False):
    os.makedirs(output_dir, exist_ok=True)
    if filter_file is not None:
        if not os.path.exists(filter_file):
            raise FileNotFoundError(f"Filter file {filter_file} does not exist.")
        else:
            videos = get_file_list(filter_file)
            print(f"Filtering {len(videos)} videos.")
    else:
        videos = [video for video in os.listdir(input_dir) if video.endswith(file_ext)]
    for video_name in tqdm(videos):
        if not video_name.endswith(file_ext):
            video_name += file_ext
        video_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, video_name)
        if resume and os.path.exists(output_path):
            continue
        resize_video(video_path, output_path, size)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--filter_file", type=str, default=None)
    parser.add_argument('--size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--file_ext', type=str, default='.avi')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--center_crop', action='store_true')
    args = parser.parse_args()
    if args.center_crop:
        resize_video = centercrop_resize_video
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist.")
    if args.num_workers > 1:
        import multiprocessing
        pool = multiprocessing.Pool(args.num_workers)
        filtered_videos = get_file_list(args.filter_file) if args.filter_file is not None else None
        if filtered_videos is None:
            videos = [video for video in os.listdir(args.input_dir) if video.endswith(args.file_ext)]
        
        # map data to workers
        for video_name in videos:
            video_name = str(video_name)
            video_name = video_name.strip()
            if not video_name.endswith(args.file_ext):
                video_name += args.file_ext
            video_path = os.path.join(args.input_dir, video_name)
            output_path = os.path.join(args.output_dir, video_name)
            if args.resume and os.path.exists(output_path):
                continue
            pool.apply_async(resize_video, args=(video_path, output_path, args.size))

    else:
        resize_videos(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            size=args.size, 
            file_ext=args.file_ext, 
            filter_file=args.filter_file, 
            resume=args.resume)

'''
python tools/resize_videos.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/result/datas \
    --output_dir /mnt/cephfs/dataset/swallow_videos_date1214_size224 \
    --filter_file data/swallow/anno/swallow_singlestage.json \
    --size 224 224 \
    --resume

python tools/resize_videos.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/result/datas \
    --output_dir /mnt/cephfs/dataset/swallow_videos_date1216_centercrop_size224 \
    --filter_file data/swallow/anno/swallow_singlestage.json \
    --size 224 224 \
    --resume
'''