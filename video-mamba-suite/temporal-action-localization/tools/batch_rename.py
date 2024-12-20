import os
from tqdm import tqdm

def rename_video(video_path, new_name):
    os.rename(video_path, new_name)

root='/mnt/cephfs/home/liyirui/project/video-mamba-suite/video-mamba-suite/temporal-action-localization/data/swallow/stage_2/raw_heatmap_sigma4'

for video_name in tqdm(os.listdir(root)):
    video_path = os.path.join(root, video_name)
    new_name = video_path.replace('.avi', '')
    rename_video(video_path, new_name)