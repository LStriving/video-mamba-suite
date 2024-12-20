import os
from tqdm import tqdm
import numpy as np
import argparse
import sys
sys.path.append('.')
from libs.utils import VideoKeypointProcessor



def get_file_list(file_path, ext='.avi'):
    with open(file_path, 'r') as f:
        files = f.readlines()
    files = [file.strip().split(",")[0] for file in files]
    files = [file + '.avi' for file in files if not file.endswith(ext)]
    return files

def main(args):
    sigma = args.sigma
    processor = VideoKeypointProcessor('/mnt/cephfs/home/zhoukai/Codes/vfss/vfss_keypoint/models/pytorch/best_model_trace.pt',
                                       sigma=sigma)
    input_dir = args.input_dir
    output_dir = args.output_dir
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist."
    assert input_dir != output_dir, "Input and output directories cannot be the same."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.filter_file is None:
        videos = [video for video in os.listdir(input_dir) if video.endswith(args.video_ext)]
    else:
        videos = get_file_list(args.filter_file, args.video_ext)
        print(f"Filtering {len(videos)} videos.")
    for video_name in tqdm(videos):
        video_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, video_name)
        _, _, cropped_fusion = processor.infer_heatmaps(video_path)
        np.save(output_path, cropped_fusion)

    print(f"Processed {len(videos)} videos.")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--filter_file", type=str, default=None)
    parser.add_argument('--sigma', type=float, default=4)
    parser.add_argument("--video_ext", type=str, default='.avi')
    args = parser.parse_args()
    main(args)

'''
python tools/extract_heatmap.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/2stages/datas \
    --output_dir data/swallow/stage_2/raw_heatmap_sigma4 \
    --filter_file /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/stage2-trainval.txt \
    --sigma 4

python tools/extract_heatmap.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/2stages/datas \
    --output_dir data/swallow/stage_2/raw_heatmap_sigma4 \
    --filter_file /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/stage2-test.txt \
    --sigma 4
'''