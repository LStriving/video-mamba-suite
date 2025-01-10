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
    if args.img_width is not None and args.img_height is not None:
        processor = VideoKeypointProcessor('/mnt/cephfs/home/zhoukai/Codes/vfss/vfss_keypoint/models/pytorch/best_model_trace.pt',
                                        image_height=args.img_height, image_width=args.img_width,
                                        sigma=sigma)
    else:
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
        output_path = output_path.replace(args.video_ext, '')
        if os.path.exists(output_path) and not args.overwrite:
            continue
        if args.feature_type == 'keypoint':
            keypoint, _, _ = processor.infer_heatmaps(video_path)
            np.save(output_path, keypoint)
        elif args.feature_type == 'line':
            _, line, _ = processor.infer_heatmaps(video_path)
            np.save(output_path, line)
        elif args.feature_type == 'fusion':
            _, _, cropped_fusion = processor.infer_heatmaps(video_path)
            np.save(output_path, cropped_fusion)
        elif args.feature_type == 'all':
            keypoint, line, cropped_fusion = processor.infer_heatmaps(video_path)
            np.save(output_path + '_keypoint', keypoint)
            np.save(output_path + '_line', line)
            np.save(output_path + '_fusion', cropped_fusion)
        else:
            raise ValueError(f"Unknown feature type: {args.feature_type}")

    print(f"Processed {len(videos)} videos.")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--filter_file", type=str, default=None)
    parser.add_argument('--sigma', type=float, default=4)
    parser.add_argument("--video_ext", type=str, default='.avi')
    parser.add_argument("--feature_type", choices=['keypoint', 'line', 'fusion','all'], default='fusion')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
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
    --output_dir data/swallow/stage_2/raw_heatmap_sigma4_keypoint \
    --filter_file /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/stage2-trainval.txt \
    --sigma 4 \
    --feature_type keypoint


python tools/extract_heatmap.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/2stages/datas \
    --output_dir data/swallow/stage_2/raw_heatmap_sigma4_line \
    --filter_file /mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/stage2-trainval.txt \
    --sigma 4 \
    --feature_type line

python tools/extract_heatmap.py \
    --input_dir /mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/2stages/datas \
    --output_dir tmp/plot \
    --filter_file tmp/tmplist \
    --sigma 4 \
    --feature_type all \
    --img_width 612 \
    --img_height 612
'''