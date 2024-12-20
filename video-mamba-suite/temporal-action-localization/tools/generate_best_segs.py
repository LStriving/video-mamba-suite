'''generate best segment for according to the ground truth'''

import os
import numpy as np
import argparse
import json
from tqdm import tqdm
import skvideo.io
import torch
from collections import OrderedDict
from pytorch_i3d import InceptionI3d


def main(args):
    '''
    load ground truth
    generate best segment
    extract feats from i3d
    save pickle with `result`
    '''
    # Load ground truth
    with open(args.gt_file, 'r') as f:
        gt = json.load(f)
    # filter all time action label and generate best segment
    all_time_seg = {}
    for video_name, video_info in gt.items():
        rank = 0
        anno = video_info['annotations']
        for action in anno:
            if action['label'] != 'AllTime':
                continue
            start = action['segment'][0]
            end = action['segment'][1]
            center = (start + end) / 2
            new_start = max(0, center - args.seg_duration / 2)
            new_end = min(video_info['duration'], center + args.seg_duration / 2)
            rank += 1
            all_time_seg[video_name + f'_{rank}'] = \
            {
                'start': new_start,
                'end': new_end
            }
    # Extract features from i3d
    # Load i3d model
    i3d_rgb, i3d_flow = load_i3d(args)
    # Extract features


    
def load_i3d(args):
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_flow.load_state_dict(torch.load(args.flow_i3d))
    i3d_rgb = InceptionI3d(7, in_channels=3)
    new_kv=OrderedDict()
    old_kv=torch.load(args.rgb_i3d)['state_dict']
    for k,v in old_kv.items():
        new_kv[k.replace("module.","")]=v
    i3d_rgb.load_state_dict(new_kv)
    #i3d_rgb = InceptionI3d(400, in_channels=3)
    #i3d_rgb.load_state_dict(torch.load(load_model_rgb))
    i3d_rgb.train(False)
    i3d_flow.train(False)
    i3d_rgb.cuda()
    i3d_flow.cuda()
    return i3d_rgb, i3d_flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='')
    parser.add_argument("--flow_dir", type=str, default="")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gt_file', type=str, default='')
    parser.add_argument("--seg_duration", type=float, default=4.004)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument("--i3d_rgb", type=str, default="")
    parser.add_argument("--i3d_flow", type=str, default="")


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.i3d_rgb):
        raise FileNotFoundError(f"i3d_rgb model {args.i3d_rgb} does not exist.")
    if not os.path.exists(args.i3d_flow):
        raise FileNotFoundError(f"i3d_flow model {args.i3d_flow} does not exist.")
    if not os.path.exists(args.gt_file):
        raise FileNotFoundError(f"Ground truth file {args.gt_file} does not exist.")
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory {args.video_dir} does not exist.")

    main(args)
