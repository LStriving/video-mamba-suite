'''Use ResNet50 to extract heatmap features'''
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import Process, set_start_method
from torchvision.models.resnet import ResNet50_Weights


def extract_features(args, gpu_num, gpu_id):
    feats = os.listdir(args.heatmap_root)
    feats = [f for f in feats if f.endswith('.npy')]

    if args.resume:
        feats_extracted = os.listdir(args.output_root)
        feats_extracted = [f for f in feats_extracted if f.endswith('.npy')]
        feats = list(set(feats) - set(feats_extracted))

    feats = sorted(feats)
    total = len(feats)
    print(f'Total {total} samples to extract.')
    # get own feat according to gpu_id and gpu_num
    if total % gpu_num == 0:
        feats = np.split(np.array(feats), gpu_num)[gpu_id]
    else:
        jobs = total // gpu_num
        job_start_idx = gpu_id * jobs
        job_end_idx = (gpu_id + 1) * jobs
        if gpu_id == gpu_num - 1:
            job_end_idx = total
        feats = feats[job_start_idx:job_end_idx]
    print(f'GPU {gpu_id} will extract {len(feats)} samples.')

    if args.debug:
        feats = feats[:10]

    data = []
    for feat in feats:
        feat_data = np.load(os.path.join(args.heatmap_root, feat))
        feat_data = torch.from_numpy(feat_data).unsqueeze(1)
        # resize
        feat_data = torch.nn.functional.interpolate(feat_data, (args.model_input_size, args.model_input_size), mode='bilinear', align_corners=False)
        data.append(feat_data) # T, H ,W


    
    device = torch.device(f'cuda:{gpu_id}')

    feat_len = [len(d) for d in data]
    data = torch.concatenate(data, dim=0)
    # if args.model_input_size != data.shape[-1]:
    #     data = torch.nn.functional.interpolate(data, (args.model_input_size, args.model_input_size), mode='bilinear', align_corners=False)

    batch_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(device)
    # remove the last layer
    model.fc = nn.Identity()
    
    model.eval()


    extracted = []
    for batch in tqdm(batch_loader):
        with torch.no_grad():
            batch = batch.repeat(1, 3, 1, 1)
            batch = batch.to(device)
            feat = model(batch)
            # feat = F.avg_pool2d(feat, 7)
            # feat = einops.rearrange(feat, 'b c h w -> b (c h w)')
            feat = feat.cpu().numpy()
            extracted.append(feat)
    extracted = np.concatenate(extracted, axis=0)
    extracted = np.split(extracted, np.cumsum(feat_len)[:-1])

    for feat, feat_name in zip(extracted, feats):
        np.save(os.path.join(args.output_root, feat_name), feat)
            

    
def main(args):
    if not os.path.exists(args.heatmap_root):
        raise ValueError(f"heatmap_root {args.heatmap_root} does not exist.")
    
    args.output_root = args.output_root + '_' + str(args.model_input_size)
    os.makedirs(args.output_root, exist_ok=True)

    if args.gpu_num > 1:
        set_start_method('spawn', force=True)
        processes = []
        for i in range(args.gpu_num):
            p = Process(target=extract_features, args=(args, args.gpu_num, i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        extract_features(args, 1, 0)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--heatmap_root', type=str, 
                        default='/mnt/cephfs/home/zhoukai/Codes/vfss/vfss_keypoint/stage2_heatmaps/fusion')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--output_root', type=str, default='data/swallow/stage_2/resnet50_keypointfeat')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_input_size", type=int, default=56)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)

'''
python tools/resnet_extract.py \
    --resume 
    --gpu_num 7 
'''