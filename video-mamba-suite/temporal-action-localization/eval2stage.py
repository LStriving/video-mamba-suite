# python imports
import argparse
import os, tarfile, io
import glob
import time
import json
from collections import OrderedDict
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import infer_one_epoch, ANETdetection, fix_random_seed, valid_one_epoch

from pytorch_i3d import InceptionI3d

# thir party imports
import skvideo.io
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def run(cfg, args, action_label=None, infer_or_eval=infer_one_epoch):
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(args.ckpt,
                                     'epoch_{:03d}.pth.tar'.format(args.epoch))
        else:
            ckpt_file_list = sorted(
                glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)
    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)
    """2. create dataset / dataloader"""
    if args.train_set:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['train_split'],
                                   **cfg['dataset'])
    else:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                                **cfg['dataset'])
    # set bs = 1, and disable shuffle
    cfg['loader']['batch_size'] = 1
    val_loader = make_data_loader(val_dataset, False, None, **cfg['loader'])
    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'])
    else:
        output_file = os.path.join(
            os.path.split(ckpt_file)[0], 'eval_results.pkl')
    """5. Test the model"""
    print("\nStart inferring model {:s} ...".format(cfg['model_name']))
    start = time.time()
    result = infer_or_eval(val_loader,
                          model,
                          -1,
                          ext_score_file=cfg['test_cfg']['ext_score_file'],
                          evaluator=det_eval,
                          output_file=output_file,
                          print_freq=args.print_freq,
                          visualize=args.visualize,
                          )
    end = time.time()
    return result

def get_label_dict(json_path, desired_actions):
    label_dict = {}
    with open(json_path, 'r') as f:
        json_db = json.load(f)

    for key, value in json_db.items():
        for act in value['annotations']:
            if desired_actions is not None and act['label'] in desired_actions:
                label_dict[act['label']] = act['label_id']
            elif desired_actions is None:
                label_dict[act['label']] = act['label_id']
            elif act['label'] not in desired_actions:
                continue
    return label_dict

################################################################################
def main(args):
    ### Stage 1
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    args.saveonly = True
    eval_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                                **cfg['dataset'])
    eval_db_vars = eval_dataset.get_attributes()
    # get action id dict from json file

    if args.test_first_stage:
        det_eval_stage1 = ANETdetection(
            eval_dataset.json_file,
            eval_dataset.split[0],
            tiou_thresholds=eval_db_vars['tiou_thresholds'],
            only_focus_on=['AllTime'])
    result = run(cfg, args, action_label=cfg['dataset']['desired_actions'], infer_or_eval=infer_one_epoch)
    # {video-id:[], t-start:[], t-end:[], score:[], label:[]}
    result['t-center'] = (result['t-start'] + result['t-end']) / 2

    # filter out low confidence results
    mask = result['score'] > args.confidence
    
    new_ids = []
    for key in result.keys():
        if key == 'video-id':
            for idx in range(len(result[key])):
                if mask[idx]:
                    new_ids.append(result[key][idx])
            result[key] = new_ids
        else:
            result[key] = result[key][mask]
    if len(set(result['video-id'])) < len(eval_dataset):
        print(f"Warning: length of stage 1 is smaller than the dataset, \
              {len(result['video-id'])} vs {len(eval_dataset)}. Try to decrease confidence threshold.")

    if args.test_first_stage:
        # after filtering, we have a new result
        print('filtering results with threshold: ', args.confidence) 
        mAP = det_eval_stage1.evaluate(result)

    results = {
        'seg-id': [],
        't-start': [],
        't-end': [],
        'score': [],
        'label': [],
        'video-id': []
    }

    new_feat_path, new_json_path = None, None
    if args.re_extract:
        # re-extract features
        new_feat_path = args.cache_dir
        cfg['cache_dir'] = args.cache_dir
        # get center and extend 
        video_root = args.video_root
        assert os.path.isdir(video_root), "Video root does not exist!"
        initI3ds(args)
        # extract features
        new_feat_center = extract_features_from_res(video_root, new_feat_path, args.flow_dir, result, cfg)
        # build new json file for stage 2 dataset
        new_json_path = build_tmp_json(cfg, new_feat_center)
    else:
        # clip target features from video features
        ...
        # new feature path
        # rebuild json file
        raise NotImplementedError
        

    ### Stage 2
    if args.config2:
        if os.path.isfile(args.config2):
            cfg = load_config(args.config2)
        else:
            raise ValueError("Config file does not exist.")
        cfg['raise_error'] = args.raise_error
        args.saveonly = False
        cfg['dataset']['feat_folder'] = new_feat_path
        cfg['dataset']['json_file'] = new_json_path
        cfg['val_split'] = ['test']
        pprint(cfg)
        det_eval = ANETdetection(
            eval_dataset.json_file,
            eval_dataset.split[0],
            tiou_thresholds=eval_db_vars['tiou_thresholds'],
            only_focus_on=cfg['dataset']['desired_actions']
        )
        # get action id dict from json file
        label_dict = get_label_dict(eval_dataset.json_file, cfg['dataset']['desired_actions'])
        if cfg['dataset']['num_classes'] == 1:  # single classification
            for action in cfg['dataset']['desired_actions']:
                action_id = label_dict[action]
                # search action ckpt dir
                action_ckpt_dirs = os.listdir(args.ckpt2)
                action_ckpt_dirs = [ckpt_dir for ckpt_dir in action_ckpt_dirs if action in ckpt_dir]
                if len(action_ckpt_dirs) > 1:
                    print(f"Warning: Multiple ckpt dirs found for action {action}, using the first one.")
                elif len(action_ckpt_dirs) == 0:
                    print(f"Error: No ckpt dir found for action {action}.")
                    raise FileNotFoundError
                action_ckpt_dir = os.path.join(args.ckpt2, action_ckpt_dirs[0])
                args.ckpt = get_best_pth_from_dir(action_ckpt_dir)
                print(f"Using ckpt: {args.ckpt}")
                cfg['dataset']['desired_actions'] = [action]
                result = run(cfg, args, action_label=action, infer_or_eval=infer_one_epoch)

                # gather results (numpy) from different actions
                results['seg-id'].extend(result['video-id'])
                results['t-start'].extend(result['t-start'].tolist())
                results['t-end'].extend(result['t-end'].tolist())
                results['score'].extend(result['score'].tolist())
                results['label'].extend([action_id]* len(result['score']))
                results['video-id'].extend([i.split("#")[0] for i in result['video-id']])
                

            # transform
            results['t-start'] = torch.tensor(results['t-start']).numpy()
            results['t-end'] = torch.tensor(results['t-end']).numpy()
            results['label'] = torch.tensor(results['label']).numpy()
            results['score'] = torch.tensor(results['score']).numpy()
        else:                                   # multiple classification
            args.ckpt = args.ckpt2
            result = run(cfg, args, action_label=cfg['dataset']['desired_actions'], infer_or_eval=infer_one_epoch)
            result['label'][result['label'] == 0] = len(label_dict) # correct the label
        # shift t-start and t-end based on stage 1 segment results (+ t-center - 2)
        for idx in range(len(results['video-id'])):
            # get seg_id
            seg_id = results['seg-id'][idx]
            center = new_feat_center[seg_id]
            # shift
            results['t-start'][idx] += center - 2
            results['t-end'][idx] += center - 2
        # evaluate
        mAP = det_eval.evaluate(results) # should be evaluated on the original video rather than the clipped video
        print(f"mAP: {mAP}")

def extract_features_from_res(video_root, new_feat_path, flow_dir, result, cfg):
    # define some constants
    IMAGE_SIZE = 128
    CLIP_DUR = 4
    WINDOW_SIZE = cfg['dataset']['num_frames']
    WINDOW_STEP = cfg['dataset']['feat_stride']
    # get center and extend 
    cache_video_id = None
    rgb_data = None
    flow_data = None
    new_feats = {}
    per_video_rank = 0
    for idx in tqdm(range(len(result['video-id']))):
        video_id = result['video-id'][idx]
        t_center = result['t-center'][idx]
        t_extend = CLIP_DUR # 4 seconds
        clip_start = t_center - t_extend / 2
        clip_end = t_center + t_extend / 2
        duration = int(video_id.split("_")[-1])
        clip_start = max(clip_start, 0)
        clip_end = min(clip_end, duration)
        start_ratio = clip_start / duration
        end_ratio = clip_end / duration
        
        if cache_video_id != video_id:
            cache_video_id = video_id
            per_video_rank = 0
            video_path = os.path.join(video_root, f"{video_id}.avi")
            assert os.path.isfile(video_path), "Video file does not exist!"
            try:
                rgb_data = skvideo.io.vread(video_path)
            except Exception as e:
                print(f"Error reading video: {video_path}")
                if cfg['raise_error']:
                    raise e
                else:
                    print(e)
                continue
            # resize video
            rgb_data=torch.from_numpy(rgb_data)
            rgb_data_tmp=torch.zeros(rgb_data.shape[0:1]+(IMAGE_SIZE,IMAGE_SIZE,3)).double()
            for index,rgb_data in enumerate(rgb_data):
                rgb_datum_tmp=torch.from_numpy(cv2.resize(rgb_data.numpy(),(IMAGE_SIZE,IMAGE_SIZE))).double()
                rgb_data_tmp[index,:,:,:]=rgb_datum_tmp
            rgb_data=rgb_data_tmp.view(-1,IMAGE_SIZE,IMAGE_SIZE,3) / 127.5 - 1
    
            flow_data=get_flow_frames_from_targz(os.path.join(flow_dir, f"{video_id}.tar.gz"))
            flow_data_tmp=torch.zeros(flow_data.shape[0:1]+(IMAGE_SIZE,IMAGE_SIZE,2)).double()
            for index,flow_data in enumerate(flow_data):
                flow_datum_tmp=torch.from_numpy(cv2.resize(flow_data.numpy(),(IMAGE_SIZE,IMAGE_SIZE))).double()
                flow_data_tmp[index,:,:,:]=flow_datum_tmp
            flow_data=flow_data_tmp
            flow_data=flow_data.view(-1,IMAGE_SIZE,IMAGE_SIZE,2)
        else:
            per_video_rank += 1
        
        # slidedata_flow=slideTensor(flow_data,args.chunk_size,args.frequency)
        saved_new_feat_name = video_id + f"#{per_video_rank}"
        os.makedirs(new_feat_path, exist_ok=True)
        new_feat_file = os.path.join(new_feat_path, f"{saved_new_feat_name}.npy")
        # extract features
        extract_features(rgb_data, flow_data, new_feat_file, start_ratio, end_ratio, WINDOW_SIZE, WINDOW_STEP)
        new_feats[saved_new_feat_name] = t_center
        # if len(new_feats) == 2: # debug
        #     build_tmp_json(cfg, new_feats)
    return new_feats

def get_flow_frames_from_targz(tar_dir):
    '''ref to swallow_a2net_vswg/tools/eval.py'''
    list_u=[]
    list_v=[]
    with tarfile.open(tar_dir) as tar:
        mems=sorted(tar.getmembers(),key=lambda x:x.path)
        for x in mems:
           if(x.size==0):
               continue
           filelikeobject=tar.extractfile(x)
           r=filelikeobject.read()
           bytes_stream = io.BytesIO(r)
           roiimg=Image.open(bytes_stream)
           nparr=np.array(roiimg,dtype=np.float64)
           norm_data=nparr/127.5-1
           if(x.path.split("/")[1]=="u"):
               list_u.append(torch.tensor(norm_data))
           else:
               list_v.append(torch.tensor(norm_data))
    res_tensor=torch.stack([torch.stack(list_u),torch.stack(list_v)],dim=3)
    return res_tensor

def slideTensor(datas,window_size,step):
    start=0
    len=datas.shape[0]
    window_datas=[]
    while(start<len):
        if(start+window_size>len-1):
            break
        window_datas.append(datas[start:start+window_size,:,:,:])
        start+=step
    result=torch.stack(window_datas, 0) # (num_windows, window_size, w, h, c)
    return result

def extract_features(rgb_data, flow_data, new_feat_file, start_ratio, end_ratio, win_size, win_step):
    rgb_time_long = rgb_data.shape[0]
    flow_time_long = flow_data.shape[0]
    # clip
    rgb_start_idx = max(0, int(start_ratio * rgb_time_long))
    rgb_end_idx = min(rgb_time_long, int(end_ratio * rgb_time_long))
    flow_start_idx = max(0, int(start_ratio * flow_time_long))
    flow_end_idx = min(flow_time_long, int(end_ratio * flow_time_long))
    rgb_data = rgb_data[rgb_start_idx:rgb_end_idx]
    flow_data = flow_data[flow_start_idx:flow_end_idx]
    # slide
    rgb_data = slideTensor(rgb_data, win_size, win_step)
    flow_data = slideTensor(flow_data, win_size, win_step)
    # get feat
    feat_spa=get_features(rgb_data,"rgb", batch_size=-1)
    feat_tem=get_features(flow_data,"flow", batch_size=-1)
    feat_spa=torch.from_numpy(feat_spa)
    feat_tem=torch.from_numpy(feat_tem)
    # concat rgb and flow features
    feat = np.concatenate([feat_spa, feat_tem], axis=1)
    # save feat
    np.save(new_feat_file, feat)
    return feat

def get_features(data, mode, batch_size=32):
    '''
    data: (T, 8, W, H, C)
    mode: 'rgb' or 'flow'
    return data with shape: (T, 1024)
    '''
    data = data.permute(0, 4, 1, 2, 3) # (T, C, 8, W, H)
    if batch_size == -1:
        batch_size = data.shape[0]
    batch_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    if mode == 'flow':
        i3d = i3d_flow
    else:
        i3d = i3d_rgb
    
    all_features = []
    for batch in batch_loader:
        batch = batch.float().cuda()
        with torch.no_grad():
            features = i3d.extract_features(batch)
        # output: (batch_size, 1024, 1, 1, 1)
        features = features[:, :, 0, 0, 0].cpu().numpy()
        all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)
    return all_features

def build_tmp_json(cfg, new_feat_center):
    old_json = cfg['dataset']['json_file']
    with open(old_json, 'r') as f:
        data = json.load(f)

    new_data = {}
    for seg_id, t_center in new_feat_center.items():
        shift = t_center - 2
        video_id = seg_id.split("#")[0]
        new_data[seg_id] = data[video_id].copy()
        new_data[seg_id]['duration'] = 4.0
        new_data[seg_id]['annotations'] = []

        for anno in data[video_id]['annotations']:
            new_anno = anno.copy()
            new_anno['segment'][0] -= shift
            new_anno['segment'][1] -= shift

            if new_anno['segment'][0] < 0 or new_anno['segment'][1] < 0:
                continue
            if new_anno['segment'][0] > 4 or new_anno['segment'][1] > 4:
                continue

            new_anno.pop("segment(frames)")
            new_data[seg_id]['annotations'].append(new_anno)
    new_json_path = os.path.join(cfg['cache_dir'], 'tmp.json')
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    return new_json_path

def initI3ds(args):
    global i3d_rgb,i3d_flow
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

def get_best_pth_from_dir(dir) -> str:
    assert os.path.isdir(dir), "Directory does not exist!"
    ckpts = os.listdir(dir)
    ckpts = [ckpt for ckpt in ckpts if ".pth.tar" in ckpt]
    ckpts = sorted(ckpts, key=lambda x: float(x.split(".pth.tar")[0].split("_")[-1]), reverse=True)
    return os.path.join(dir, ckpts[0])

################################################################################
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument(
        '--config',
        type=str,
        metavar='DIR',
        default='./configs/exp/surgery_i3d_rgb_shuffle_w37.yaml',
        help='path to a config file for stage 1')
    parser.add_argument('--config2',
                        type=str,
                        metavar='DIR',
                        default='',
                        help='path to a config file for stage 2')
    parser.add_argument('--ckpt2',
                        type=str,
                        metavar='DIR',
                        default='',
                        help='path to checkpoints for stage 2')
    parser.add_argument('--ckpt',
                        type=str,
                        metavar='DIR',
                        default='ckpt/surgery_i3d_rgb_shuffle_w37_m9216',
                        help='path to a checkpoint')
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t',
                        '--topk',
                        default=-1,
                        type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument(
        '--saveonly',
        action='store_true',
        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument("-v",
                        "--visualize",
                        action='store_true',
                        help="visualize the results")
    parser.add_argument("--train_set", action='store_true', help="eval train set")
    parser.add_argument('--confidence', type=float, default=0.3, help='confidence threshold for stage 1 results')
    parser.add_argument("--re-extract", action='store_true', help="whether to re-extract features at stage 2")
    parser.add_argument('--video_root', type=str, metavar='DIR', default='/mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/result/datas', help='path to video root (specific for stage 2)')
    parser.add_argument('--cache_dir', type=str, metavar='DIR', default='./cache', help='path to cache dir for the extracted features (specific for stage 2)')
    parser.add_argument('--flow_dir', type=str, metavar='DIR', default='/mnt/cephfs/home/chenzhuokun/git/swallowProject/result/flow_frames', help='path to flow dir (specific when re-extract)')
    parser.add_argument("--flow_i3d", type=str, metavar='DIR', default='/mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/pretrained/flow_imagenet.pt', help='path to flow i3d model')
    parser.add_argument("--rgb_i3d", type=str, metavar='DIR', default='/mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/pretrained/pretrained_swallow_i3d.pth', help='path to rgb i3d model')
    parser.add_argument("--raise_error", action='store_true', help="raise error when video reading error")
    parser.add_argument("--test_first_stage", action='store_true', help="test first stage on AllTime")
    args = parser.parse_args()
    main(args)