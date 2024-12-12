# python imports
import argparse
import os
import time
import datetime
import pickle
from pprint import pprint
import numpy as np
from tqdm import tqdm
import glob

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.datasets.swallow import MultiModalDataset
from libs.modeling import make_meta_arch, make_two_tower
from libs.utils import (infer_one_epoch, ANETdetection,
                        fix_random_seed)
from train2stage import get_label_dict_from_file
from eval2stage import get_best_pth_from_dir, get_label_dict, initI3ds, build_tmp_json, extract_features_from_res, remap_action_labels, run

def get_cfg(config_file):
    if os.path.isfile(config_file):
        cfg = load_config(config_file)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    return cfg

def run2tower(cfg, cfg2, args, action_label=None):
    # get checkpoint file
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

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # set batch size to 1 for evaluation
    cfg['loader']['batch_size'] = 1
    cfg2['loader']['batch_size'] = 1
    if args.train_set:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['train_split'],
                                   **cfg['dataset'])
        val_dataset2 = make_dataset(cfg2['dataset_name'], False, cfg2['train_split'],
                                   **cfg2['dataset'])
    else:
        val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'],
                                **cfg['dataset'])
        val_dataset2 = make_dataset(cfg2['dataset_name'], False, cfg2['val_split'],
                                **cfg2['dataset'])
    multi_valdataset = MultiModalDataset(val_dataset, val_dataset2) 
    cfg['loader']['batch_size'] = 1
    val_loader = make_data_loader(multi_valdataset, False , rng_generator, **cfg['loader'])
    
    # load model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])
    # two-tower model
    model = make_two_tower(args.tower_name, model, model2, cfg, cfg, **cfg['two_tower'])
    # not ideal for multi GPU training, ok for now
    model_eval = nn.DataParallel(model, device_ids=cfg['devices'])

    """load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))
    # load ema model instead
    print("Loading from EMA model ...")
    model_eval.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    """5. Test the model"""
    print("\nStart inferring model {:s} ...".format(cfg['model_name']))

    
    train_label_dict = val_dataset.label_dict
    eval_label_dict = get_label_dict_from_file(cfg['dataset']['json_file'], action_label)

    
    result = infer_one_epoch(
        val_loader,
        model_eval,
        -1,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=999999 #args.print_freq
    )
    # remap action labels
    result = remap_action_labels(result, train_label_dict, eval_label_dict)
    return result

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
        
    # if cache exists, load it
    save_cache_name = os.path.basename(args.ckpt).split(".pth.tar")[0] + ".pkl"
    save_cache_path = os.path.join(args.cache_dir, save_cache_name)
    cfg['cache_dir'] = args.cache_dir
    cfg['heatmap_dir'] = args.heatmap_dir
    cfg['image_size'] = args.image_size
    cfg['heatmap_size'] = args.heatmap_size
    if os.path.isfile(save_cache_path):
        print(f"Loading cache from {save_cache_path}")
        with open(save_cache_path, 'rb') as f:
            cache = pickle.load(f)
        result = cache['result']
        new_feat_center = cache['new_feat_center']
        new_feat_path = args.cache_dir
        new_json_path = cache['new_json_path']
    else:
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

        extracted_feat_num = len(result['video-id'])
        new_feat_path, new_json_path = None, None
        # re-extract features
        new_feat_path = args.cache_dir

        os.makedirs(new_feat_path, exist_ok=True)
        
        os.makedirs(cfg['heatmap_dir'], exist_ok=True)
        os.makedirs(cfg['cache_dir'], exist_ok=True)
        if args.re_extract:
            # get center and extend 
            video_root = args.video_root
            assert os.path.isdir(video_root), "Video root does not exist!"
            initI3ds(args)
            # extract features
            # extract rgb+flow features
            # if not extracted
            cfg['cropped_videos'] = args.cropped_videos_visual
            cfg['heatmap'] = False
            new_feat_center = extract_features_from_res(video_root, new_feat_path, args.flow_dir, result, cfg)
        else: # much faster but worser performance (drop 10% mAP)
            CLIP_DUR = 4.004
            # get the original feature
            old_feat_root = cfg['dataset']['feat_folder']
            video_rank, cur_video_id = 0, None
            new_feat_center = {}
            for idx, res in tqdm(enumerate(result['video-id'])):
                feat_path = os.path.join(old_feat_root, f"{res}.npy")
                assert os.path.isfile(feat_path), "Feature file does not exist!"
                if cur_video_id != res:
                    cur_video_id = res
                    video_rank = 0
                video_rank += 1
                seg_id = f'{res}#{video_rank}'
                crop_feat_path = os.path.join(new_feat_path, f"{seg_id}.npy")
                data = np.load(feat_path)
                duration = int(res.split("_")[-1])
                clip_start = result['t-center'][idx] - CLIP_DUR / 2
                clip_end = result['t-center'][idx] + CLIP_DUR / 2
                clip_start = max(clip_start, 0)
                clip_end = min(clip_end, duration)
                start_ratio = clip_start / duration
                end_ratio = clip_end / duration
                start_idx = int(start_ratio * data.shape[0])
                end_idx = int(end_ratio * data.shape[0])
                new_data = data[start_idx:end_idx]
                np.save(crop_feat_path, new_data)
                new_feat_center[seg_id] = result['t-center'][idx]
            
        # build new json file for stage 2 dataset
        new_json_path = build_tmp_json(cfg, new_feat_center)
        # save pickle 
        cache = {
            'result': result,
            'new_feat_center': new_feat_center,
            'new_json_path': new_json_path
        }
        with open(save_cache_path, 'wb') as f:
            pickle.dump(cache, f)

    # check heatmap features extracted
    # extract heatmap features
    cfg['heatmap_dir'] = args.heatmap_dir
    os.makedirs(cfg['heatmap_dir'], exist_ok=True)
    heatmap_feats = [i for i in os.listdir(cfg['heatmap_dir']) if i.endswith('.npy')]
    if len(heatmap_feats) != len(result['video-id']):
        if args.heatmap_branch == 'rgb':
            args.rgb = args.heatmap_i3d
            initI3ds(args)
        elif args.heatmap_branch == 'flow':
            args.flow = args.heatmap_i3d
            initI3ds(args)
        elif args.heatmap_branch == '' or args.heatmap_branch is None or args.heatmap_branch.lower() == 'none':
            pass
        else:
            raise ValueError(f"Invalid heatmap branch: {args.heatmap_branch}")
        cfg['heatmap_branch'] = args.heatmap_branch
        # continue to extract heatmap features
        cfg['cropped_videos'] = args.cropped_videos_heatmap
        cfg['heatmap'] = args.heatmap
        extract_features_from_res(args.video_root, cfg['heatmap_dir'], None, result, cfg)

    results = {
            'seg-id': [],
            't-start': [],
            't-end': [],
            'score': [],
            'label': [],
            'video-id': []
    }


    if args.config2:
        if os.path.isfile(args.config2):
            cfg = load_config(args.config2)
        else:
            raise ValueError(f"Config file {args.config2} does not exist.")
        if os.path.isfile(args.config3):
            cfg2 = load_config(args.config3)
        else:
            raise ValueError(f"Config file {args.config3} does not exist.")
        cfg['raise_error'] = cfg2['raise_error'] = args.raise_error
        args.saveonly = False
        cfg['dataset']['feat_folder'] = new_feat_path # cache dir
        cfg2['dataset']['feat_folder'] = args.heatmap_dir
        cfg['dataset']['json_file'] = new_json_path
        cfg2['dataset']['json_file'] = new_json_path
        cfg['val_split'] = ['test']
        cfg2['val_split'] = ['test']
        pprint(cfg)
        pprint(cfg2)
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
                cfg2['dataset']['desired_actions'] = [action]
                result = run2tower(cfg, cfg2, args, action_label=action)

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
            results = run2tower(cfg, cfg2, args, action_label=cfg['dataset']['desired_actions'])
            results['seg-id'] = results['video-id']
            results['video-id'] = [i.split("#")[0] for i in results['video-id']]
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
        print(f"mAP: {mAP[0] * 100}")


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR',default='configs/2stage/mamba_swallow_i3d_eval_stage1.yaml',
                        help='path to a config file, stage 1')
    parser.add_argument('--config2', metavar='DIR',
                        help='path to a config file, stage 2 tower 1')
    parser.add_argument('--config3', metavar='DIR',
                        help='path to a config filem stage 2 tower 2')
    parser.add_argument('--ckpt',
                        type=str,
                        metavar='DIR',
                        default='ckpts/ckpt_swallow/mamba_swallow_i3d_stage1_mamba_swallow_stage1_2_0.0001/epoch_024_0.82621.pth.tar',
                        help='path to a checkpoint')
    parser.add_argument('-t',
                        '--topk',
                        default=-1,
                        type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument(
        '--saveonly',
        action='store_true',
        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument("--train_set", action='store_true', help="eval train set")
    parser.add_argument("--re-extract", action='store_true', help="whether to re-extract features at stage 2")
    parser.add_argument('--video_root', type=str, metavar='DIR', default='/mnt/cephfs/ec/home/chenzhuokun/git/swallowProject/result/datas', help='path to video root (specific for stage 2)')
    parser.add_argument('--cache_dir', type=str, metavar='DIR', default='./cache', help='path to cache dir for the extracted features (specific for stage 2)')
    parser.add_argument('--flow_dir', type=str, metavar='DIR', default='/mnt/cephfs/home/chenzhuokun/git/swallowProject/result/flow_frames', help='path to flow dir (specific when re-extract)')
    parser.add_argument("--flow_i3d", type=str, metavar='DIR', default='/mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/pretrained/flow_imagenet.pt', help='path to flow i3d model')
    parser.add_argument("--rgb_i3d", type=str, metavar='DIR', default='/mnt/cephfs/home/liyirui/project/swallow_a2net_vswg/pretrained/pretrained_swallow_i3d.pth', help='path to rgb i3d model')
    parser.add_argument('--heatmap_i3d', type=str, metavar='DIR', default='/mnt/cephfs/home/zhoukai/Codes/vfss/vfss_tal/log/lr0_001_bs8_i3d_swallow_ce_224_rot30_prob0_8/best_ckpt.pt',help='path for heatmap i3d model')
    parser.add_argument("--raise_error", action='store_true', help="raise error when video reading error")
    parser.add_argument("--test_first_stage", action='store_true', help="test first stage on AllTime")
    parser.add_argument("--heatmap", action='store_true', help="use heatmap as input")
    parser.add_argument("--heatmap_dir", type=str, metavar='DIR', default='tmp/heatmap', help='path to heatmap dir')
    parser.add_argument('--ckpt2',
                        type=str,
                        metavar='DIR',
                        default='',
                        help='path to checkpoints for stage 2')
    parser.add_argument('--tower_name', default='Convfusion', type=str,
                        help='name of the two-tower model (default: Convfusion)')
    parser.add_argument('--confidence', default=0.3, type=float,
                        help='confidence threshold for stage 1')
    parser.add_argument('--cropped_videos_visual', action='store_true', help='the input (video_root) are cropped videos')
    parser.add_argument('--cropped_videos_heatmap', action='store_true', help='the heatmap input (video_root) are cropped videos')
    parser.add_argument('--heatmap_branch', choices=['rgb', 'flow'], default='rgb', help='which branch to extract heatmap features')
    parser.add_argument('--heatmap_size', type=int, default=224, help='size of heatmap')
    parser.add_argument('--image_size', type=int, default=128, help='size of image')
    args = parser.parse_args()
    main(args)
