# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch, make_two_tower
from libs.utils import (train_one_epoch, valid_one_epoch, infer_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler, twotower_train_one_epoch,
                        twotower_infer_one_epoch, fix_random_seed, ModelEma)
from train2stage import get_label_dict_from_file

def get_cfg(config_file):
    if os.path.isfile(config_file):
        cfg = load_config(config_file)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)
    return cfg

def run(cfg, cfg2, args, action_label=None):
    """1. get configuration from a yaml file"""
    args.start_epoch = 0
    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    cfg2_filename = os.path.basename(args.config2).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + cfg2_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + cfg2_filename + '_' + \
                str(args.output)+'_'+str(cfg['loader']['batch_size'])+'_'+str(cfg['opt']['learning_rate']))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    # train dataset for tower 1
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'], 
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    
    # dataset for tower 2
    train_dataset2 = make_dataset(
        cfg2['dataset_name'], True, cfg2['train_split'], **cfg2['dataset'], 
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars2 = train_dataset2.get_attributes()
    cfg2['model']['train_cfg']['head_empty_cls'] = train_db_vars2['empty_label_ids']

    # data loaders
    train_loader2 = make_data_loader(
        train_dataset2, True, rng_generator, **cfg2['loader'])
    assert len(train_dataset) == len(train_dataset2), "Training datasets must have the same length!"

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])

    """3.5 Initialize (partially) from pre-trained model"""
    if args.backbone_1:
        if os.path.isfile(args.backbone_1):
            checkpoint = torch.load(args.backbone_1,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{:s}' for tower 1".format(args.backbone_1))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.backbone_1))
            return
    if args.backbone_2:
        if os.path.isfile(args.backbone_2):
            checkpoint = torch.load(args.backbone_2,
                map_location = lambda storage, loc: storage.cuda(
                    cfg2['devices'][0]))
            model2.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{:s}' for tower 2".format(args.backbone_2))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.backbone_2))

    # two-tower model
    model = make_two_tower(args.tower_name, model, model2, cfg, cfg2)

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint('config 1:', fid)
        pprint(cfg, stream=fid)
        pprint('config 2:', fid)
        pprint(cfg2, stream=fid)
        fid.flush()

    """5. training / validation loop"""
    print("\nStart training model {:s} ...".format(args.tower_name))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    """6. create val dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    val_dataset2 = make_dataset(
        cfg2['dataset_name'], False, cfg2['val_split'], **cfg2['dataset']
    )
    val_loader2 = make_data_loader(
        val_dataset2, False, None, 1, cfg2['loader']['num_workers']
    )
    assert len(val_dataset) == len(val_dataset2), "Validation datasets must have the same length!"

    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds'],
                only_focus_on = action_label
            )
    train_label_dict = val_dataset.label_dict
    eval_label_dict = get_label_dict_from_file(val_dataset.json_file, action_label)
    remap = False
    for label, train_label_id in train_label_dict.items():
        if label in eval_label_dict:
            if train_label_id != eval_label_dict[label]:
                remap = True
                break
        else:
            print(f"Warning: {label} not found in eval_label_dict")
            remap = True
            break

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        twotower_train_one_epoch(
            train_loader,
            train_loader2,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        
        start_eval = 5 if max_epochs > 30 else 0

        if epoch>=start_eval or not cfg['opt']['warmup']:#(max_epochs//4):

            # model
            model_eval = make_meta_arch(cfg['model_name'], **cfg['model'])
            model_eval2 = make_meta_arch(cfg2['model_name'], **cfg2['model'])
            model_eval = make_two_tower(args.tower_name, model_eval, model_eval2, cfg, cfg2)
            # not ideal for multi GPU training, ok for now
            model_eval = nn.DataParallel(model_eval, device_ids=cfg['devices'])
            model_eval.load_state_dict(model_ema.module.state_dict())

            # set up evaluator
            output_file = None
            
            """5. Test the model"""
            print("\nStart testing model {:s} ...".format(cfg['model_name']))
            start = time.time()
            result = twotower_infer_one_epoch(
                val_loader,
                val_loader2,
                model_eval,
                -1,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=tb_writer,
                print_freq=999999 #args.print_freq
            )
            # remap action labels
            if remap:
                for label, train_label_id in train_label_dict.items():
                    if label in eval_label_dict:
                        result['label'][result['label'] == train_label_id] = eval_label_dict[label] + 1000
                    else:
                        print(f"Warning: {label} not found in eval_label_dict")
                result['label'] -= 1000
            _, mAP = det_eval.evaluate(result)
            if tb_writer is not None:
                tb_writer.add_scalar('validation/mAP', mAP, epoch)
            end = time.time()
            # print("All done! Total time: {:0.2f} sec".format(end - start))
            print(epoch,mAP)

            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}_{:.5f}.pth.tar'.format(epoch,mAP)
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
def main(args):
    from torch.multiprocessing import Process, set_start_method
    """main function that handles training / inference"""
    cfg = get_cfg(args.config)
    cfg2 = get_cfg(args.config2)
    
    # get stage
    stage = cfg['dataset']['stage_at']
    assert stage in [1, 2], "Stage must be 1 or 2!"
    # get desired action label
    action_label = cfg['dataset']['desired_actions']
    assert set(action_label) == set(cfg2['dataset']['desired_actions']),\
          "Action labels must be the same for two configs!"
    
    if stage == 1:
        assert len(action_label) == 1, "Stage 1 only supports one action label!"

    if cfg['dataset']['num_classes'] == 1:
        # looping over all actions
        output = args.output

        set_start_method('spawn', force=True)
        processes = []
        for rank, action in enumerate(action_label):
            p = Process(target=train_action, args=(cfg, cfg2, args, output, action, rank))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        run(cfg, cfg2, args, action_label)

def train_action(cfg, cfg2, args, output, action, rank):
    output_prefix = f'{action}_'
    args.output = f'{output_prefix}{output}'
    cfg['dataset']['desired_actions'] = [action]
    cfg['devices'] = [f'cuda:{rank}']
    cfg2['devices'] = [f'cuda:{rank}']
    run(cfg, cfg2, args, action)

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file, will use the training config')
    parser.add_argument('config2', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--backbone_1', default=None, type=str, metavar='PATH',
                        help='path to a checkpoint for tower 1(default: none)')
    parser.add_argument('--backbone_2', default=None, type=str, metavar='PATH',
                        help='path to a checkpoint for tower 2(default: none)')
    parser.add_argument('--tower_name', default='Convfusion', type=str,
                        help='name of the two-tower model (default: Convfusion)')
    args = parser.parse_args()
    main(args)
