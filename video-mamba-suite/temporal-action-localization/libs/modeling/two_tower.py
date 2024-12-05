import torch
import torch.nn as nn

from .meta_archs import PtTransformerClsHead, PtTransformerRegHead
from .models import register_two_tower, make_neck


@register_two_tower('Convfusion')
class Convfusion(nn.Module):
    def __init__(self, visual_tower, heatmap_tower, cfg1, cfg2, include_ori_loss=False):
        super(Convfusion, self).__init__()
        self.visual_tower = visual_tower
        self.heatmap_tower = heatmap_tower
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self.include_ori_loss = include_ori_loss
        model_cfg = cfg1['model']

        embd_dim = model_cfg['embd_dim'] + cfg2['model']['embd_dim']
        backbone_arch = model_cfg['backbone_arch']
        fpn_dim = model_cfg['fpn_dim']
        head_dim = model_cfg['head_dim']
        head_kernel_size = model_cfg['head_kernel_size']
        head_with_ln = model_cfg['head_with_ln']
        head_num_layers = model_cfg['head_num_layers']
        train_cfg = cfg1['train_cfg']
        
        self.fusion_module = make_neck(
            'fusion_fpn', 
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim, # reduce the dimension
                'start_level': 0,
                'end_level': backbone_arch[-1] + 1,
            })
        
        # create the classification and regression heads
        # get param from cfg1
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, model_cfg['num_classes'],
            kernel_size=head_kernel_size,
            prior_prob=train_cfg['cls_prior_prob'],
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(model_cfg['regression_range']),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

    def forward(self, video_list, heatmap_list): # TODO: CHECK
        # get the FPN features from the visual tower
        visual_fpn_feats, visual_fpn_masks, visual_points = self.get_fpn_feat(self.visual_tower, video_list)
        # get the FPN features from the heatmap tower
        heatmap_fpn_feats, heatmap_fpn_masks, heatmap_points = self.get_fpn_feat(self.heatmap_tower, heatmap_list)
        # fuse the features from the two towers
        ## 1. concatenate the features from the two towers
        fused_feats = []
        for visual_feat, heatmap_feat in zip(visual_fpn_feats, heatmap_fpn_feats):
            fused_feat = torch.cat([visual_feat, heatmap_feat], dim=1)
            fused_feats.append(fused_feat)
        fused_masks = []
        for visual_mask, heatmap_mask in zip(visual_fpn_masks, heatmap_fpn_masks):
            fused_mask = torch.cat([visual_mask, heatmap_mask], dim=1)
            fused_masks.append(fused_mask)
        ## 2. apply the fusion module to fuse the features
        fused_feats, fpn_masks = self.fusion_module(fused_feats, visual_fpn_masks)
        # to reg/cls head
        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fused_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fused_feats, fpn_masks)
        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            # print(video_list)
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.visual_tower.label_points(
                visual_points, gt_segments, gt_labels)

            # compute the loss and return
            losses = self.visual_tower.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            if self.include_ori_loss:
                v_losses = self.visual_tower(video_list)
                h_losses = self.heatmap_tower(heatmap_list)
                losses['final_loss'] += v_losses['final_loss'] + h_losses['final_loss']
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.visual_tower.inference(
                video_list, visual_points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results 
    
    def get_fpn_feat(self, module, input_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = module.preprocessing(input_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = module.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = module.neck(feats, masks)
        # fpn_feats [16, 256, 768] ..[16, 256, 384]..[16, 256, 24]

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = module.point_generator(fpn_feats)
        return fpn_feats, fpn_masks, points
    
    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]