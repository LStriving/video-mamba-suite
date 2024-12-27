import math
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from fairscale.nn.checkpoint import checkpoint_wrapper

from .models import register_video_stem, register_image_stem
from .blocks import MaskMambaBlock, MaskedPoolingMHCA, MaskedPoolingMHCAv2, MaskMultiScaleMambaBlock
from pytorch_i3d import InceptionI3d

@register_image_stem('resnet50')
class ResNet50(nn.Module):
    '''
        ResNet50 image stem
    '''
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        self.module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.module.fc = nn.Identity()
    
    def forward(self, x):
        channel_dim = x.shape[1]
        if channel_dim == 1:
            x = x.repeat(1,3,1,1)
        return self.module(x) # output dim: 2048
    
@register_image_stem('resnet50_projector')
class ResNet50Projector(nn.Module):
    '''
        ResNet50 image stem with projection head
    '''
    def __init__(
        self, 
        out_dim,
        **kwargs
    ):
        super().__init__()
        self.module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.module.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        channel_dim = x.shape[1]
        if channel_dim == 1:
            x = x.repeat(1,3,1,1)
        return self.module(x)

@register_image_stem('projector')
class Projector(nn.Module):
    '''
        Projector image stem
    '''
    def __init__(
        self, 
        in_dim,
        out_dim,
        **kwargs
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.proj(x)

@register_image_stem('s-mvit')
class SpatilMViT(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            in_channels: int,
            embed_dim: int,
            num_heads: int,
            attn_dropout: float,
            num_layers: int,
            pool_size: int,
            pool_mode: str,
            act_checkpoint: bool,
            *args, 
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        

        # init
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        self.conv_proj = self.wrapper(self.conv_proj)

        self.n_layers = num_layers
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCA(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )


    def _process_image(self, x):
        n, c, h, w = x.shape
        torch._assert(h == w == self.image_size, f"Input image size must be {self.image_size}x{self.image_size}")
        torch._assert(h % self.patch_size == 0, f"Image size must be divisible by patch size {self.patch_size}")
        n_w = w // self.patch_size
        n_h = h // self.patch_size

        
        x = self.conv_proj(x)
        x = x.reshape(n, self.embed_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        
        return x


    def forward(self, x):
        # x: (B, C, H, W)
        x = self._process_image(x)
        # x: (B, N, C) where N is the number of patches

        x = x.transpose(1, 2)
        mask = torch.ones((x.shape[0], 1, x.shape[-1]), device=x.device).bool()
        for i in range(self.n_layers):
            x, mask = self.blocks[i](x, mask)

        # desired output shape: (B, C)
        x = x.squeeze(-1)
        return x

@register_image_stem('s-mvitV2')
class SpatilMViT_v2(SpatilMViT):
    def __init__(self, 
                 image_size: int, 
                 patch_size: int, 
                 in_channels: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 attn_dropout: float, 
                 num_layers: int, 
                 pool_size: int, 
                 pool_mode: str, 
                 act_checkpoint: bool, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(image_size, 
                         patch_size, 
                         in_channels, 
                         embed_dim, 
                         num_heads, 
                         attn_dropout, 
                         num_layers, 
                         pool_size, 
                         pool_mode, 
                         act_checkpoint, 
                         *args, 
                         **kwargs)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCAv2(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )
    

@register_image_stem('MVMamba')
class SpatilVMamba(SpatilMViT):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        gamma: float = 0.0, 
        mamba_type: str = 'mdbm',
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMultiScaleMambaBlock(
                    n_embd=self.embed_dim,
                    drop_path_rate=drop_path_rate,
                    n_ds_stride=self.pool_size,
                    pool_method=self.pool_mode,
                    gamma=gamma,
                    use_mamba_type=mamba_type.lower()
                ))
            )

@register_image_stem('postpoolVMamba')
class PostPoolVMamba(SpatilMViT):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMambaBlock(
                    n_embd=self.embed_dim,
                    n_ds_stride=self.pool_size,
                    pool_method=self.pool_mode,
                    drop_path_rate=drop_path_rate,
                ))
            )

@register_video_stem('i3d')
class Inception3D(nn.Module):
    '''
        I3D video stem
    '''
    def __init__(
        self,
        in_channels: int,
        dropout_prob: float,
        pretrained: str,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__()
        self.i3d = InceptionI3d(num_classes=7, in_channels=in_channels, dropout_keep_prob=dropout_prob, **kwargs)
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.i3d = self.wrapper(self.i3d)
        if pretrained:
            self._init_from_pretrained(pretrained)

    def _init_from_pretrained(self, pretrained):
        os.path.exists(pretrained), f"Pretrained model {pretrained} does not exist."
        ckpt = torch.load(pretrained)['state_dict']
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            new_ckpt[k.replace('module.', '')] = v
        self.i3d.load_state_dict(new_ckpt)
        print(f"I3D: Load pretrained model from {pretrained}")

    def forward(self, x):
        return self.i3d.extract_features(x)[:,:,0,0,0]

@register_image_stem('posec3d')
class PoseC3D(nn.Module):
    '''
        PoseC3D image stem
    '''
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        pass

@register_video_stem('tcn')
class TCN(nn.Module):
    '''
        TCN video stem
    '''
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        pass

@register_video_stem('mvit')
class MViT(nn.Module):
    '''
        MViT video stem
        Refer to  https://arxiv.org/pdf/2104.11227
    '''
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        num_layers: int,
        pool_size: int,
        pool_mode: str,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__()
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.embed_dim = embed_dim
        self.n_layers = num_layers
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCA(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )

    def forward(self, x, mask=None):
        # input shape: (B, C, T) desired output shape: (B, C) 
        ## T is the frame length within a segment/window
        if mask is None:
            b, c, t = x.shape
            mask = torch.ones((b,1,t), device=x.device)

        for i in range(self.n_layers):
            x, mask = self.blocks[i](x, mask)
            
        return x, mask

@register_video_stem('MMamba')
class MMamba(MViT):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        gamma: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMultiScaleMambaBlock(
                    n_embd=self.embed_dim,
                    drop_path_rate=drop_path_rate,
                    n_ds_stride=self.pool_size,
                    pool_method=self.pool_mode,
                    gamma=gamma
                ))
            )

@register_video_stem('postpoolMamba')
class PostPoolMamba(MViT):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMambaBlock(
                    n_embd=self.embed_dim,
                    n_ds_stride=self.pool_size,
                    drop_path_rate=drop_path_rate,
                    pool_method=self.pool_mode
                ))
            )

@register_video_stem('mvitV2')
class MViT_v2(MViT):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        num_layers: int,
        pool_size: int,
        pool_mode: str,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__(
            embed_dim, 
            num_heads, 
            attn_dropout, 
            num_layers, 
            pool_size, 
            pool_mode, 
            act_checkpoint, 
            **kwargs
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCAv2(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )