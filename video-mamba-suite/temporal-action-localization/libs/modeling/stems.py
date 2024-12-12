import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from .models import register_video_stem, register_image_stem
from .blocks import MaskedPoolingMHCA
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

@register_video_stem('i3d')
class Inception3D(nn.Module):
    '''
        I3D video stem
    '''
    def __init__(
        self,
        in_channels: int,
        dropout_prob: float,
        **kwargs
    ):
        super().__init__()
        self.i3d = InceptionI3d(in_channels=in_channels, dropout_keep_prob=dropout_prob, **kwargs)

    def forward(self, x):
        return self.i3d.extract_features(x)

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
        **kwargs
    ):
        super().__init__()
        self.n_layers = num_layers
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                MaskedPoolingMHCA(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                )
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