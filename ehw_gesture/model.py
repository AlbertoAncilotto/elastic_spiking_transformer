import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, List
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate, functional
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import numpy as np

try:
    import cupy as cp
    x = cp.arange(10)
    y = x * 2
    cp.cuda.runtime.deviceSynchronize()
    BACKEND = 'cupy'
except Exception:
    BACKEND = 'torch'
    

__all__ = ['spikformer']

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        x = self.fc2_lif(x)
        return x
    
class XiMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., num_granularities=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.max_hidden = hidden_features
        self.granularities = self._generate_granularities(hidden_features, num_granularities)
        self.num_granularities = num_granularities
        
        self.fc1_conv = nn.Conv1d(in_features, self.max_hidden, kernel_size=1, stride=1)
        self.fc2_conv = nn.Conv1d(self.max_hidden, out_features, kernel_size=1, stride=1)
        
        self.fc1_bns = nn.ModuleList([
            nn.BatchNorm1d(g) for g in self.granularities
        ])
        
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        
        self.c_output = out_features
    
    def _generate_granularities(self, max_hidden, num_granularities):
        """Generate log-spaced granularities ending at max_hidden."""
        if num_granularities == 1:
            return [max_hidden]
        
        min_hidden = max(32, max_hidden // (2 ** (num_granularities)))
        granularities = np.logspace(
            np.log2(min_hidden), 
            np.log2(max_hidden), 
            num=num_granularities, 
            base=2.0
        )
        
        granularities = [int(np.round(g / 32) * 32) for g in granularities]
        granularities[-1] = max_hidden
        granularities = sorted(list(set(granularities)))
        return granularities
    
    def forward(self, x, granularity=None):
        T, B, C, N = x.shape
        
        if granularity is None:
            granularity_idx = np.random.randint(0, self.num_granularities)
        else:
            granularity_idx = granularity
        
        current_hidden = self.granularities[granularity_idx]
        x_flat = x.flatten(0, 1)  # (T*B, C, N)
        x = self.fc1_conv(x_flat)  # (T*B, max_hidden, N)
        x = x[:, :current_hidden, :] 
        x = self.fc1_bns[granularity_idx](x)  # (T*B, current_hidden, N)
        x = x.reshape(T, B, current_hidden, N).contiguous()

        x = self.fc1_lif(x)  # (T, B, current_hidden, N)
        
        x_flat = x.flatten(0, 1)  # (T*B, current_hidden, N)
        
        weight_sliced = self.fc2_conv.weight[:, :current_hidden, :]  # (out_features, current_hidden, 1)
        bias = self.fc2_conv.bias
        
        x = torch.nn.functional.conv1d(x_flat, weight_sliced, bias, stride=1)  # (T*B, out_features, N)
        x = self.fc2_bn(x)  # (T*B, out_features, N)
        x = x.reshape(T, B, self.c_output, N).contiguous()
        x = self.fc2_lif(x)
        
        return x
    
    def get_granularity_info(self):
        params = self.get_granularity_parameters()
        return {
            'granularities': self.granularities,
            'num_granularities': self.num_granularities,
            'max_hidden': self.max_hidden,
            'params': params
        }
    
    def get_granularity_parameters(self):
        params = []
        in_f = self.fc1_conv.in_channels
        out_f = self.fc2_conv.out_channels

        for h in self.granularities:
            total = (
                in_f * h * 1 + h +          # fc1_conv (kernel_size=1)
                2 * h +                     # fc1_bn
                out_f * h * 1 + out_f +     # fc2_conv (kernel_size=1)
                2 * out_f                   # fc2_bn
            )
            params.append(total)

        return params

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)

        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=BACKEND)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,N))

        return x

class XiSSA(nn.Module):
    def __init__(self, dim, num_heads=8, num_granularities=4, qkv_bias=False, 
                 qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, lower_heads_limit=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.max_num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_granularities = num_granularities
        self.scale = qk_scale if qk_scale is not None else 0.25
        self.lower_heads_limit = lower_heads_limit
        self.head_granularities = self._generate_head_granularities(num_heads, num_granularities)
        
        # Q, K, V convolutions
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        
        # Multiple BatchNorms for each granularity
        self.q_bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(num_granularities)])
        self.k_bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(num_granularities)])
        self.v_bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(num_granularities)])
        
        # LIF nodes
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=BACKEND)
        
        # Projection layers
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        
    def _generate_head_granularities(self, max_heads, num_granularities):
        if num_granularities == 1:
            return [max_heads]
        min_heads = max(self.lower_heads_limit, max_heads // (2 ** (num_granularities - 1)))
        granularities = np.logspace(
            np.log2(min_heads), 
            np.log2(max_heads), 
            num_granularities, 
            dtype=int, 
            base=2
        )
        granularities[-1] = max_heads
        return list(granularities)
    
    def get_granularity_info(self):
        params = self.get_granularity_parameters()
        return {
            'head_granularities': self.head_granularities,
            'num_granularities': self.num_granularities,
            'max_num_heads': self.max_num_heads,
            'params': params
        }
    
    def get_granularity_parameters(self):
        params = []
        dim = self.dim
        hd = self.head_dim

        for h in self.head_granularities:
            qkv_lin = 3 * (dim * h + dim)
            qkv_bn = 3 * (2 * h)
            proj = h * dim + dim
            proj_bn = 2 * dim
            attn = 3 * (dim * h * hd)
            total = qkv_lin + qkv_bn + attn + proj + proj_bn
            params.append(total)

        return params
    
    def forward(self, x, granularity=None):
        T, B, C, N = x.shape
        
        # Select granularity
        if granularity is None:
            granularity_idx = np.random.randint(0, self.num_granularities)
        else:
            granularity_idx = granularity
        
        current_num_heads = self.head_granularities[granularity_idx]
        self.current_num_heads = current_num_heads
        x_flat = x.flatten(0, 1)  # (T*B, C, N)
        
        # Q branch
        q = self.q_conv(x_flat)  # (T*B, C, N)
        q = self.q_bns[granularity_idx](q).reshape(T, B, C, N).contiguous()
        q = self.q_lif(q)  # (T, B, C, N)
        # Transpose to (T, B, N, C) for head reshaping
        q = q.transpose(-1, -2)  # (T, B, N, C)
        q = q.reshape(T, B, N, self.max_num_heads, self.head_dim)[:, :, :, :current_num_heads, :]
        q = q.permute(0, 1, 3, 2, 4).contiguous()  # (T, B, current_num_heads, N, head_dim)
        
        # K branch
        k = self.k_conv(x_flat)  # (T*B, C, N)
        k = self.k_bns[granularity_idx](k).reshape(T, B, C, N).contiguous()
        k = self.k_lif(k)  # (T, B, C, N)
        k = k.transpose(-1, -2)  # (T, B, N, C)
        k = k.reshape(T, B, N, self.max_num_heads, self.head_dim)[:, :, :, :current_num_heads, :]
        k = k.permute(0, 1, 3, 2, 4).contiguous()  # (T, B, current_num_heads, N, head_dim)
        
        # V branch
        v = self.v_conv(x_flat)  # (T*B, C, N)
        v = self.v_bns[granularity_idx](v).reshape(T, B, C, N).contiguous()
        v = self.v_lif(v)  # (T, B, C, N)
        v = v.transpose(-1, -2)  # (T, B, N, C)
        v = v.reshape(T, B, N, self.max_num_heads, self.head_dim)[:, :, :, :current_num_heads, :]
        v = v.permute(0, 1, 3, 2, 4).contiguous()  # (T, B, current_num_heads, N, head_dim)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1))  # (T, B, current_num_heads, N, N)
        x = (attn @ v) * self.scale  # (T, B, current_num_heads, N, head_dim)
        
        # Zero padding if using fewer heads than max
        if current_num_heads < self.max_num_heads:
            pad = torch.zeros(T, B, self.max_num_heads - current_num_heads, N, self.head_dim, 
                            device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)  # Concatenate along head dimension
        
        # Reshape back: (T, B, max_num_heads, N, head_dim) -> (T, B, N, max_num_heads, head_dim)
        x = x.transpose(2, 3).contiguous()
        # Merge heads: (T, B, N, C)
        x = x.reshape(T, B, N, C)
        # Transpose back to channel-first: (T, B, C, N)
        x = x.transpose(-1, -2).contiguous()
        
        x = self.attn_lif(x)  # (T, B, C, N)
        x = x.flatten(0, 1)  # (T*B, C, N)
        x = self.proj_conv(x)  # (T*B, C, N)
        x = self.proj_bn(x).reshape(T, B, C, N).contiguous()
        x = self.proj_lif(x)
        
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, num_granularities=4, attn_lower_heads_limit=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XiSSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, num_granularities=num_granularities, lower_heads_limit=attn_lower_heads_limit)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = XiMLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, num_granularities=num_granularities)
    
    def forward(self, x, granularity=None):
        x = x + self.attn(x, granularity=granularity[0] if isinstance(granularity, list) else granularity)
        x = x + self.mlp(x, granularity=granularity[1] if isinstance(granularity, list) else granularity)
        return x

class XiConv(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple] = 3,
        stride: Union[int, Tuple] = 1,
        padding: Optional[Union[int, Tuple]] = None,
        groups: Optional[int] = 1,
        act: Optional[bool] = False,
        gamma: Optional[float] = 2,
        pool: Optional[bool] = None,
        batchnorm: Optional[bool] = False,

    ):
        super().__init__()
        self.compression = int(gamma)
        self.attention_lite_ch_in = c_out // self.compression // 2
        self.pool = pool
        self.batchnorm = batchnorm

        if self.compression > 1:
            self.compression_conv = nn.Conv2d(
                c_in, c_out // self.compression, 1, 1, groups=groups, bias=False
            )

        self.main_conv = nn.Conv2d(
            c_out // self.compression if self.compression > 1 else c_in,
            c_out,
            kernel_size,
            stride,
            groups=groups,
            padding=kernel_size // 2 if padding is None else padding,
            bias=False,
        )
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

        if pool:
            self.mp = nn.MaxPool2d(pool)

        if batchnorm:
            self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor):

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)

        if self.pool:
            x = self.mp(x)

        # main conv and activation
        x = self.main_conv(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)

        return x

class XiConvMultiGran(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple] = 3,
        stride: Union[int, Tuple] = 1,
        padding: Optional[Union[int, Tuple]] = None,
        groups: Optional[int] = 1,
        act: Optional[bool] = False,
        gamma: Optional[float] = 2,
        pool: Optional[bool] = None,
        batchnorm: Optional[bool] = False,
        num_granularities: int = 4,
        lower_filter_limit: int = 4,
    ):
        super().__init__()
        self.compression = int(gamma)
        self.pool = pool
        self.batchnorm = batchnorm
        self.num_granularities = num_granularities
        self.c_out = c_out
        self.c_in = c_in
        self.lower_filter_limit = lower_filter_limit
        
        self.c_compr_granularities = self._generate_granularities(c_out // self.compression, num_granularities)
        
        self.c_compressed = c_out // self.compression if self.compression > 1 else c_in
        if self.compression > 1:
            self.compression_conv = nn.Conv2d(
                c_in, c_out // self.compression, 1, 1, groups=groups, bias=False
            )

        
        self.main_conv = nn.Conv2d(
            self.c_compressed,
            c_out,
            kernel_size,
            stride,
            groups=groups,
            padding=kernel_size // 2 if padding is None else padding,
            bias=False,
        )
        
        if batchnorm:
            self.main_bn = nn.BatchNorm2d(c_out)

        if pool:
            self.mp = nn.MaxPool2d(pool)

    def _generate_granularities(self, max_channels, num_granularities):
        """Generate log-spaced channel granularities ending at c_compressed."""
        if num_granularities == 1:
            return [max_channels]
        
        min_channels = max(self.lower_filter_limit, max_channels // (2 ** (num_granularities-1)))
        granularities = np.logspace(
            np.log2(min_channels), 
            np.log2(max_channels), 
            num=num_granularities, 
            base=2.0
        )
        
        granularities = [int(np.round(g / 4) * 4) for g in granularities]
        granularities[-1] = max_channels
        granularities = sorted(list(set(granularities)))
        return granularities

    def forward(self, x: torch.Tensor, granularity=None):
        if granularity is None:
            granularity_idx = np.random.randint(0, self.num_granularities)
        else:
            granularity_idx = granularity
        
        current_c_compressed = self.c_compr_granularities[granularity_idx]
        
        # Compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)

        x = x[:, :current_c_compressed, :, :]  # Slice channels

        if self.pool:
            x = self.mp(x)

        # Main convolution
        weight_sliced = self.main_conv.weight[:, :current_c_compressed, :, :] 
        bias = self.main_conv.bias
        x = torch.nn.functional.conv2d(x, weight_sliced, bias, stride=self.main_conv.stride,
                                       padding=self.main_conv.padding, groups=self.main_conv.groups) 
        
        if self.batchnorm:
            x = self.main_bn(x)

        return x
    
    def get_granularity_info(self):
        params = self.get_granularity_parameters()
        return {
            'c_compr_granularities': self.c_compr_granularities,
            'num_granularities': self.num_granularities,
            'params': params
        }
    
    def get_granularity_parameters(self):
        params = []
        c_in = self.c_in
        for c_compr_g in self.c_compr_granularities:
            total = 0
            if self.compression > 1:
                total += c_in * (c_compr_g) * 1  # compression conv
                if self.batchnorm:
                    total += 2 * (c_compr_g)  # compression bn
            total += c_compr_g * self.c_out * self.main_conv.kernel_size[0] * self.main_conv.kernel_size[1]  # main conv
            if self.batchnorm:
                total += 2 * (self.c_out)  # main bn
            params.append(total)
        return params

    
class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, alpha=1.0):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        c1 = int((embed_dims // 8) * alpha)
        c2 = int((embed_dims // 4) * alpha)
        c3 = int((embed_dims // 2) * alpha)
        print(f"SPS channels: {c1}, {c2}, {c3}, embed_dims: {embed_dims}")
        
        self.proj_conv = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c1)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(c2)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(c3)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(c3, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
        x = self.proj_bn(x).reshape(T,B,-1,H,W).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//8, W//8).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T,B,-1,H//16,W//16).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0,1)
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H//16)*(H//16)).contiguous()

        return x

class XiSPSv2(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, alpha=1.0, num_granularities=1, lower_filter_limit=4):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.multi_granularity = num_granularities>1

        c1 = int((embed_dims // 8) * alpha)
        c2 = int((embed_dims // 4) * alpha)
        c3 = int((embed_dims // 2) * alpha)
        
        # self.proj_conv = nn.Xi(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_conv = XiConv(in_channels, c1, kernel_size=3, stride=1, padding=1)
        self.proj_bn = nn.BatchNorm2d(c1)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = XiConv(c1, c2, kernel_size=3, stride=1, padding=1)
        self.proj_bn1 = nn.BatchNorm2d(c2)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = XiConv(c2, c3, kernel_size=3, stride=1, padding=1)#, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(c3)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        if num_granularities>1:
            self.proj_conv3 = XiConvMultiGran(c3, embed_dims, kernel_size=3, stride=1, padding=1, num_granularities=num_granularities, lower_filter_limit=lower_filter_limit)
        else:
            self.proj_conv3 = XiConv(c3, embed_dims, kernel_size=3, stride=1, padding=1)#, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        if num_granularities>1:
            self.rpe_conv = XiConvMultiGran(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, num_granularities=num_granularities, lower_filter_limit=lower_filter_limit)
        else:
            self.rpe_conv = XiConv(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1)#, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)

    def forward(self, x, granularity=None):
        T, B, C, H, W = x.shape
        if granularity is None:
            granularity_idx = np.random.randint(0, 4)
        else:
            granularity_idx = granularity 
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
        x = self.proj_bn(x).reshape(T,B,-1,H,W).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)
        
        if self.multi_granularity:
            x = self.proj_conv3(x, granularity_idx)
        else:
            x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//8, W//8).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        if self.multi_granularity:
            x_rpe = self.rpe_bn(self.rpe_conv(x, granularity_idx)).reshape(T,B,-1,H//16,W//16).contiguous()
        else:
            x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T,B,-1,H//16,W//16).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0,1)
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H//16)*(H//16)).contiguous()

        return x
    
class XiSPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, alpha=1.0, num_granularities=1, lower_filter_limit=4):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.multi_granularity = num_granularities>1

        c1 = int((embed_dims // 8) * alpha)
        c2 = int((embed_dims // 4) * alpha)
        c3 = int((embed_dims // 2) * alpha)
        
        self.proj_conv = XiConv(in_channels, c1, kernel_size=3, batchnorm=True, pool=2, act=False, gamma=1)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        
        self.proj_conv1 = XiConv(c1, c2, kernel_size=3, batchnorm=True, pool=2, act=False, gamma=1)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        
        self.proj_conv2 = XiConv(c2, c3, kernel_size=3, batchnorm=True, pool=2, act=False)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)
        
        if num_granularities>1:
            self.proj_conv3 = XiConvMultiGran(c3, embed_dims, kernel_size=3, batchnorm=True, gamma=1, pool=2, act=False, num_granularities=num_granularities, lower_filter_limit=lower_filter_limit)
        else:
            self.proj_conv3 = XiConv(c3, embed_dims, kernel_size=3, batchnorm=True, pool=2)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=BACKEND)


    def forward(self, x, granularity=None):
        T, B, C, H, W = x.shape
        if granularity is None:
            granularity_idx = np.random.randint(0, 4)
        else:
            granularity_idx = granularity 

        x = self.proj_conv(x.flatten(0,1)).reshape(T,B,-1,H//2,W//2).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()

        x = self.proj_conv1(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x).reshape(T, B, -1, H//8, W//8).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()

        if self.multi_granularity:
            x = self.proj_conv3(x, granularity_idx)
        else:
            x = self.proj_conv3(x)
        x = x.reshape(T, B, -1, H//16, W//16).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()

        x = x.reshape(T, B, -1, (H//16)*(W//16)).contiguous()

        return x


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 16, num_granularities=4, sps_alpha=1.0,
                 use_xisps = False, xisps_elastic = False, attn_lower_heads_limit=2, sps_lower_filter_limit=4,
                 **kwargs):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.num_granularities = num_granularities
        self.xisps_elastic = xisps_elastic
        self.use_xisps = use_xisps

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        if not use_xisps:
            patch_embed = SPS(img_size_h=img_size_h,
                                    img_size_w=img_size_w,
                                    patch_size=patch_size,
                                    in_channels=in_channels,
                                    embed_dims=embed_dims,
                                    alpha=sps_alpha)
        else:
            if sps_alpha <= 1.0:
                # print("AAAAAAAAAAAAAAAAAA WRONG SPSSSS VALUE AAAAAAAAAAAAAAA")
                print("Using XiSPS v1: alpha = ", sps_alpha)
                patch_embed = XiSPS(img_size_h=img_size_h,
                                        img_size_w=img_size_w,
                                        patch_size=patch_size,
                                        in_channels=in_channels,
                                        embed_dims=embed_dims,
                                        alpha=sps_alpha,
                                        num_granularities=num_granularities if xisps_elastic else 1,
                                        lower_filter_limit=sps_lower_filter_limit)
            else:
                print("Using XiSPS v2: alpha = ", sps_alpha)
                patch_embed = XiSPSv2(img_size_h=img_size_h,
                                        img_size_w=img_size_w,
                                        patch_size=patch_size,
                                        in_channels=in_channels,
                                        embed_dims=embed_dims,
                                        alpha=sps_alpha,
                                        num_granularities=num_granularities if xisps_elastic else 1,
                                        lower_filter_limit=sps_lower_filter_limit)

        
        num_patches = patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, num_granularities=num_granularities, attn_lower_heads_limit=attn_lower_heads_limit)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        self.norm = norm_layer(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)
        self.model_granularity_info = self.get_granularity_info()

    def get_granularity_info(self):
        fixed_params = {}
        variable_info = {}
        
        # Collect patch_embed info
        for name, module in self.patch_embed.named_modules():
            key = f'patch_embed_{name}'
            if isinstance(module, XiConvMultiGran):
                variable_info[key] = module.get_granularity_info()
            elif isinstance(module, XiConv):
                fixed_params[key] = sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Conv2d) and not self.use_xisps:
                fixed_params[key] = sum(p.numel() for p in module.parameters())
        
        # Collect block info
        for b_id, b in enumerate(self.block):
            variable_info[f'block_{b_id}_mlp'] = b.mlp.get_granularity_info()
            variable_info[f'block_{b_id}_attn'] = b.attn.get_granularity_info()
        
        # Collect head info
        fixed_params['head'] = sum(p.numel() for p in self.head.parameters())
        
        # Compute totals
        total_fixed = sum(fixed_params.values())
        var_min = sum(info['params'][0] for info in variable_info.values())
        var_max = sum(info['params'][-1] for info in variable_info.values())
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FIXED PARAMETERS: {total_fixed:,}")
        print(f"{'='*60}")
        for name, count in fixed_params.items():
            pct = 100 * count / total_fixed
            print(f"  {name:40s} {count:12,} ({pct:5.1f}%)")
        
        print(f"\n{'='*60}")
        print(f"VARIABLE PARAMETERS: [{var_min:,}, {var_max:,}]")
        print(f"{'='*60}")
        for name, info in variable_info.items():
            print(f"  {name:40s} [{info['params'][0]:,}, {info['params'][-1]:,}]")
        
        print(f"\n{'='*60}")
        print(f"TOTAL MODEL SIZE:")
        print(f"  Minimum: {total_fixed + var_min:,}")
        print(f"  Maximum: {total_fixed + var_max:,}")
        print(f"{'='*60}\n")
        
        return {
            'fixed': fixed_params,
            'variable': variable_info,
            'totals': {
                'fixed': total_fixed,
                'variable_min': var_min,
                'variable_max': var_max,
                'model_min': total_fixed + var_min,
                'model_max': total_fixed + var_max
            }
        }

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # * Acc@1 (G:0) = 5.870194295163291, Acc@1 (G:3) = 5.00206696982224, loss = 3.0941646684299817 with XiSPS
    #  * Acc@1 (G:0) = 12.195121951219512, Acc@1 (G:3) = 9.880115750310045, loss = 2.9920475378509397 with SPS
    def forward_features(self, x, granularity=None): 
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        if self.xisps_elastic:
            x = patch_embed(x, granularity=granularity[0] if isinstance(granularity, list) else granularity)
        else:
            x = patch_embed(x)#, granularity=granularity[0] if isinstance(granularity, list) else granularity)
        for blk in block:
            x = blk(x, granularity=granularity[1:] if isinstance(granularity, list) else granularity)
        return x.mean(3)
    
    def sample_granularity(self, granularity):
        if granularity is None:
            # Sample with bias towards larger granularities
            # Probabilities: [0.1, 0.2, 0.3, 0.4] for num_granularities=4
            probs = np.arange(1, self.num_granularities + 1)
            probs = probs / probs.sum()
            return np.random.choice(self.num_granularities, p=probs)
        return granularity

    def get_granularity_parameters(self, granularity):
        """
        Get the number of parameters for a given granularity configuration.
        
        Args:
            granularity: Either a single int (0-3) applied to all layers,
                        or a list [feat_extractor_gran, attn_gran, mlp_gran]
                        
        Returns:
            Total number of parameters at the given granularity
        """
        # Parse granularity input
        if isinstance(granularity, (list, tuple)):
            feat_gran = granularity[0]
            attn_gran = granularity[1] if len(granularity) > 1 else granularity[0]
            mlp_gran = granularity[2] if len(granularity) > 2 else granularity[0]
        else:
            feat_gran = attn_gran = mlp_gran = granularity
        
        # Get fixed parameters from cached info
        total_fixed = self.model_granularity_info['totals']['fixed']
        
        # Calculate variable parameters based on granularity
        variable_params = 0
        
        # Patch embed (feature extractor) - uses feat_gran
        for name, info in self.model_granularity_info['variable'].items():
            if 'patch_embed' in name:
                variable_params += info['params'][feat_gran]
            elif 'attn' in name:
                variable_params += info['params'][attn_gran]
            elif 'mlp' in name:
                variable_params += info['params'][mlp_gran]
        
        return total_fixed + variable_params

    def forward(self, x, granularity=None):

        granularity = self.sample_granularity(granularity)

        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x, granularity=granularity)
        x = self.head(x.mean(0))
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        # patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        # in_channels=2, num_classes=10, qkv_bias=False,depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    import torchinfo
    model = spikformer(sps_alpha=0.5, num_granularities=4,
            patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
            in_channels=2, num_classes=10, qkv_bias=False, depths=2, sr_ratios=1,
            use_xisps = True, xisps_elastic = True,)
    
    x = torch.randn(16, 2, 2, 128, 128)  # T, B, C, H, W
    out = model(x)
    functional.reset_net(model)

    out = model(x, granularity=2)
    functional.reset_net(model)

    out = model(x, granularity=[3, 2, 1])
    functional.reset_net(model)
        
    print("\n\n ================== Model Summary: ================== \n\n")
    torchinfo.summary(model)

    print("\n\n ================== Patch Embedding Summary: ================== \n\n")
    torchinfo.summary(model.patch_embed)

    print("\n\n ================== Single Block Summary: ================== \n\n")
    torchinfo.summary(model.block)

    print("\n\n ================== CLS Summary: ================== \n\n")
    torchinfo.summary(model.head)

    model.get_granularity_info()


