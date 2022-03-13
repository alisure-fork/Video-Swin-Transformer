import torch
from torch import nn
from operator import mul
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import reduce, lru_cache
from timm.models.layers import DropPath, trunc_normal_

from mmcv.runner import load_checkpoint
from mmaction.utils import get_root_logger
from ..builder import BACKBONES


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    pass


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    pass


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    pass


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    pass


######################################################################


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    pass


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        pass

    def forward(self, x, mae_unmask=None):
        """ (num_windows*B, N, C)"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 相对位置
        if mae_unmask is None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N
        else:
            one_bias_list = []
            for mask in mae_unmask:
                position_bias = self.relative_position_index[mask][:, mask]
                one_bias = self.relative_position_bias_table[position_bias.reshape(-1)].reshape(N, N, -1)
                one_bias_list.append(one_bias)
                pass
            relative_position_bias = torch.cat([one.unsqueeze(0) for one in one_bias_list])
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
            pass

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    pass


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        pass

    def forward(self, x, mae_unmask=None):
        """ (B, D, H, W, C)"""
        shortcut = x

        device = x.device
        B, D, H, W, C = x.shape
        window_size = get_window_size((D, H, W), self.window_size)

        # 原始的
        if mae_unmask is None:
            x = self.norm1(x)

            # partition windows, 每个窗口中的数据
            x_windows = window_partition(x, window_size)  # B*nW, Wd*Wh*Ww, C  # [512, 392, 96]
            attn_windows = self.attn(x_windows)  # B*nW, Wd*Wh*Ww, C

            attn_windows = attn_windows.view(-1, *(window_size + (C,)))
            x = window_reverse(attn_windows, window_size, B, D, H, W)  # B D' H' W' C

            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            ###############################################################################################################
            # partition windows, 每个窗口中的数据
            x_windows = window_partition(x, window_size)  # B*nW, Wd*Wh*Ww, C  # [512, 392, 96]

            batch_range = torch.arange(B, device=device)[:, None]
            unmask_x_windows = x_windows[batch_range, mae_unmask]
            unmask_x_windows = self.norm1(unmask_x_windows)

            # W-MSA/SW-MSA
            unmask_attn_windows = self.attn(unmask_x_windows, mae_unmask=mae_unmask)  # B*nW, Wd*Wh*Ww, C

            # res
            unmask_shortcut = window_partition(shortcut, window_size)[batch_range, mae_unmask]
            unmask_windows = self.drop_path(unmask_attn_windows) + unmask_shortcut
            unmask_windows = unmask_windows + self.drop_path(self.mlp(self.norm2(unmask_windows)))

            attn_windows = torch.zeros_like(x_windows)
            attn_windows[batch_range, mae_unmask] = unmask_windows
            attn_windows = attn_windows.view(-1, *(window_size + (C,)))
            x = window_reverse(attn_windows, window_size, B, D, H, W)  # B D' H' W' C
            ###############################################################################################################

        return x

    pass


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, depth, num_heads, window_size=(1,7,7), mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer, use_checkpoint=use_checkpoint) for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
            pass
        pass

    def forward(self, x, mae_unmask=None):
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        for blk in self.blocks:
            x = blk(x, mae_unmask=mae_unmask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

    pass


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class Decoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, num_patches, decoder_depth,
                 decoder_heads, decoder_dim_head, pixel_values_per_patch):
        super().__init__()
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads,
                                   dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        pass

    def forward(self, x, unmasked_indices, masked_indices, masked_patches):
        B = x.size(0)
        unmask_tokens = self.enc_to_dec(x)
        unmask_tokens = unmask_tokens + self.decoder_pos_emb(unmasked_indices)  # 未掩掉的块

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=B, n=masked_indices.size(1))
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)  # 所有掩掉的块
        all_decoder_tokens = torch.cat([unmask_tokens, mask_tokens], 1)

        decoded_tokens = self.decoder(all_decoder_tokens)
        pred_pixel_values = self.to_pixels(decoded_tokens)
        pred_masked_pixel_values = pred_pixel_values[:, -masked_indices.size(1):]
        ############################################################################################################

        # calculate reconstruction loss
        # recon_loss = F.mse_loss(pred_masked_pixel_values, masked_patches)
        return pred_masked_pixel_values, masked_patches

    pass


class Projector(nn.Module):

    def __init__(self, in_dim, pro_dim, num_patches, depth, heads, dim_head):
        super().__init__()
        self.enc_to_dec = nn.Linear(in_dim, pro_dim) if in_dim != pro_dim else nn.Identity()
        self.pos_emb = nn.Embedding(num_patches + 1, pro_dim)
        self.projector_token = nn.Parameter(torch.randn(1, 1, pro_dim))
        self.projector = Transformer(dim=pro_dim, depth=depth, heads=heads,
                                   dim_head=dim_head, mlp_dim=pro_dim * 4)
        pass

    def forward(self, x, unmasked_indices, masked_indices=None):
        x = self.enc_to_dec(x)
        b, n, d = x.shape

        projector_token = repeat(self.projector_token, '() 1 d -> b 1 d', b=b)
        x = torch.cat((projector_token, x), dim=1)
        token_pos = self.pos_emb(torch.tensor([[self.pos_emb.num_embeddings-1]] * b).to(x.device))
        pos_emb = torch.cat((token_pos, self.pos_emb(unmasked_indices)), dim=1)
        tokens = x + pos_emb
        projector = self.projector(tokens)
        return projector[:, 0]

    pass


class Projector2(nn.Module):

    def __init__(self, in_dim, pro_dim, num_patches, depth, heads, dim_head):
        super().__init__()
        self.enc_to_dec = nn.Linear(in_dim, pro_dim) if in_dim != pro_dim else nn.Identity()
        self.projector = Transformer(dim=pro_dim, depth=depth, heads=heads,
                                   dim_head=dim_head, mlp_dim=pro_dim * 4)
        self.projector_token = nn.Parameter(torch.randn(1, 1, pro_dim))
        self.mask_token = nn.Parameter(torch.randn(pro_dim))
        self.pos_emb = nn.Embedding(num_patches + 1, pro_dim)
        pass

    def forward(self, x, unmasked_indices, masked_indices):
        b = x.shape[0]
        device = x.device

        unmask_tokens = self.enc_to_dec(x)
        unmask_tokens = unmask_tokens + self.pos_emb(unmasked_indices)  # 未掩掉的块

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=b, n=masked_indices.size(1))
        mask_tokens = mask_tokens + self.pos_emb(masked_indices)  # 掩掉的块

        projector_token = repeat(self.projector_token, '() 1 d -> b 1 d', b=b)
        projector_indices = torch.tensor([[self.pos_emb.num_embeddings - 1]] * b).to(device)
        projector_token = projector_token + self.pos_emb(projector_indices)
        all_tokens = torch.cat([projector_token, unmask_tokens, mask_tokens], dim=1)

        projector = self.projector(all_tokens)
        return projector[:, 0]

    pass


@BACKBONES.register_module()
class SwinTransformer3DJoint(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self, pretrained=None, pretrained2d=False, patch_size=(4,4,4), in_chans=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=(2,7,7), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=False, frozen_stages=-1, use_checkpoint=False,
                 mask_ratio=0.5, generative=False, discriminative=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.generative = generative
        self.discriminative = discriminative

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2**i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer],
                               window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            pass

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        # Decoder
        if self.generative:
            self.decoder = Decoder(encoder_dim=768, decoder_dim=256, num_patches=49*16, decoder_depth=6,
                                   decoder_heads=4, decoder_dim_head=64, pixel_values_per_patch=6144)
            pass
        if self.discriminative:
            self.mask_ratio = 0.5
            # self.mask_ratio = 0.67

            self.has_predictor = True
            # projector = Projector
            # self.has_predictor = False
            projector = Projector2

            self.projector = projector(in_dim=768, pro_dim=512, num_patches=49*8, depth=3, heads=8, dim_head=64)
            self.projector_bn_last = nn.BatchNorm1d(512)
            if self.has_predictor:
                self.predictor = nn.Sequential(nn.Linear(512, 512 * 4), nn.BatchNorm1d(512 * 4),
                                               nn.ReLU(inplace=True), nn.Linear(512 * 4, 512))
                pass
            pass

        self._freeze_stages()
        pass

    def forward(self, x):
        device = x.device

        final_12_image = x
        if self.generative or self.discriminative:
            # 1:用于训练的，2:被扣掉的，3:被整体扣掉的
            x_b, x_c, x_t, x_w, x_h = x.size()
            p1, p2, p3 = self.patch_size[0], x_w // self.window_size[-2], x_h // self.window_size[-1]
            batch_range = torch.arange(x_b, device=device)[:, None]

            all_patch_img = rearrange(x, "b c (t p1) (w p2) (h p3) -> b (w h t) (p1 p2 p3 c)", p1=p1, p2=p2, p3=p3)
            all_patch_num = all_patch_img.size(1)  # 总共多少个块
            image_num_patch = (x_w // p2) * (x_h // p3)  # 空间上多少个块
            num_mask = int(image_num_patch * self.mask_ratio)  # mask多少个

            t_mask = torch.rand(x_b, 2, device=device).argsort(dim=-1)  # Mask: 前半部分还是后半部分
            unmask_time, final_3_global_id = rearrange(torch.cat([
                t_mask.unsqueeze(-1) * all_patch_num // 2 + one for one in range(all_patch_num // 2)], dim=2), "b c t -> c b t")
            final_3_data = all_patch_img[batch_range, final_3_global_id]
            unmask_time_x = all_patch_img[batch_range, unmask_time]

            unmask_time_from_x, _ = rearrange(torch.cat([
                t_mask.unsqueeze(-1) * x_t // 2 + one for one in range(x_t // 2)], dim=2), "b c t -> c b t")
            final_12_image = rearrange(x[batch_range, :, unmask_time_from_x], "b t c w h -> b c t w h")

            # 空间轴
            rand_indices = torch.rand(x_b, image_num_patch, device=device).argsort(dim=-1)


            ###########################################################################################################
            # 第一份
            ###########################################################################################################
            final_1_local_id = torch.cat([rand_indices[:, num_mask:] + image_num_patch * one for one in range(x_t // p1 // 2)], dim=1)
            final_2_local_id = torch.cat([rand_indices[:, :num_mask] + image_num_patch * one for one in range(x_t // p1 // 2)], dim=1)
            final_1_global_id = torch.cat([one.unsqueeze(0) + one_mask[0] * all_patch_num // 2 for one, one_mask in zip(final_1_local_id, t_mask)])
            final_2_global_id = torch.cat([one.unsqueeze(0) + one_mask[0] * all_patch_num // 2 for one, one_mask in zip(final_2_local_id, t_mask)])

            # final_1_data = unmask_time_x[batch_range, final_1_local_id]
            final_2_data = unmask_time_x[batch_range, final_2_local_id]
            # final_1_data = all_patch_img[batch_range, final_1_global_id]
            # final_2_data = all_patch_img[batch_range, final_2_global_id]
            final_23_global_id = torch.cat([final_2_global_id, final_3_global_id], dim=1)
            final_23_data = torch.cat([final_2_data, final_3_data], dim=1)
            ###########################################################################################################


            ###########################################################################################################
            # 第二份
            ###########################################################################################################
            final_1_local_id_2 = torch.cat([rand_indices[:, :rand_indices.shape[1] - num_mask] + image_num_patch * one for one in range(x_t // p1 // 2)], dim=1)
            final_2_local_id_2 = torch.cat([rand_indices[:, rand_indices.shape[1] - num_mask:] + image_num_patch * one for one in range(x_t // p1 // 2)], dim=1)
            final_1_global_id_2 = torch.cat([one.unsqueeze(0) + one_mask[0] * all_patch_num // 2 for one, one_mask in zip(final_1_local_id_2, t_mask)])
            final_2_global_id_2 = torch.cat([one.unsqueeze(0) + one_mask[0] * all_patch_num // 2 for one, one_mask in zip(final_2_local_id_2, t_mask)])

            # final_1_data_2 = unmask_time_x[batch_range, final_1_local_id_2]
            final_2_data_2 = unmask_time_x[batch_range, final_2_local_id_2]
            # final_1_data_2 = all_patch_img[batch_range, final_1_global_id_2]
            # final_2_data_2 = all_patch_img[batch_range, final_2_global_id_2]
            final_23_global_id_2 = torch.cat([final_2_global_id_2, final_3_global_id], dim=1)
            final_23_data_2 = torch.cat([final_2_data_2, final_3_data], dim=1)
            ###########################################################################################################
            pass

        x = self.patch_embed(final_12_image)
        x = self.pos_drop(x)
        for index, layer in enumerate(self.layers[:-1]):
            x = layer(x.contiguous())
            pass

        if not (self.generative or self.discriminative):  # 无
            x = self.layers[-1](x.contiguous())

            x = rearrange(x, 'n c d h w -> n d h w c')
            x = self.norm(x)
            x = rearrange(x, 'n d h w c -> n c d h w')
            return x
        else:  # 至少有一个
            x_1 = self.layers[-1](x.contiguous(), mae_unmask=final_1_local_id)
            x_2 = self.layers[-1](x.contiguous(), mae_unmask=final_1_local_id_2)
            x_1 = rearrange(x_1, 'n c d h w -> n d h w c')
            x_2 = rearrange(x_2, 'n c d h w -> n d h w c')

            B, D, H, W, C = x_1.shape
            window_size = get_window_size((D, H, W), self.window_size)
            batch_range = torch.arange(B, device=device)[:, None]

            x_1 = window_partition(x_1, window_size)
            x_2 = window_partition(x_2, window_size)
            x_1 = x_1[batch_range, final_1_local_id]
            x_2 = x_2[batch_range, final_1_local_id_2]

            result_dict = {}
            if self.generative:
                generative_output = self.decoder(x_1, final_1_global_id, final_23_global_id, final_23_data)
                generative_output2 = self.decoder(x_2, final_1_global_id_2, final_23_global_id_2, final_23_data_2)
                result_dict["generative_output"] = generative_output
                result_dict["generative_output2"] = generative_output2
                pass

            if self.discriminative:  # 判别，有两个
                x_1_pro = self.projector(x_1, final_1_local_id, final_2_local_id)
                x_2_pro = self.projector(x_2, final_1_local_id_2, final_2_local_id_2)
                h1, h2 = self.projector_bn_last(x_1_pro), self.projector_bn_last(x_2_pro)

                if self.has_predictor:
                    x_1_pre, x_2_pre = self.predictor(h1), self.predictor(h2)
                else:
                    x_1_pre, x_2_pre = h1.argmax(dim=1), h2.argmax(dim=1)

                result_dict["discriminative_output"] = (h1, h2, x_1_pre, x_2_pre, self.has_predictor)
                pass
            return result_dict
        pass

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        pass

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger,
                                map_location='cpu', revise_keys=[(r'^backbone\.', '')])
                pass
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        pass

    def inflate_weights(self, logger):
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2)\
                                                    .repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2*self.window_size[1]-1, 2*self.window_size[2]-1), mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()
        pass

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()
        pass

    pass

