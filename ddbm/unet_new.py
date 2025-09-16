from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    LayerNorm2d
)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    

# ADD: Cross Attention Block 
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        # Present: GroupNorm
        self.q_norm = torch.nn.GroupNorm(32 if channels >= 32 else 1, channels, affine=True)
        self.kv_norm = torch.nn.GroupNorm(32 if channels >= 32 else 1, channels, affine=True)

        self.q = conv_nd(1, channels, channels, 1)
        self.kv = conv_nd(1, channels, channels*2, 1)
        
        # TODO: use_new_attention_order=True, need to change "not" 
        if use_new_attention_order:
            self.attention = QKVCrossAttention(self.num_heads)
        else:
            self.attention = CWCrossAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond):
        return checkpoint(self._forward, (x,cond), self.parameters(), self.use_checkpoint)

    def _forward(self, x, cond):
        b, c, *spatial = x.shape
        assert c == self.channels, f"Q ch {c} != block {self.channels}"
        assert x.shape == cond.shape, f"x.shape != cond.shape"
        # Reshape
        x = x.reshape(b, c, -1)  # Batchsize, Channels, 256*256
        cond = cond.reshape(b, c, -1)
        # Norm -> Proj
        q = self.q(self.q_norm(x))
        kv = self.kv(self.kv_norm(cond))
        # out
        h = self.attention(q, kv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class DBCRCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
    
        # Layer Norm
        self.q_norm = LayerNorm2d(channels)
        self.kv_norm = LayerNorm2d(channels)
        
        # 1x1 Conv Layer
        self.q_proj = conv_nd(2, channels, channels, 1)
        self.k_proj = conv_nd(2, channels, channels, 1)
        self.v_proj = conv_nd(2, channels, channels, 1)

        # mlp
        self.mlp = nn.Sequential(
            conv_nd(2, channels, channels*2, 1),
            nn.GELU(),
            conv_nd(2, channels*2, channels, 1), 
        )

    def forward(self, x, cond):
        return checkpoint(self._forward, (x, cond), self.parameters(), self.use_checkpoint)

    def _forward(self, x, cond):
        assert x.shape == cond.shape, 'the shape of x does not equal to cond'
        B, C, H, W = x.shape
        HW = H*W

        # normalization & proj
        q = self.q_proj(self.q_norm(x))
        k = self.k_proj(self.kv_norm(cond))
        v = self.v_proj(self.kv_norm(cond))

        # heads: (B, C, HW)
        # Single Head 
        q_head = q.view(B, C, HW)
        k_head = k.view(B, C, HW)
        v_head = v.view(B, C, HW)

        # q_head: (B, C, HW), k_head^t: (B, HW, C)
        # attn: (B, C, C)
        attn = torch.matmul(q_head, k_head.transpose(-2, -1))
        attn = attn * (HW ** -0.5)
        attn = attn.softmax(dim=-1)

        # attn: (B, C, C), v_head: (B, C, HW)
        # z: (B, C, HW)
        z = torch.matmul(attn, v_head)
        z = z.reshape(B, C, H, W)

        z_sum = x + z
        out = z_sum + self.mlp(z_sum)
        return out

    
# ADD: Cross Attention
class CWCrossAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, kv):
        bs, width, length = q.shape
        ch = width // self.n_heads

        k, v = kv.chunk(2, dim=1)
        
        q = q.reshape(bs*self.n_heads, ch, length)
        k = k.reshape(bs*self.n_heads, ch, length)
        v = v.reshape(bs*self.n_heads, ch, length)
        
        a = F.scaled_dot_product_attention(q, k, v)
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        a = F.scaled_dot_product_attention(q, k, v)
        return a.transpose(-2, -1).reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# ADD:Multi-modal Cross Attention
class QKVCrossAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, kv):
        bs, width, length = q.shape
        ch = width // self.n_heads

        k, v = kv.chunk(2, dim=1)

        q = q.reshape(bs*self.n_heads, ch, length)
        k = k.reshape(bs*self.n_heads, ch, length)
        v = v.reshape(bs*self.n_heads, ch, length)
        
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        a = F.scaled_dot_product_attention(q, k, v)
        return a.transpose(-2, -1).reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = (
            q.reshape(bs * self.n_heads, ch, length),
            k.reshape(bs * self.n_heads, ch, length),
            v.reshape(bs * self.n_heads, ch, length),
        )
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        a = F.scaled_dot_product_attention(q, k, v)
        return a.transpose(-2, -1).reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class NAFBlock(TimestepBlock):
    def __init__(
        self, 
        c,
        emb_channels,
        DW_Expand=2, 
        FFN_Expand=2,
        drop_out_rate=0.,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        dims=2,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        dw_channel = c * DW_Expand
        self.conv1 = conv_nd(dims, c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = conv_nd(dims, dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = conv_nd(dims, dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_nd(dims, dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = conv_nd(dims, c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = conv_nd(dims, ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # emb_layers          
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * c if use_scale_shift_norm else c,
            ),
        )

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.norm1(x)

        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.sg(h)

        h = x * self.sca(h)
        h = self.conv3(h)
        h = self.dropout1(h)
        y = x + h * self.beta

        z = self.norm2(y)
        z = self.conv4(z)
        z = self.sg(z)
        z = self.conv5(z)
        z = self.dropout2(z)

        return y + z * self.gamma

class SARBlock(nn.Module):
    def __init__(
        self, 
        c,
        DW_Expand=2, 
        FFN_Expand=2,
        drop_out_rate=0.,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        dims=2,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        dw_channel = c * DW_Expand
        self.conv1 = conv_nd(dims, c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = conv_nd(dims, dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = conv_nd(dims, dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_nd(dims, dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = conv_nd(dims, c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = conv_nd(dims, ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    # def forward(self, x):
    #     return checkpoint(self._forward, x, self.parameters(), self.use_checkpoint)

    def forward(self, x):
        h = self.norm1(x)

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.sg(h)

        h = x * self.sca(h)
        h = self.conv3(h)
        h = self.dropout1(h)
        y = x + h * self.beta

        z = self.norm2(y)
        z = self.conv4(z)
        z = self.sg(z)
        z = self.conv5(z)
        z = self.dropout2(z)

        return y + z * self.gamma

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        condition_mode=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.condition_mode = condition_mode

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels = in_channels * 2 if condition_mode == "concat" else in_channels

        self.attn_cnt = {i: 1 * self.num_res_blocks for i in attention_resolutions}
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self.input_cond_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, 2, ch, 3, padding=1))])
        self.attn_blocks = nn.ModuleList([])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                sar_layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    sar_layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    self.attn_blocks.append(
                        CrossAttentionBlock(
                            ch, 
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_cond_blocks.append(TimestepEmbedSequential(*sar_layers))

                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                self.input_cond_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, xT=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.condition_mode == "concat":
            x = torch.cat([x, xT], dim=1)
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        hs = []

        timesteps = timesteps.to(self.dtype)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            print("self.num_classes is not None")
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.to(self.dtype)
        s = y.to(self.dtype)
        
        attn = 0
        ds = 1
        ds_flag = self.image_size
        attn_budget = dict(self.attn_cnt)
        for idx in range(len(self.input_blocks)):

            h = self.input_blocks[idx](h, emb)
            s = self.input_cond_blocks[idx](s, emb)
            if ds in self.attention_resolutions and attn_budget[ds] > 0:
                h = self.attn_blocks[attn](h, s)
                attn += 1
                attn_budget[ds] -= 1

            if ds_flag != h.shape[-1]:
                ds_flag = h.shape[-1]
                ds *= 2

            hs.append(h)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = self.out(h)
        h = h.to(x.dtype)
        return h


class NAFNetModel(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_naf_blocks,
        dropout,
        middle_blk_num,
        enc_blk_nums,
        dec_blk_nums,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        condition_mode=None,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_naf_blocks = num_naf_blocks
        self.dropout = dropout
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.condition_mode = condition_mode

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = model_channels
        in_channels = in_channels * 2 if condition_mode == "concat" else in_channels

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(
            conv_nd(dims, in_channels, ch, 3, padding=1))])
        

        self.input_cond_blocks = nn.ModuleList([conv_nd(dims, 2, ch, 3, padding=1)])
        
        self.attn_blocks = nn.ModuleList([])
        self.middle_blocks = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])

        # Encoder 
        for idx, num in enumerate(enc_blk_nums):
            for _ in range(num_naf_blocks*num):
                self.input_blocks.append(TimestepEmbedSequential(
                    NAFBlock(
                        ch, 
                        time_embed_dim,
                        drop_out_rate=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                ))
                self.input_cond_blocks.append(
                    SARBlock(
                        ch,
                        drop_out_rate=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                )
            self.attn_blocks.append(
                DBCRCrossAttentionBlock(
                    ch, 
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                )
            )
            # downsampling
            self.input_blocks.append(TimestepEmbedSequential(
                conv_nd(dims, ch, ch * 2, 2, 2)
            ))
            if idx != len(enc_blk_nums) - 1:
                self.input_cond_blocks.append(
                    conv_nd(dims, ch, ch * 2, 2, 2)
                )
            ch = ch * 2

        # Middle
        for _ in range(num_naf_blocks*middle_blk_num):
            self.middle_blocks.append(TimestepEmbedSequential(
                NAFBlock(
                    ch, 
                    time_embed_dim,
                    drop_out_rate=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_checkpoint=use_checkpoint,
                )
            ))

        # Decoder
        for num in dec_blk_nums:
            self.output_blocks.append(
                TimestepEmbedSequential(
                    conv_nd(dims, ch, ch * 2, 1, bias=False),
                    nn.PixelShuffle(2),
            ))
            ch = ch // 2
            for _ in range(num_naf_blocks*num):
                self.output_blocks.append(TimestepEmbedSequential(
                    NAFBlock(
                        ch, 
                        time_embed_dim,
                        drop_out_rate=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                ))

        self.output_blocks.append(
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1))
        )

    # y = SAR
    def forward(self, x, timesteps, xT=None, y=None):
        if self.condition_mode == "concat":
            x = torch.cat([x, xT], dim=1)

        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        timesteps = timesteps.to(self.dtype)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            print("self.num_classes is not None")
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # type
        h = x.to(self.dtype)
        s = y.to(self.dtype)

        # input embedding
        h = self.input_blocks[0](h, emb)
        s = self.input_cond_blocks[0](s)
        enc = 1
        for i, num in enumerate(self.enc_blk_nums):
            for _ in range(num*self.num_naf_blocks):
                h = self.input_blocks[enc](h, emb)
                s = self.input_cond_blocks[enc](s)
                enc += 1
            
            # cross-attention
            h = self.attn_blocks[i](h, s)

            # downsampling
            h = self.input_blocks[enc](h, emb)
            if i != len(self.enc_blk_nums) - 1:
                s = self.input_cond_blocks[enc](s)
            enc += 1

        for modules in self.middle_blocks:
            h = modules(h, emb)

        dec = 0
        for num in self.dec_blk_nums:
            h = self.output_blocks[dec](h, emb)
            dec += 1

            for _ in range(num*self.num_naf_blocks):
                h = self.output_blocks[dec](h, emb)
                dec += 1
        
        h = self.output_blocks[dec](h)
        h = h.to(x.dtype)
        
        return h


if __name__=="__main__":
    
    import torch
    from torchinfo import summary
    from ddbm.unet import UNetModel, NAFNetModel

    # model = UNetModel(
    #     image_size=256,
    #     in_channels=13,
    #     model_channels=64,
    #     out_channels=13,
    #     num_res_blocks=2,
    #     attention_resolutions=[16,8,4],
    #     dropout=0.1,
    #     channel_mult=(1, 1, 2, 2, 4, 4),
    #     conv_resample=True,
    #     dims=2,
    #     num_classes=None,
    #     use_checkpoint=True,
    #     use_fp16=False,
    #     num_heads=4,
    #     num_head_channels=64,
    #     num_heads_upsample=-1,
    #     use_scale_shift_norm=False,
    #     resblock_updown=True,
    #     use_new_attention_order=True,
    #     condition_mode=None,
    # )

    model = NAFNetModel(
        image_size=256,
        in_channels=4,
        model_channels=16,
        out_channels=4,
        num_naf_blocks=2,
        dropout=0,
        middle_blk_num=6,
        enc_blk_nums=[1,1,2,4],
        dec_blk_nums=[1,1,1,1],
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        condition_mode=None,
        use_new_attention_order=True,
        num_heads=4,
        num_head_channels=16,   
    )
    x = torch.randn(2, 4, 256, 256)
    t = torch.tensor([0, 0])
    y = torch.randn(2, 2, 256, 256)

    summary(model, input_data={'x':x, 'timesteps':t, 'y':y}, depth=4)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
