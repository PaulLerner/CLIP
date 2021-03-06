from collections import OrderedDict
from typing import Tuple, Union, Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

from .simple_tokenizer import SimpleTokenizer as _Tokenizer, EOT_STR


_tokenizer = _Tokenizer()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Note, in torch.nn this is more or less equivalent to TransformerEncoderLayer"""
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: Tensor):
        x = self.ln_1(x)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x + self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class SingleheadTransformerDecoderLayer(nn.Module):
    """Adapted from ResidualAttentionBlock and torch.nn.TransformerDecoderLayer
    This should be more or less equivalent to torch.nn.TransformerDecoderLayer
    but using SingleheadAttention for cross-attention instead of nn.MultiheadAttention

    Note that n_head is only used for self-attention !
    Cross-attention only has one head

    Also kdim and vdim are only passed to cross-attention,
    input to self-attention have the same dimension by definition
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None,
                 dropout: float = 0., bias: bool = True, add_bias_kv: bool = False,
                 add_zero_attn: bool = False, kdim: int = None, vdim: int = None):
        super().__init__()

        # note we leave this ambiguous "attn" name to easily load weights
        # from a pre-trained ResidualAttentionBlock
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          bias=bias, add_bias_kv=add_bias_kv,
                                          add_zero_attn=add_zero_attn)
        self.ln_1 = LayerNorm(d_model)

        self.cross_attn = SingleheadAttention(d_model, dropout=dropout,
                                              bias=bias, add_bias_kv=add_bias_kv,
                                              add_zero_attn=add_zero_attn,
                                              kdim=kdim, vdim=vdim)
        # likewise here we leave ln_2 for the pre-mlp normalization
        self.ln_cross_attn = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: Tensor, memory: Tensor,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        r"""
        Pass the inputs (and mask) through the decoder layer.

        Args:
            x: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        x = self.ln_1(x)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x + self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        x = self.ln_cross_attn(x)
        x = x + self.cross_attn(x, memory, memory, need_weights=False,
                                attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


def single_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Adapted from torch.nn.functional.multi_head_attention_forward, see SingleheadAttention

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim
    scaling = float(head_dim) ** -0.5

    q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
    len1, len2 = q_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == query.size(-1)

    k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
    len1, len2 = k_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == key.size(-1)

    v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
    len1, len2 = v_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == value.size(-1)

    if in_proj_bias is not None:
        q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
        k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
        v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
    else:
        q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
        k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
        v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, 1, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, 1, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1)
    else:
        return attn_output, None


class SingleheadAttention(nn.Module):
    """Adapted from torch.nn.MultiheadAttention
    We assume that head_dim == embed_dim which allows to remove the final projection layer (torch.nn.MultiheadAttention.out_proj)

    This restricts the use of a single attention head
    """

    def __init__(self, embed_dim, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None):
        super(SingleheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim

        self.q_proj_weight = nn.Parameter(Tensor(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(Tensor(embed_dim, self.kdim))
        self.v_proj_weight = nn.Parameter(Tensor(embed_dim, self.vdim))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        return single_head_attention_forward(
            query, key, value, self.embed_dim,
            self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight)


class Transformer(nn.Module):
    """Note, in torch.nn this is more or less equivalent to TransformerEncoder"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: Tensor):
        return self.resblocks(x)


class TransformerDecoder(nn.Module):
    """Adapted from Transformer and torch.nn.TransformerDecoder
    This should be more or less equivalent to torch.nn.TransformerDecoder
    but using SingleheadTransformerDecoderLayer instead of torch.nn.TransformerDecoderLayer

    Additional arguments are passed to SingleheadTransformerDecoderLayer
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None, **kwargs):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([SingleheadTransformerDecoderLayer(width, heads, attn_mask, **kwargs) for _ in range(layers)])

    def forward(self, x: Tensor, memory: Tensor,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        """See SingleheadTransformerDecoderLayer"""
        for mod in self.resblocks:
            x = mod(x, memory, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask)
        return x


class BaseVisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

    def forward(self, x: Tensor):
        """Runs through the Transformer but doesn't project in the multimodal space
        Also keeps all of the tokens' hidden states instead of keeping only the class_embedding"""
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x



class VisualTransformer(BaseVisualTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__(input_resolution, patch_size, width, layers, heads)

        self.output_dim = output_dim
        scale = width ** -0.5
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: Tensor):
        """Keeps only class embedding as final representation and project it in the multimodal space"""
        x = super().forward(x)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class BaseCLIP(nn.Module):
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 ):
        """Beware this doesn't call initialize_parameters, you should call it in the child class !"""
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

    def initialize_parameters(self):
        """Note that (Base)VisualTransformer parameters are initialized in its constructor"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


class CLIP(BaseCLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__(context_length, vocab_size, transformer_width)
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        super().initialize_parameters()
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class CLIPDecoder(BaseCLIP):
    """Same as CLIP but uses a Transformer Decoder for the text instead of an 'Encoder'
    i.e. adds cross attention between the text Transformer and the vision Transformer
    Also adds a final classification layer to predict the next token in the text

    Note that TransformerDecoder cross-attention parameters are initialized in SingleheadAttention

    Note that text is named "input_ids" to respect HF-transformers convention
    """
    def __init__(self,
                 embed_dim: int,
                 # vision encoder
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # multimodal decoder
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 decoder_dropout: float = 0.,
                 bias: bool = True,
                 add_bias_kv: bool = False,
                 add_zero_attn: bool = False,
                 separator: int = _tokenizer.encode("?")[0],
                 eos: int = _tokenizer.encode(EOT_STR)[0]
                 ):
        super().__init__(context_length, vocab_size, transformer_width)
        if isinstance(vision_layers, (tuple, list)):
            raise NotImplementedError("Cannot use ModifiedResNet for now, please use VisualTransformer instead")
        else:
            vision_heads = vision_width // 64
            self.visual = BaseVisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads
            )
        self.transformer = TransformerDecoder(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=decoder_dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=vision_width,
            vdim=vision_width
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)
        self.separator = separator
        self.eos = eos
        self.initialize_parameters()

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def forward(self, input_ids, image, **kwargs):
        """
        Parameters
        ----------
        input_ids: Tensor
            (batch_size, context_length)
            Beware this is the first argument unlike in CLIP and BaseCLIP
        image: Tensor
            (batch_size, in_channels, height, width)
            Beware this is the second argument unlike in CLIP and BaseCLIP
        **kwargs: additionnal arguments (e.g. strings) are ignored

        Returns
        -------
        x: Tensor
            (batch_size, context_length, embed_dim)
        """
        # encode image with visual encoder
        image_features = self.encode_image(image)

        # embed text
        x = self.token_embedding(input_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        # fix shape and pass through the multimodal transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        image_features = image_features.permute(1, 0, 2)
        x = self.transformer(x, image_features,
                             memory_mask=None, memory_key_padding_mask=None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # predict the next token
        x = self.linear(x)
        return self.log_softmax(x)

    def generate(self, input_ids, image, *args, **kwargs):
        """
        Ignores additional arguments.
        Disables gradient calculation.
        Returns Greedy decoding.
        """
        with torch.no_grad():
            return self.greedy_decoding(input_ids, image)

    def greedy_decoding(self, input_ids, image):
        """
        Parameters
        ----------
        see forward

        Returns
        -------
        prediction: Tensor
            (batch_size, context_length)
            Same as input after predicting the answer
        """
        # find separator ("?") in the input
        batch_size = input_ids.shape[0]
        max_index_batch = torch.full((batch_size,), self.context_length - 1, device=input_ids.device)
        where = input_ids == self.separator
        nonzero = where.nonzero(as_tuple=True)
        # no separator in the batch
        if not where.any():
            raise ValueError(f"didn't find the separator '{self.separator}' in the batch:\n{where}")
        # exactly one separator per item in the batch (this should always be the case)
        elif nonzero[0].shape[0] == batch_size and nonzero[0].unique().shape[0] == batch_size:
            first_where = nonzero[1]
            all_items = nonzero[0]
        # multiple separators per item in the batch
        else:
            raise ValueError(f"multiple separators '{self.separator}' in the batch:\n{where}")
        first_where = first_where.minimum(max_index_batch)

        # encode image with visual encoder
        image_features = self.encode_image(image)

        # embed text
        token_embeddings = self.token_embedding(input_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]
        question = token_embeddings + self.positional_embedding.type(self.dtype)

        # fix shape and pass through the multimodal transformer
        question = question.permute(1, 0, 2)  # NLD -> LND
        image_features = image_features.permute(1, 0, 2)
        question = self.transformer(question, image_features,
                                    memory_mask=None, memory_key_padding_mask=None)
        question = question.permute(1, 0, 2)  # LND -> NLD

        # predict the next token
        prediction = input_ids
        question = self.linear(question)
        answer = question[all_items, first_where].argmax(-1)

        # did we predict EOS ?
        reached_eos = answer == self.eos
        first_where = (first_where + 1).minimum(max_index_batch)
        prediction[all_items, first_where] = answer

        # sequential decoding until max_length or eos (update input w.r.t prediction)
        while not torch.logical_or((first_where == max_index_batch), reached_eos).all():
            token_embeddings[all_items, first_where] = self.token_embedding(answer).type(self.dtype)
            question = token_embeddings + self.positional_embedding.type(self.dtype)
            question = question.permute(1, 0, 2)  # NLD -> LND
            # TODO: cache previous hidden-states instead of recomputing every time
            question = self.transformer(question, image_features,
                                        memory_mask=None, memory_key_padding_mask=None)
            question = question.permute(1, 0, 2)  # LND -> NLD

            # predict the next token
            question = self.linear(question)
            answer = question[all_items, first_where].argmax(-1)

            # did we predict EOS ? (previously OR at this step)
            reached_eos = torch.logical_or(reached_eos, answer == self.eos)
            first_where = (first_where + 1).minimum(max_index_batch)
            prediction[all_items, first_where] = answer

        return prediction


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, SingleheadAttention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, training=False, Class=CLIP, fp16=True, context_length=None, pretrained=True, **kwargs):
    """

    Parameters
    ----------
    state_dict: OrderedDict[Tensor]
    training: bool, optional
    Class: type, optional
    fp16: bool, optional
    context_length: int, optional
        Defaults to the size of the pre-trained model (i.e. in state_dict)
    pretrained: bool, optional
        Whether to load weights from pre-trained model (default) or just the config
        jit always loads pre-trained weights.
    **kwargs: additional arguments are passed to Class

    Returns
    -------
    model: BaseCLIP
    """
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    # resize positional embedding if necessary (i.e. when fine-tuning context is different from pre-training context)
    old_context_length = state_dict["positional_embedding"].shape[0]
    if context_length is None:
        context_length = old_context_length
    elif context_length < old_context_length:
        state_dict["positional_embedding"] = state_dict["positional_embedding"][: context_length]
    elif context_length > old_context_length:
        raise NotImplementedError(f"Target context length {context_length} is greater "
                                  f"than pre-trained context length {old_context_length}")

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = Class(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads,
        transformer_layers, **kwargs
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # map pre-trained weights to CLIPDecoder
    if isinstance(model, CLIPDecoder):
        # remove irrelevant pre-trained weights
        for weight in ['logit_scale', 'visual.ln_post.bias', 'visual.ln_post.weight']:
            state_dict.pop(weight, None)

        # embedding matrix to classification layer
        state_dict["linear.weight"] = state_dict["token_embedding.weight"]

        # attention in torch uses F.linear so we have to transpose the projection matrices
        text_projection = state_dict.pop("text_projection").T
        visual_projection = state_dict.pop("visual.proj").T
        ln_final_bias = state_dict.pop('ln_final.bias')
        ln_final_weight = state_dict.pop('ln_final.weight')
        for i in range(transformer_layers):
            # visual multimodal projection to cross-attention projection
            state_dict[f"transformer.resblocks.{i}.cross_attn.k_proj_weight"] = visual_projection
            state_dict[f"transformer.resblocks.{i}.cross_attn.v_proj_weight"] = visual_projection

            # text multimodal projection to cross-attention projection
            state_dict[f"transformer.resblocks.{i}.cross_attn.q_proj_weight"] = text_projection

            # don't forget the layer normalization
            state_dict[f"transformer.resblocks.{i}.ln_cross_attn.bias"] = ln_final_bias
            state_dict[f"transformer.resblocks.{i}.ln_cross_attn.weight"] = ln_final_weight

    if pretrained:
        loading_output = model.load_state_dict(state_dict, strict=not isinstance(model, CLIPDecoder))
        if loading_output.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in state_dict:\n{loading_output.unexpected_keys}")
        if loading_output.missing_keys:
            print(f"The following keys were not loaded from state_dict:\n{loading_output.missing_keys}")

    if fp16:
        convert_weights(model)

    return model.train(training)
