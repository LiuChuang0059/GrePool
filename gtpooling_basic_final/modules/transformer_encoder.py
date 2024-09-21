import random

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor

import copy
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.parameter import Parameter
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from .mha import MultiheadAttention
import models.gnn as gnn


def random_select_tokens(cls_attn, left_tokens):
    B, N = cls_attn.shape
    select_ids = [torch.LongTensor(random.sample(range(N), left_tokens)) for i in range(B)]
    select_ids = torch.stack(select_ids, 0)
    return select_ids.cuda()


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, token_ratio=0.5, dropout_attn: float = 0.1) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = nhead

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout_attn = Dropout(dropout_attn)

        self.gumbel = nn.Linear(d_model, 1)
        self.keep_rate = token_ratio

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        # TODO: 排除 CLS token，它应该在最后一位

        x = src
        # X shape : [Length, batch size, feature_dim]

        # raise Exception("Pause!")
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            tmp, index, idx, noise_index, noise_idx, cls_attn, left_tokens, noise_tokens = self._sa_block(x, src_mask,
                                                                                                          src_key_padding_mask)
            x = self.norm1(x + tmp)  # N, B, C = x.shape

            x = x.permute(1, 0, 2)  # B, N, C = x.shape
            if index is not None:
                non_cls = x[:, :-1]
                cls = torch.unsqueeze(x[:, -1], 1)  # [B, 1, C]
                if src_key_padding_mask is not None:
                    src_key_padding_mask = torch.gather(src_key_padding_mask, dim=1, index=idx)
                    zeros = src_key_padding_mask.data.new(src_key_padding_mask.size(0), 1).fill_(0)
                    src_key_padding_mask = torch.cat([src_key_padding_mask, zeros], dim=1)

                x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]
                x_noise = torch.gather(non_cls, dim=1, index=noise_index)  # [B, noise_tokens, C]
                x = torch.cat([x_others, cls], dim=1)  # [B, left_tokens + 1, C]
                noise_x = x_noise  # torch.cat([x_noise, cls], dim=1)  # [B, noise_tokens + 1, C]

            x = x.permute(1, 0, 2)
            noise_x = noise_x.permute(1, 0, 2)

            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))

            x = self.norm2(x + self._ff_block(x))
            noise_x = self.norm3(noise_x + self._ff_block(noise_x))
            n_tokens = x.shape[0] - 1

        # return x, n_tokens, idx, src_key_padding_mask
        return x, noise_x, n_tokens, idx, src_key_padding_mask

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:

        N, B, C = x.shape
        # x size: [Length, batch size, feature_dim];
        # attn size: [B, H, N + 1, N + 1]
        # feature_dim = num_head * head_dim
        q, x, attn = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        # q: [B * nheads, N, head_dim]
        q_out = q.view(B, self.num_heads, N, -1)
        # print("x: {}".format(x.size()))
        # print("q_out: {}".format(q_out.size()))
        attn_out = attn @ q_out
        # print("attn_out: {}".format(attn_out.size()))
        # ignoring average head maybe [B, H, N + 1, N + 1]
        # print("attn: {}".format(attn.size()))
        cls_attn = attn[:, :, -1, :-1]  # size: [B, H, 1, N]
        # print("cls attn: {}".format(cls_attn.size()))
        # hoping head importance size to be [B, H, N + 1]
        head_importance = attn_out[:, :, :-1, :].norm(dim=-1)
        # print("head importance: {}".format(head_importance.size()))
        head_importance = head_importance / (head_importance.sum(dim=1, keepdims=True) + 1e-8)
        # print("head importance: {}".format(head_importance.size()))
        cls_attn = cls_attn * head_importance
        # print("cls attn: {}".format(cls_attn.size()))
        cls_attn = cls_attn.sum(dim=1)

        # drop tokens
        cls_attn = self.dropout_attn(cls_attn)
        left_tokens = math.ceil(self.keep_rate * (N - 1))
        noise_tokens = (N - 1) - left_tokens
        if noise_tokens == 0:
            noise_token = 1
        # topk for important tokens
        _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
        # TODO:
        # 1) random drop
        #  select_idx = random_select_tokens(cls_attn, left_tokens)
        # 2) reverse attn drop(bottomk)
        #    bottomk for important tokens
        # reverse_cls_attn = -cls_attn
        #
        _, noise_idx = torch.topk(-cls_attn, noise_tokens, dim=1, largest=True, sorted=True)  # [B, noise_tokens]
        noise_index = noise_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, noise_tokens, C]

        return self.dropout1(x), index, idx, noise_index, noise_idx, cls_attn, left_tokens, noise_tokens

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # self.block_skip_gating = nn.Parameter(torch.Tensor([-1, 1]).expand(num_layers, 2).clone())
        # self.use_gumbel = True
        # self.gumbel_hard = True

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        # save noise for first layer
        noise = None
        left_tokens = []
        idxs = []
        for i, mod in enumerate(self.layers):
            # output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            # output, left_token, idx, src_key_padding_mask = mod(output, src_mask=mask,
            #                                                     src_key_padding_mask=src_key_padding_mask)
            output, output_noise, left_token, idx, src_key_padding_mask = mod(output, src_mask=mask,
                                                                              src_key_padding_mask=src_key_padding_mask)
            left_tokens.append(left_token)
            idxs.append(idx)
            if i == 0:
                noise = output_noise

        if self.norm is not None:
            output = self.norm(output)
            output_noise = self.norm(output_noise)

        return output, output_noise, left_tokens, idxs


class TransformerNodeEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("transformer")
        group.add_argument("--d_model", type=int, default=128, help="transformer d_model.")
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--transformer_dropout", type=float, default=0.3)
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--num_encoder_layers", type=int, default=4)
        group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
        group.add_argument("--transformer_norm_input", action="store_true", default=False)
        group.add_argument("--token_ratio", type=float, default=0.5)
        group.add_argument("--dropout_attn", type=float, default=0.1)

    def __init__(self, args):
        super().__init__()

        self.d_model = args.d_model
        self.num_layer = args.num_encoder_layers
        # Creating Transformer Encoder Model

        # encoder_layer = nn.TransformerEncoderLayer(
        encoder_layer = TransformerEncoderLayer(
            args.d_model, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation,
            token_ratio=args.token_ratio, dropout_attn=args.dropout_attn
        )
        encoder_norm = nn.LayerNorm(args.d_model)
        self.transformer = TransformerEncoder(encoder_layer, args.num_encoder_layers, encoder_norm)
        self.max_input_len = args.max_input_len

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.d_model)
        self.cls_embedding = None
        if args.graph_pooling == "cls":
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, args.d_model], requires_grad=True))

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)
            if src_padding_mask is not None:
                zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
                src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out, transformer_noise, left_tokens, idxs = self.transformer(padded_h_node,
                                                                                 src_key_padding_mask=src_padding_mask)  # (S, B, h_d)

        return transformer_out, transformer_noise, src_padding_mask, idxs
