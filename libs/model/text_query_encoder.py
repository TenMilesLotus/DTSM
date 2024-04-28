# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 10/3/2020 1:27 PM

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout
from torchvision.transforms import Resize 

def clones(_to_clone_module, _clone_times):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(_to_clone_module) for _ in range(_clone_times)])


def subsequent_mask(_target):
    batch_size = _target.size(0)
    sequence_length = _target.size(1)
    return torch.tril(torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool))


class MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(self, _multi_attention_heads, _dimensions, _dropout=0.1):
        """

        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert _dimensions % _multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(_dimensions / _multi_attention_heads)
        self.h = _multi_attention_heads
        self.linears = clones(nn.Linear(_dimensions, _dimensions), 4)  # (q, k, v, last output layer)
        self.attention = None
        self.dropout = nn.Dropout(p=_dropout)

    @torch.jit.script_method
    def dot_product_attention(self, _query, _key, _value, _mask):
        """
        Compute 'Scaled Dot Product Attention

        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        d_k = _value.size(-1)
        score = torch.matmul(_query, _key.transpose(-2, -1)) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
        if _mask is not None:
            score = score.masked_fill(_mask == 0, -1e9)  # score (N, h, seq_len, seq_len)
        p_attn = F.softmax(score, dim=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return torch.matmul(p_attn, _value), p_attn

    @torch.jit.script_method
    def forward(self, _query, _key, _value, _mask):
        batch_size = _query.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        _query, _key, _value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (_query, _key, _value))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(_query, _key, _value, _mask=_mask)
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, _dimensions, _feed_forward_dimensions, _dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(_dimensions, _feed_forward_dimensions)
        self.w_2 = nn.Linear(_feed_forward_dimensions, _dimensions)
        self.dropout = nn.Dropout(p=_dropout)

    def forward(self, _input_tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(_input_tensor))))


class PositionalEncoding(torch.jit.ScriptModule):
    """Implement the PE function."""

    def __init__(self, _dimensions, _dropout=0.1, _max_len=5000):
        """

        :param _dimensions:
        :param _dropout:
        :param _max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=_dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(_max_len, _dimensions)
        position = torch.arange(0, _max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, _dimensions, 2).float() *
                             -(math.log(10000.0) / _dimensions))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    @torch.jit.script_method
    def forward(self, _input_tensor):
        _input_tensor = _input_tensor + self.pe[:, :_input_tensor.size(1)]  # pe 1 5000 512
        return self.dropout(_input_tensor)

class Text_query_encoder(nn.Module):
    def __init__(self, _multi_heads_count, _dimensions, _stacks, _dropout, _feed_forward_size):
        super(Text_query_encoder, self).__init__()
        self.position = PositionalEncoding(_dimensions, _dropout, _max_len=5000)
        self.layer_norm = torch.nn.LayerNorm(_dimensions, eps=1e-6)
        self.stacks = _stacks
        self.dropout = Dropout(_dropout)
        self.attention = MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
        self.position_feed_forward = PositionwiseFeedForward(_dimensions, _feed_forward_size, _dropout)

    def eval(self):
        self.position.eval()
        self.attention.eval()
        self.position_feed_forward.eval()
        self.layer_norm.eval()
        self.dropout.eval()

    def _generate_mask(self, _position_encode_tensor):
        target_length = _position_encode_tensor.size(1)
        return torch.ones((target_length, target_length), device=_position_encode_tensor.device)

    def forward(self, _input_tensor, text_mask_tensor):  # torch.Size([2, 240, 512])
        b, c, h, w = _input_tensor.shape
        _input_tensor = _input_tensor.view(b, c, h * w).permute((0, 2, 1))
        _input_tensor = self.position(_input_tensor)
        source_mask = self._generate_mask(_input_tensor)

        text_mask_tensor = text_mask_tensor.view(b, c, h * w).permute((0, 2, 1))
        text_mask_tensor = self.position(text_mask_tensor)
        for i in range(self.stacks):
            normed_input = self.layer_norm(_input_tensor)
            normed_line = self.layer_norm(text_mask_tensor)
            output = normed_input + self.dropout(
                self.attention(normed_line, normed_input, normed_input, source_mask)
            )
            normed_output = self.layer_norm(output)
            output = normed_output + self.dropout(self.position_feed_forward(normed_output))
        output = self.layer_norm(output)
        output = output.permute((0, 2, 1)).view(b, c, h, w)
        return output