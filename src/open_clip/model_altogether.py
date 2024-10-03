# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from enum import Enum
from typing import Tuple, Optional, Union
from torch import nn
from torch.nn import functional as F

from transformers import AutoModelForCausalLM


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=F.relu, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self

        layers = []
        for i in range(num_layers):
            layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super().__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        self.dim_clip = dim_clip

    def forward(self, x):
        if x.size(-1) != self.dim_clip:
            x = x.reshape(-1, self.dim_clip)  # support anyresflat

        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out


class Altogether(nn.Module):
    def __init__(
        self, prefix_length: int, pad_token_id: int, clip_length = None, 
        clip_emb_size = 512, num_layers = 8, decoder="facebook/opt-1.3b", **kwargs
    ):
        super().__init__()
        self.prefix_length = prefix_length

        # from src.open_clip.model_altogether.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        # replace_llama_attn_with_flash_attn()
        self.gpt = AutoModelForCausalLM.from_pretrained(
            decoder,
            device_map=kwargs["device"],
            pad_token_id=pad_token_id,
            attn_implementation="flash_attention_2",
        )

        self.gpt_embedding_size = self.gpt.get_input_embeddings().weight.shape[1]
        self.clip_project = TransformerMapper(
            clip_emb_size, self.gpt_embedding_size, prefix_length, clip_length, num_layers)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.get_input_embeddings()(tokens)
        prefix_projections = self.clip_project(prefix).reshape(embedding_text.size(0), -1, self.gpt_embedding_size)
        embedding_cat = torch.cat([prefix_projections, embedding_text], dim=1)

        out = self.gpt(inputs_embeds=embedding_cat, attention_mask=attention_mask)
        return out
