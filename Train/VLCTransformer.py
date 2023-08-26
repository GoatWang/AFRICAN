import math
import torch
from torch import nn
from collections import OrderedDict
from typing import Callable, Sequence, Tuple, Optional
from open_clip import Transformer, LayerNorm, LayerScale

class CrossResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()

        # for q to do self-attention
        self.ln_0 = norm_layer(d_model)
        self.attn_0 = nn.MultiheadAttention(d_model, n_head)
        self.ls_0 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # for q, k, v to do cross-attention
        self.ln_1_q = norm_layer(d_model)
        self.ln_1_kv = norm_layer(d_model)
        self.attn_1 = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # mlp
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            attn: nn.Module,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):

        # self-attention on q
        q_x = self.ln_0(q_x)
        q_x = q_x + self.ls_0(self.attention(attn=self.attn_0, q_x=q_x, k_x=q_x, v_x=q_x, attn_mask=attn_mask))

        # cross-attention on qkv
        q_x = self.ln_1_q(q_x)
        k_x = self.ln_1_kv(k_x)
        v_x = self.ln_1_kv(v_x)
        x = q_x + self.ls_1(self.attention(attn=self.attn_1, q_x=q_x, k_x=k_x, v_x=v_x, attn_mask=attn_mask))

        # mlp
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

class CrossTransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            CrossResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            q = r(q, k, v, attn_mask=attn_mask)
        return q

class VLCTransformer_OC(nn.Module):
    def __init__(
            self,
            seq_len: int = 32,
            width: int = 768,
            layers: int = 12,
            heads: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        # class embeddings and positional embeddings
        scale = width ** -0.5
        
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(seq_len, width))

        self.ln_pre_q = norm_layer(width)
        self.ln_pre_k = norm_layer(width)
        self.ln_pre_v = norm_layer(width)
        self.transformer = CrossTransformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.ln_post_q = norm_layer(width)
        self.ln_post_k = norm_layer(width)
        self.ln_post_v = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q + self.positional_embedding.to(q.dtype)
        q, k, v = self.ln_pre_q(q), self.ln_pre_k(k), self.ln_pre_v(v)        

        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)  # NLD -> LND
        q = self.transformer(q ,k ,v)
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)  # NLD -> LND

        q, k, v = self.ln_post_q(q), self.ln_post_k(k), self.ln_post_v(v)        

        B, F, W = q.shape
        tokens = (q.view(B*F, W) @ self.proj).view(B, F, W)
        return tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask=None):
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)
        
        Q = Q.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        dot_product = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.scale
        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, float("-inf"))
        attention = torch.nn.functional.softmax(dot_product, dim=-1)
        
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], self.d_model)
        return self.fc_out(out)    
    

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, forward_expansion, dropout):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.ReLU(),
            nn.Linear(forward_expansion * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        attention_out = self.attention(query, query, query, mask)
        query = self.norm0(query + self.dropout(attention_out))

        attention_out = self.attention(query, key, value, mask)
        x = self.norm1(query + self.dropout(attention_out))

        forward_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward_out))
        return x

class VLCTransformer(nn.Module):
    """Vision-Language Cross Transformer"""
    def __init__(self, d_model, num_heads, num_layers, forward_expansion, dropout):
        super(VLCTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.positional_enc = PositionalEncoding(d_model)
    
    def forward(self, query, key, value, mask=None):
        query = self.positional_enc(query)
        key = self.positional_enc(key)
        value = self.positional_enc(value)
        
        for layer in self.layers:
            query = layer(query, key, value, mask)
        return query

if __name__ == '__main__':
    vlc_transformer = VLCTransformer_OC(140, 768, 12, 12)
    q, k, v = torch.randn(1, 140, 768), torch.randn(1, 9, 768), torch.randn(1, 9, 768)
    q = vlc_transformer(q, k, v)
    print(q.shape)

    vlc_transformer = VLCTransformer(768, 12, 12, 4, 0.1)
    q, k, v = torch.randn(1, 140, 768), torch.randn(1, 9, 768), torch.randn(1, 9, 768)
    q = vlc_transformer(q, k, v)
    print(q.shape)
