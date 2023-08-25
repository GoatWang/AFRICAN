import torch
import torch.nn as nn
import math

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.ReLU(),
            nn.Linear(forward_expansion * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        attention_out = self.attention(query, key, value, mask)
        # Add & Norm
        x = self.norm1(query + self.dropout(attention_out))
        forward_out = self.feed_forward(x)
        # Add & Norm
        x = self.norm2(x + self.dropout(forward_out))
        return x

class VLCTransformer.py(nn.Module):
    """Vision-Language Cross Transformer"""
    def __init__(self, d_model, num_heads, num_layers, forward_expansion, dropout):
        super(VLCTransformer.py, self).__init__()
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
