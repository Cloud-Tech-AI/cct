import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNorm(nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, src):
        return self.norm(src)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dmodel: int, nheads: int):
        super().__init__()
        self.dmodel = dmodel
        self.nheads = nheads
        head_dim = dmodel // nheads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dmodel, 3 * dmodel)

    def forward(self, src):
        qkv = self.qkv(src).chunk(3, dim=-1)
        q, k, v = map(
            lambda x: x.reshape(*x.shape[:-1], self.nheads, -1).transpose(1, 2), qkv
        )
        # q => k => v => (batch, nheads, seq, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        # attn => (batch, nheads, seq, seq)
        attn_out = attn @ v
        # attn_out => (batch, nheads, seq, head_dim)
        out = attn_out.transpose(1, 2).reshape(*src.shape)
        # out => (batch, seq, dmodel)
        return out


class AddNorm(nn.Module):
    def __init__(self, dmodel: int):
        super().__init__()
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, src, residual):
        return self.norm(src + residual)


class FeedForward(nn.Module):
    def __init__(self, dmodel: int, ffn_scaling_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dmodel, dmodel * ffn_scaling_dim)
        self.fc2 = nn.Linear(dmodel * ffn_scaling_dim, dmodel)

    def forward(self, src):
        return self.fc2(F.gelu(self.fc1(src)))


class TransformerEncoder(nn.Module):
    def __init__(self, dmodel: int, nheads: int, ffn_scaling_dim: int):
        super().__init__()
        self.pre_norm = PreNorm(dmodel)
        self.attention = MultiHeadSelfAttention(dmodel, nheads)
        self.add_norm = AddNorm(dmodel)
        self.feed_forward = FeedForward(dmodel, ffn_scaling_dim)

    def forward(self, src):
        src = self.pre_norm(src)
        src = self.add_norm(self.attention(src), src)
        src = self.add_norm(self.feed_forward(src), src)
        return src

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
