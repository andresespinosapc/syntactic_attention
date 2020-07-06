import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, attention_activation=None):
        super().__init__()
        if attention_activation is None:
            self.attention_activation = lambda attn, mask, queries: F.softmax(attn, dim=-1)
        else:
            self.attention_activation = attention_activation

    def forward(self, queries, keys, values):
        attn = torch.bmm(queries, keys.transpose(1, 2))
        attn = self.attention_activation(
            attn,
            torch.zeros_like(attn, dtype=torch.bool).to(device),
            queries,
        )
        result = torch.bmm(attn, values)

        return result, attn
