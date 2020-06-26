import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries, keys, values):
        attn = torch.bmm(queries, keys.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        result = torch.bmm(attn, values)

        return result, attn
