import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib import BottleNeck

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, head_num=8, dropout=0.1) -> None:
        super(TransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, head_num, dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_size, head_num, dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, hidden_size)

    def forward(self, x_padding, x_mask, y_padding , y_mask):
        self_attn_output = self.self_attn(query=x_padding,
                                     key=x_padding, 
                                     value=x_padding,
                                     attn_mask=None,
                                     key_padding_mask=x_mask
                                     )[0]
        x_padding = x_padding + self.dropout1(self_attn_output)
        x_padding = self.norm1(x_padding)

        cross_attn_output = self.cross_attn(query=x_padding,
                                            key=y_padding,
                                            value=y_padding,
                                            attn_mask=None,
                                            key_padding_mask=y_mask
                                            )[0]
        x_padding = x_padding + self.dropout2(cross_attn_output)
        x_padding = self.norm2(x_padding)

        output = self.linear1(x_padding)
        output = F.relu(output)
        output = self.dropout3(output)
        output = self.linear2(output)

        x_padding = x_padding + self.dropout4(output)
        x_padding = self.norm3(x_padding)

        return x_padding