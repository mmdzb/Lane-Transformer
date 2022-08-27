import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib import MLP, Smooth_Encoder, actor_smooth_decoder, map_smooth_decoder

class ActorSubNet(nn.Module):
    def __init__(self, args, hidden_size, depth=None):
        super(ActorSubNet, self).__init__()
        if depth is None:
            depth = 2

        self.Attn = nn.ModuleList([nn.MultiheadAttention(hidden_size, 8, dropout=0.1) for _ in range(depth)])
        self.Norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(depth)])
        self.smooth_traj = Smooth_Encoder(args, hidden_size)

        self.final_layer = actor_smooth_decoder()

    def forward(self, inputs, inputs_mask, polyline_mask, device):
        hidden_states_batch = inputs
        hidden_states_mask = inputs_mask

        smooth_input = hidden_states_batch[polyline_mask]
        smooth_output = self.smooth_traj(smooth_input)

        hidden_states_batch = torch.zeros((inputs.shape[0], inputs.shape[1], smooth_output.shape[2]), dtype=torch.float32).to(device)
        hidden_states_batch[polyline_mask] = smooth_output
        
        for layer_index, layer in enumerate(self.Attn):
            temp = hidden_states_batch
            q = k = v = hidden_states_batch.permute(1,0,2)
            hidden_states_batch = layer(q, k, value=v, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)  
            hidden_states_batch = hidden_states_batch + temp
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)

        hidden_states_batch = self.final_layer(hidden_states_batch)
        return hidden_states_batch


class MapSubNet(nn.Module):

    def __init__(self, args, hidden_size, depth=None):
        super(MapSubNet, self).__init__()
        if depth is None:
            depth = 2

        input_dim = 8

        self.MLPs = nn.ModuleList([MLP(input_dim, hidden_size // 8), MLP(hidden_size // 4, hidden_size // 2)])
        self.Attn = nn.ModuleList([nn.MultiheadAttention(hidden_size // 8, 8, dropout=0.1), nn.MultiheadAttention(hidden_size // 2, 8, dropout=0.1)])
        self.Norms = nn.ModuleList([nn.LayerNorm(hidden_size // 4), nn.LayerNorm(hidden_size)])

        self.final_layer = map_smooth_decoder()

    def forward(self, inputs, inputs_mask, device):
        hidden_states_batch = inputs
        hidden_states_mask = inputs_mask

        for layer_index, layer in enumerate(self.Attn):
            hidden_states_batch = self.MLPs[layer_index](hidden_states_batch)
            temp = hidden_states_batch 
            q = k = v = hidden_states_batch.permute(1,0,2)
            hidden_states_batch = layer(q, k, value=v, attn_mask=None, key_padding_mask=hidden_states_mask)[0].permute(1,0,2)
            hidden_states_batch = torch.cat([hidden_states_batch, temp], dim=2)
            hidden_states_batch = self.Norms[layer_index](hidden_states_batch)
            hidden_states_batch = F.relu(hidden_states_batch)
        
        hidden_states_batch = self.final_layer(hidden_states_batch)
        return hidden_states_batch

