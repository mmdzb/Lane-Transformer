from turtle import forward
from hypothesis import target
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
from model.SubNet import ActorSubNet, MapSubNet
from model.lib import Encoder_MLP, Decoder_MLP, Goal_Decoder_MLP, Smooth_Encoder
from model.TransformerDecoder import TransformerDecoder
from model.Decoder import Pred

class LaneTransformer(nn.Module):
    def __init__(self, args):
        super(LaneTransformer, self).__init__()
        self.args = args
        hidden_size = args.hidden_size

        self.actor_net = ActorSubNet(args, hidden_size)
        self.map_net = MapSubNet(args, hidden_size)

        self.global_graph = nn.MultiheadAttention(hidden_size,8,dropout=0.1)   

        self.A2L = TransformerDecoder(hidden_size)
        self.L2A = TransformerDecoder(hidden_size)

        self.A2L_again = TransformerDecoder(hidden_size)
        self.L2A_again = TransformerDecoder(hidden_size)

        self.decoder = Pred(args)

    def forward_encode_sub_graph(self, mapping, device):
        batch_size = len(mapping['actor_polyline_input_length'])
        actor_max_polyline_num, actor_max_vec_num, actor_vec_len = mapping['actor_max_polyline_num'], mapping['actor_max_vec_num'], mapping['actor_vec_len']
        map_max_polyline_num, map_max_vec_num, map_vec_len = mapping['map_max_polyline_num'], mapping['map_max_vec_num'], mapping['map_vec_len']
        hidden_size = 128

        actor_input, actor_input_mask = torch.flatten(mapping['actor_total_input_padding'], start_dim=0, end_dim=1).to(device), mapping['actor_input_mask'].to(device)
        map_input, map_input_mask = torch.flatten(mapping['map_total_input_padding'], start_dim=0, end_dim=1).to(device), mapping['map_input_mask'].to(device)
        actor_valid_polyline = mapping['valid_actor_polyline']


        actor_states_batch = self.actor_net(actor_input, actor_input_mask, actor_valid_polyline, device)
        map_states_batch = self.map_net(map_input, map_input_mask, device)
    
        actor_polyline_padding , actor_polyline_mask = actor_states_batch.view(batch_size, actor_max_polyline_num, hidden_size), mapping['actor_polyline_mask'].to(device)
        map_polyline_padding, map_polyline_mask = map_states_batch.view(batch_size, map_max_polyline_num, hidden_size), mapping['map_polyline_mask'].to(device)

        lanes = map_polyline_padding.permute(1, 0, 2)
        lanes_mask = map_polyline_mask
        agents = actor_polyline_padding.permute(1, 0, 2)
        agents_mask = actor_polyline_mask
        
        lanes = lanes + self.A2L(lanes, lanes_mask, agents, agents_mask)
        agents = agents + self.L2A(agents, agents_mask, lanes, lanes_mask)

        lanes = lanes + self.A2L_again(lanes, lanes_mask, agents, agents_mask)
        agents = agents + self.L2A_again(agents, agents_mask, lanes, lanes_mask)

 
        return agents.permute(1, 0, 2)

    def forward(self, mapping, device):

        agent_states_batch = self.forward_encode_sub_graph(mapping, device)

        inputs = agent_states_batch.permute(1, 0, 2)
        inputs_mask = mapping['global_graph_mask'].to(device)
        hidden_states = self.global_graph(query=inputs, 
                                    key=inputs,
                                    value=inputs,
                                    key_padding_mask=inputs_mask
                                    )[0]
        hidden_states = hidden_states.permute(1, 0, 2)

             
        a, b = [], []
        for i in range(inputs_mask.shape[0]):
            a.append(torch.tensor(i).to(device))
            b.append(torch.tensor([0, 0],dtype=torch.float32).to(device))
        out = self.decoder(hidden_states[:, 0, :], a, b)

        outputs = torch.vstack(out['reg']).view(inputs_mask.shape[0], 6, 30, 2).permute(1, 0, 2, 3)
        score = F.softmax(torch.vstack(out['cls']), dim=1)

        return outputs, score
