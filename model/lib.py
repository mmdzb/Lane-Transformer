from re import X
from turtle import forward
from fractions import gcd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size // 2)
        self.norm = nn.LayerNorm(output_size // 2)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(output_size // 2, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        return x

class Encoder_MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder_MLP, self).__init__()    
        self.linear1 = nn.Linear(input_size, hidden_size // 8)
        self.linear2 = nn.Linear(hidden_size // 8, hidden_size // 4)
        self.linear3 = nn.Linear(hidden_size // 4, hidden_size // 2)
        self.linear4 = nn.Linear(hidden_size // 2, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size // 8)
        self.norm2 = nn.LayerNorm(hidden_size // 4)
        self.norm3 = nn.LayerNorm(hidden_size // 2)
        self.norm4 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):    
        hidden_states = self.norm1(F.relu(self.linear1(hidden_states)))
        hidden_states = self.norm2(F.relu(self.linear2(hidden_states)))
        hidden_states = self.norm3(F.relu(self.linear3(hidden_states)))
        hidden_states = self.norm4(F.relu(self.linear4(hidden_states)))
        return hidden_states

class Goal_Decoder_MLP(nn.Module):
    def __init__(self,input_size, hidden_size, output_size=None):
        super(Goal_Decoder_MLP, self).__init__()
        if output_size is None:
            output_size = hidden_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states):
        hidden_states = F.relu(self.linear1(hidden_states))
        hidden_states = F.relu(self.linear2(hidden_states))
        hidden_states = self.linear3(hidden_states)
        return hidden_states

class Decoder_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None):
        super(Decoder_MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU(inplace=True)


        self.norm1 = nn.GroupNorm(gcd(32, hidden_size), hidden_size)
        self.norm2 = nn.GroupNorm(gcd(32, hidden_size), hidden_size)

        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        out += x

        out = self.relu(out)
        out = self.linear3(out)
        
        return out

class Smooth_Encoder(nn.Module):
    def __init__(self, args, hidden_size) -> None:
        super(Smooth_Encoder, self).__init__()
        cov_hidden_channel = hidden_size // 8

        if args.data_type == 'lane_gcn':
            input_dim = 3
        else:
            input_dim = 8

        self.cov1 = nn.Conv1d(input_dim, cov_hidden_channel, kernel_size=3, padding=1)
        self.cov2 = nn.Conv1d(cov_hidden_channel, 2 * cov_hidden_channel, kernel_size=3, padding=1)
        self.cov3 = nn.Conv1d(2 * cov_hidden_channel, 4 * cov_hidden_channel, kernel_size=3, padding=1)
        self.cov4 = nn.Conv1d(4 * cov_hidden_channel, 8 * cov_hidden_channel, kernel_size=3, padding=1)
        self.cov5 = nn.Conv1d(8 * cov_hidden_channel, 16 * cov_hidden_channel, kernel_size=3, padding=1)

        self.linear = nn.Linear(15 * cov_hidden_channel, hidden_size)

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(cov_hidden_channel)
        self.norm3 = nn.BatchNorm1d(2 * cov_hidden_channel)
        self.norm4 = nn.BatchNorm1d(4 * cov_hidden_channel)
        self.norm5 = nn.BatchNorm1d(8 * cov_hidden_channel)
        # self.norm7 = nn.BatchNorm1d(16 * cov_hidden_channel)
        self.norm6 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm1(x)

        time1 = time.time()
        temp1 = self.norm2(F.relu(self.cov1(x)))
        time2 = time.time()
        temp2 = self.norm3(F.relu(self.cov2(temp1)))
        time3 = time.time()
        temp3 = self.norm4(F.relu(self.cov3(temp2)))
        time4 = time.time()
        temp4 = self.norm5(F.relu(self.cov4(temp3)))
        time5 = time.time()
        # temp5 = self.norm7(F.relu(self.cov5(temp4)))
        # time6 = time.time()

        # x = torch.cat((temp1, temp2, temp3, temp4, temp5), dim=1)
        x = torch.cat((temp1, temp2, temp3, temp4), dim=1)
        # x = torch.cat((temp1, temp2, temp3), dim=1)
        # x = torch.cat((temp1, temp2), dim=1)

        x = self.linear(x.permute(0, 2, 1))

        # print(time2-time1, time3-time2, time4-time3, time5-time4, time6-time5)
        return self.norm6(x)

class map_smooth_decoder(nn.Module):
    def __init__(self):
        super(map_smooth_decoder, self).__init__()

        self.cov1 = nn.Conv1d(9, 5, kernel_size=3, padding=1)
        self.cov2 = nn.Conv1d(5, 1, kernel_size=3, padding=1)

        self.test1 = nn.Linear(9, 5)
        self.test2 = nn.Linear(5, 1)
        self.test_norm1 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.test_norm1(F.relu(self.test1(x)))
        x = self.test2(x)

        return x[:, :, 0]

class actor_smooth_decoder(nn.Module):
    def __init__(self):
        super(actor_smooth_decoder, self).__init__()

        self.test1 = nn.Linear(19, 9)
        self.test2 = nn.Linear(9, 1)
        self.test_norm1 = nn.BatchNorm1d(128)
        
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.test_norm1(F.relu(self.test1(x)))
        x = self.test2(x)

        return x[:, :, 0]

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(BottleNeck, self).__init__()
        self.hidden_channels = in_channels // 4

        self.cov1 = nn.Conv1d(in_channels, self.hidden_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(self.hidden_channels)
        self.cov2 = nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(self.hidden_channels)
        self.cov3 = nn.Conv1d(self.hidden_channels, in_channels, kernel_size=1)
        self.norm3 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        residual = x

        out = self.cov1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.cov2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.cov3(out)
        out = self.norm3(out)

        out = out + residual
        out = F.relu(out)

        out = out.permute(0, 2, 1)
        return out
