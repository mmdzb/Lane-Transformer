import torch
import torch.nn as nn
from torch.nn import functional as F
from fractions import gcd

class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.linear3 = nn.Linear(n_out, n_out)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
            self.norm3 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.norm3(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out
    
class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')
        
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out

class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts, agt_ctrs, dest_ctrs):
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)
        bs = dest_ctrs.shape[0]
        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(bs*num_mods, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(bs*num_mods, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class Pred(nn.Module):
    def __init__(self, config):
        super(Pred, self).__init__()
        self.config = config
        norm = "BN"
        ng = 1

        n_actor = 128

        pred = []
        for i in range(6):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * 30),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors, actor_idcs, actor_ctrs):
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), 30, 2)
        bs = reg.shape[0]
        num_mod = reg.shape[1]
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs
        last = reg.shape[2]-1
        dest_ctrs = reg[:, :, last].detach()
        feats = self.att_dest(actors, torch.vstack(actor_ctrs), dest_ctrs)
        cls = self.cls(feats).view(bs, 6)

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(sort_idcs.shape[0]).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(bs, 1).repeat(1, sort_idcs.size(1)).view(bs*num_mod)
        sort_idcs = sort_idcs.view(sort_idcs.shape[0]*sort_idcs.shape[1])
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), 30, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            out["cls"].append(cls[i])
            out["reg"].append(reg[i])
        return out