"""Shared-target causal next-session broker classifier; reuse SetEncoder/TCN."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from event_models import SetEncoder,CausalBlock
from event_broker_probe import shift_mapping

MODES=['own_only','relational','frozen_peers','shifted_peers']


class RelationModel(nn.Module):
    def __init__(self,vocabulary,context_features=18,width=24,mode='relational'):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode
        self.set_encoder=SetEncoder(vocabulary,8,width)
        self.target_identity=nn.Embedding(vocabulary+2,width,padding_idx=0)
        self.own=nn.Sequential(nn.Linear(14,width),nn.GELU(),nn.LayerNorm(width))
        self.fusion=nn.Linear(width*3,width)
        self.temporal=nn.Sequential(*[CausalBlock(width,d) for d in [1,2,4]])
        self.head=nn.Sequential(nn.Linear(width*2+context_features,width),nn.GELU(),nn.Linear(width,5))
        if mode in ['own_only','frozen_peers']:self.set_encoder.requires_grad_(False)

    def peer_representation(self,tokens,mask,ids,remove_peer=None):
        # tokens: queryday,target,history,peer,features
        days,targets,history,peers,features=tokens.shape
        if targets!=peers:raise ValueError('Target and selected-peer universes must match')
        diagonal=torch.eye(targets,device=mask.device,dtype=torch.bool)[None,:,None,:]
        safe=mask&~diagonal
        if remove_peer is not None:
            safe=safe.clone();safe[:,:,:,remove_peer]=False
        broker_ids=ids[None,None,None,:].expand(days,targets,history,peers)
        return self.set_encoder(tokens.reshape(days*targets,history,peers,features),
            broker_ids.reshape(days*targets,history,peers),safe.reshape(days*targets,history,peers)).reshape(days,targets,history,-1)

    def forward(self,own,context,tokens,mask,ids,remove_peer=None):
        days,targets,history,_=own.shape
        target=self.target_identity(ids)[None,:,None,:].expand(days,targets,history,-1)
        local=self.own(own)
        peer=torch.zeros_like(local) if self.mode=='own_only' else self.peer_representation(tokens,mask,ids,remove_peer)
        fusion=self.fusion(torch.cat([local,peer,target],dim=-1))
        temporal=self.temporal(fusion.reshape(days*targets,history,-1).transpose(1,2))[:,:,-1]
        price=context[:,None,:].expand(days,targets,-1).reshape(days*targets,-1)
        identity=self.target_identity(ids)[None,:,:].expand(days,targets,-1).reshape(days*targets,-1)
        return self.head(torch.cat([temporal,price,identity],dim=-1)).reshape(days,targets,5)


class RelationDataset(Dataset):
    def __init__(self,data,rows,mode,seed):
        self.data=data;self.rows=rows;self.mode=mode
        n,brokers=data['y'].shape
        self.mapping=np.tile(np.arange(n)[:,None],(1,brokers))
        if mode=='shifted_peers':
            for target in range(brokers):self.mapping[:,target]=shift_mapping(data['splits'],seed,target)[0]

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        i=int(self.rows[index]);source=self.mapping[i]
        return tuple(torch.from_numpy(np.array(value,copy=True)) for value in
            [self.data['own'][i],self.data['context'][i],self.data['tokens'][source],
             self.data['mask'][source],self.data['y'][i]])
