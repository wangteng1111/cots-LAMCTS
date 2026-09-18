"""OPTv1-initialized Alpha Lense pretraining model."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

MAXLEN=64
FIELDS=8
D_MODEL=64
SPEC_DIM=16

class OpticalEncoderV1(nn.Module):
    """Exact encoder topology used by OPTv1 mlm_contrast_base."""
    def __init__(self,d=64,heads=4,layers=3,ff=128):
        super().__init__();self.d=d
        self.proj=nn.Sequential(nn.Linear(FIELDS,d),nn.LayerNorm(d),nn.GELU())
        self.mask_token=nn.Parameter(torch.randn(1,1,d)*.02)
        self.cls=nn.Parameter(torch.randn(1,1,d)*.02)
        self.pos=nn.Parameter(torch.randn(1,MAXLEN+1,d)*.02)
        el=nn.TransformerEncoderLayer(d,heads,ff,dropout=.08,activation='gelu',batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(el,layers,norm=nn.LayerNorm(d))
    def encode(self,x,valid,maskpos=None):
        h=self.proj(x)
        if maskpos is not None:h=torch.where(maskpos.unsqueeze(-1),self.mask_token.expand(h.shape[0],h.shape[1],-1),h)
        h=torch.cat([self.cls.expand(x.shape[0],-1,-1),h],1)+self.pos[:,:x.shape[1]+1]
        pm=torch.cat([torch.zeros((x.shape[0],1),dtype=torch.bool,device=x.device),~valid.bool()],1)
        return self.enc(h,src_key_padding_mask=pm)

class AlphaLensePretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical=OpticalEncoderV1()
        self.spec_encoder=nn.Sequential(nn.Linear(SPEC_DIM,64),nn.GELU(),nn.Linear(64,64),nn.LayerNorm(64))
        self.merit=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        self.value=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        self.feasibility=nn.Linear(128,1)
        self.reachable_feasibility=nn.Linear(128,1)
        self.violation=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
        self.reachable_violation=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
        self.topology_count=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
        self.surface_delta=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,5))
    def forward(self,tokens,valid,spec):
        h=self.optical.encode(tokens,valid)
        sp=self.spec_encoder(spec)
        joint=torch.cat([h[:,0],sp],-1)
        sph=sp[:,None,:].expand(-1,tokens.shape[1],-1)
        per=torch.cat([h[:,1:tokens.shape[1]+1],sph],-1)
        return {
          'merit':self.merit(joint).squeeze(-1),
          'value':self.value(joint).squeeze(-1),
          'feasibility_logit':self.feasibility(joint).squeeze(-1),
          'reachable_feasibility_logit':self.reachable_feasibility(joint).squeeze(-1),
          'violation':self.violation(joint).squeeze(-1),
          'reachable_violation':self.reachable_violation(joint).squeeze(-1),
          'topology_count':self.topology_count(joint).squeeze(-1),
          'surface_delta':self.surface_delta(per),
          'embedding':joint,
        }

def load_optv1_encoder(model:AlphaLensePretrainModel,checkpoint:str|Path)->dict:
    checkpoint=Path(checkpoint)
    sd=torch.load(checkpoint,map_location='cpu',weights_only=True)
    target=model.optical.state_dict();sub={k:v for k,v in sd.items() if k in target}
    missing=sorted(set(target)-set(sub));shape_bad=[k for k,v in sub.items() if tuple(v.shape)!=tuple(target[k].shape)]
    if missing or shape_bad:raise RuntimeError(f'OPTv1 encoder mismatch missing={missing} shape_bad={shape_bad}')
    model.optical.load_state_dict(sub,strict=True)
    return {'checkpoint':str(checkpoint),'loaded_keys':len(sub)}
