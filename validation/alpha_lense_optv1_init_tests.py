#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from alpha_lense.optv1_alpha_model import AlphaLensePretrainModel,load_optv1_encoder
CK=Path('/var/lib/cots-lamcts/checkpoints/opt10_completion_v1/mlm_contrast_base_seed17.pt')
if not CK.is_file():raise FileNotFoundError(CK)
m=AlphaLensePretrainModel()
info=load_optv1_encoder(m,CK)
x=torch.zeros(2,64,8);mask=torch.ones(2,64,dtype=torch.bool);spec=torch.zeros(2,16)
with torch.no_grad():o=m(x,mask,spec)
assert o['surface_delta'].shape==(2,64,5)
assert o['merit'].shape==(2,) and o['value'].shape==(2,)
h=hashlib.sha256(CK.read_bytes()).hexdigest()
print(json.dumps({'status':'passed','checkpoint':str(CK),'sha256':h,'loaded_keys':info['loaded_keys'],'outputs':{k:list(v.shape) for k,v in o.items()}}))
