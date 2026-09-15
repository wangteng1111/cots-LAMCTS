"""Alpha Lense core.

Search-driven policy/value learning for constrained COTS optical design.
Network proposes; MCTS improves; physics decides; DesignSpec defines the lens.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence
import math, random
import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor
MARGIN_NAMES=("efl","f_number","t_stop","entrance_pupil","diameter","image_circle","bfd","total_length","distortion","illumination","price")

@dataclass(frozen=True)
class Candidate:
    parts: tuple[int,...]

@dataclass(frozen=True)
class DesignSpec:
    efl_target_mm: float
    efl_tol_mm: float
    max_f_number: float|None=None
    max_t_stop: float|None=None
    min_entrance_pupil_mm: float|None=None
    max_component_diameter_mm: float|None=None
    min_image_circle_mm: float|None=None
    min_bfd_mm: float|None=None
    max_total_length_mm: float|None=None
    max_distortion_pct: float|None=None
    min_relative_illumination: float|None=None
    max_price: float|None=None
    def vector(self)->Tensor:
        vals=(self.efl_target_mm,self.efl_tol_mm,self.max_f_number,self.max_t_stop,self.min_entrance_pupil_mm,self.max_component_diameter_mm,self.min_image_circle_mm,self.min_bfd_mm,self.max_total_length_mm,self.max_distortion_pct,self.min_relative_illumination,self.max_price)
        out=[]
        for v in vals: out.extend((0. if v is None else float(v),0. if v is None else 1.))
        return torch.tensor(out,dtype=torch.float32)

@dataclass
class EvalResult:
    J_quality: float
    metrics: dict[str,float]
    feasible: bool=True
    margins: dict[str,float]=field(default_factory=dict)
    violations: tuple[str,...]=()
    @property
    def quality_value(self): return -math.log(max(float(self.J_quality),1e-12))

class ConstraintEvaluator:
    def __call__(self,raw:EvalResult,spec:DesignSpec)->EvalResult:
        m=raw.metrics; d={}
        def upper(k,v,lim):
            if lim is not None and v is not None:d[k]=float(lim)-float(v)
        def lower(k,v,lim):
            if lim is not None and v is not None:d[k]=float(v)-float(lim)
        efl=m.get("efl_mm",m.get("efl"))
        if efl is not None:d["efl"]=spec.efl_tol_mm-abs(float(efl)-spec.efl_target_mm)
        upper("f_number",m.get("f_number",m.get("fno")),spec.max_f_number)
        upper("t_stop",m.get("t_stop"),spec.max_t_stop)
        lower("entrance_pupil",m.get("entrance_pupil_mm"),spec.min_entrance_pupil_mm)
        upper("diameter",m.get("max_component_diameter_mm"),spec.max_component_diameter_mm)
        lower("image_circle",m.get("image_circle_mm"),spec.min_image_circle_mm)
        lower("bfd",m.get("bfd_mm",m.get("bfd")),spec.min_bfd_mm)
        upper("total_length",m.get("total_length_mm"),spec.max_total_length_mm)
        upper("distortion",abs(m["distortion_pct"]) if "distortion_pct" in m else None,spec.max_distortion_pct)
        lower("illumination",m.get("min_relative_illumination",m.get("min_illum")),spec.min_relative_illumination)
        upper("price",m.get("price"),spec.max_price)
        bad=tuple(k for k,v in d.items() if v<0)
        return EvalResult(float(raw.J_quality),dict(m),not bad,d,bad)

def violation_score(r:EvalResult)->float:
    # Normalization belongs in adapters; this is only a monotonic fallback among infeasible states.
    return sum(max(0.,-float(v)) for v in r.margins.values())

def better(a:EvalResult,b:EvalResult|None)->bool:
    if b is None:return True
    if a.feasible!=b.feasible:return a.feasible
    return a.J_quality<b.J_quality if a.feasible else violation_score(a)<violation_score(b)

def tree_value(r:EvalResult)->float:
    # Feasible values occupy > -100. Infeasible values cannot outrank them.
    return r.quality_value if r.feasible else -1000.-violation_score(r)

class Catalog(Protocol):
    def legal_actions(self,state:Candidate,spec:DesignSpec)->Sequence[int]:...
    def apply(self,state:Candidate,action:int)->Candidate:...
    def action_features(self,state:Candidate,actions:Sequence[int],spec:DesignSpec)->Tensor:...

class AlphaLenseNet(nn.Module):
    def __init__(self,token_dim=8,d_model=64,nhead=4,layers=3,ff=192,action_dim=32,max_surfaces=64,spec_dim=24,margin_dim=len(MARGIN_NAMES)):
        super().__init__(); self.margin_dim=margin_dim
        self.proj=nn.Linear(token_dim,d_model);self.cls=nn.Parameter(torch.zeros(1,1,d_model));self.pos=nn.Parameter(torch.zeros(1,max_surfaces+1,d_model))
        enc=nn.TransformerEncoderLayer(d_model,nhead,ff,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc,layers);self.norm=nn.LayerNorm(d_model)
        self.spec_encoder=nn.Sequential(nn.Linear(spec_dim,64),nn.GELU(),nn.Linear(64,d_model))
        joint=2*d_model
        self.merit=nn.Sequential(nn.Linear(joint,64),nn.GELU(),nn.Linear(64,1));self.value=nn.Sequential(nn.Linear(joint,64),nn.GELU(),nn.Linear(64,1))
        self.adjust=nn.Linear(joint,action_dim);self.feasibility=nn.Linear(joint,1);self.margins=nn.Linear(joint,margin_dim);self.action_key=nn.Linear(action_dim,joint,bias=False)
    def encode(self,tokens,mask,spec):
        b,n,_=tokens.shape;x=self.proj(tokens);x=torch.cat((self.cls.expand(b,-1,-1),x),1)+self.pos[:,:n+1]
        pad=torch.cat((torch.zeros(b,1,dtype=torch.bool,device=mask.device),~mask.bool()),1)
        h=self.norm(self.encoder(x,src_key_padding_mask=pad)[:,0]);return torch.cat((h,self.spec_encoder(spec)),1)
    def forward(self,tokens,mask,spec,action_features=None):
        h=self.encode(tokens,mask,spec);o={"embedding":h,"merit":self.merit(h).squeeze(-1),"value":self.value(h).squeeze(-1),"adjustment":self.adjust(h),"feasibility_logit":self.feasibility(h).squeeze(-1),"margins":self.margins(h)}
        if action_features is not None:
            keys=self.action_key(action_features);o["policy_logits"]=torch.einsum("bd,bad->ba",h,keys)/math.sqrt(h.shape[-1])
        return o

@dataclass
class Node:
    state:Candidate
    prior:Tensor=field(default_factory=lambda:torch.empty(0));actions:Sequence[int]=()
    N:Tensor=field(default_factory=lambda:torch.empty(0));W:Tensor=field(default_factory=lambda:torch.empty(0));children:dict[int,"Node"]=field(default_factory=dict);expanded:bool=False
    @property
    def Q(self):return torch.where(self.N>0,self.W/self.N.clamp_min(1),torch.zeros_like(self.W))

@dataclass
class SearchResult:
    next_state:Candidate
    policy:Tensor
    best_state:Candidate|None
    best_eval:EvalResult|None
    physics_evals:int
    root:Node

class AlphaLenseMCTS:
    """PUCT where selected new leaves are grounded during search, so physics changes the visit policy."""
    def __init__(self,net,catalog:Catalog,tokenize:Callable,physics:Callable,spec:DesignSpec,c_puct=1.5,device="cuda"):
        self.net,self.catalog,self.tokenize,self.physics,self.spec=net,catalog,tokenize,physics,spec;self.c_puct,self.device=c_puct,device
        self.constraints=ConstraintEvaluator();self.cache:dict[Candidate,EvalResult]={}
    def evaluate(self,state):
        if state not in self.cache:self.cache[state]=self.constraints(self.physics(state),self.spec)
        return self.cache[state]
    @torch.no_grad()
    def expand(self,node:Node)->float:
        acts=list(self.catalog.legal_actions(node.state,self.spec));node.expanded=True
        if not acts:return -1000.
        tok,mask=self.tokenize(node.state);af=self.catalog.action_features(node.state,acts,self.spec);sv=self.spec.vector().to(self.device)[None]
        o=self.net(tok[None].to(self.device),mask[None].to(self.device),sv,af[None].to(self.device));node.actions=acts;node.prior=F.softmax(o["policy_logits"][0],-1).cpu();node.N=torch.zeros(len(acts));node.W=torch.zeros(len(acts))
        p=float(torch.sigmoid(o["feasibility_logit"])[0]);return float(o["value"][0])-5.*(1.-p)
    def search(self,root_state:Candidate,simulations=256,physics_budget=32,temperature=1.0)->SearchResult:
        root=Node(root_state);self.expand(root);best_state=None;best_eval=None;start_cache=len(self.cache)
        for sim in range(simulations):
            node=root;path=[]
            while node.expanded and node.actions:
                u=self.c_puct*node.prior*math.sqrt(float(node.N.sum())+1.)/(1.+node.N);i=int(torch.argmax(node.Q+u));path.append((node,i))
                if i not in node.children:node.children[i]=Node(self.catalog.apply(node.state,node.actions[i]))
                node=node.children[i]
                if not node.expanded:break
            predicted=self.expand(node)
            # Dense oracle is expensive: ground previously unseen leaves until budget is exhausted.
            if len(self.cache)-start_cache < physics_budget and node.state not in self.cache:
                r=self.evaluate(node.state);v=tree_value(r)
                if better(r,best_eval):best_state,best_eval=node.state,r
            else:v=predicted
            for p,i in reversed(path):p.N[i]+=1;p.W[i]+=v
        if not root.actions:return SearchResult(root_state,torch.empty(0),best_state,best_eval,len(self.cache)-start_cache,root)
        visits=root.N.pow(1./max(temperature,1e-6));pi=visits/visits.sum().clamp_min(1e-12)
        # Continue with best physics-grounded descendant when available; otherwise visit winner.
        nxt=best_state if best_state is not None else self.catalog.apply(root.state,root.actions[int(torch.argmax(root.N))])
        return SearchResult(nxt,pi,best_state,best_eval,len(self.cache)-start_cache,root)

@dataclass
class SearchTarget:
    state:Candidate;spec:DesignSpec;surface_tokens:Tensor;mask:Tensor;policy:Tensor;search_value:float;immediate_merit:float;feasible:float;margins:Tensor

class ReplayBuffer:
    def __init__(self,capacity=200000):self.capacity,self.data=capacity,[]
    def add(self,x):self.data.append(x);self.data[:]=self.data[-self.capacity:]
    def sample(self,n):return random.sample(self.data,min(n,len(self.data)))

def margin_tensor(r:EvalResult)->Tensor:return torch.tensor([float(r.margins.get(k,0.)) for k in MARGIN_NAMES],dtype=torch.float32)

def train_step(net,optimizer,batch:Sequence[SearchTarget],catalog:Catalog,device="cuda",policy_w=1.,value_w=1.,merit_w=.5,feas_w=.5,margin_w=.25):
    tokens=torch.stack([b.surface_tokens for b in batch]).to(device);masks=torch.stack([b.mask for b in batch]).to(device);specs=torch.stack([b.spec.vector() for b in batch]).to(device)
    acts=[list(catalog.legal_actions(b.state,b.spec)) for b in batch]
    if len({len(a) for a in acts})!=1:raise ValueError("bucket minibatches by legal-action count")
    af=torch.stack([catalog.action_features(b.state,a,b.spec) for b,a in zip(batch,acts)]).to(device);o=net(tokens,masks,specs,af)
    pi=torch.stack([b.policy for b in batch]).to(device);z=torch.tensor([b.search_value for b in batch],dtype=torch.float32,device=device);q=torch.tensor([b.immediate_merit for b in batch],dtype=torch.float32,device=device);f=torch.tensor([b.feasible for b in batch],dtype=torch.float32,device=device);m=torch.stack([b.margins for b in batch]).to(device)
    lpi=-(pi*F.log_softmax(o["policy_logits"],-1)).sum(-1).mean();lv=F.mse_loss(o["value"],z);lq=F.mse_loss(o["merit"],q);lf=F.binary_cross_entropy_with_logits(o["feasibility_logit"],f);lm=F.smooth_l1_loss(o["margins"],m)
    loss=policy_w*lpi+value_w*lv+merit_w*lq+feas_w*lf+margin_w*lm;optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return {k:float(v.detach()) for k,v in {"loss":loss,"policy":lpi,"value":lv,"merit":lq,"feasibility":lf,"margins":lm}.items()}
