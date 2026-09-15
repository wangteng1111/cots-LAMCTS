"""Alpha Lense v0.1 — constraint-conditioned, physics-grounded search learning.

Network proposes; MCTS improves; physics decides; DesignSpec defines the lens.
Final preference is feasibility-first, never image-quality-only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence
import math, random
import torch
from torch import nn
import torch.nn.functional as F
Tensor = torch.Tensor

@dataclass(frozen=True)
class Candidate:
    parts: tuple[int, ...]

@dataclass(frozen=True)
class DesignSpec:
    """Requested lens. None means unconstrained for that field."""
    efl_target_mm: float
    efl_tol_mm: float
    max_f_number: float | None = None
    max_t_stop: float | None = None
    min_entrance_pupil_mm: float | None = None
    max_component_diameter_mm: float | None = None
    min_image_circle_mm: float | None = None
    min_bfd_mm: float | None = None
    max_total_length_mm: float | None = None
    max_distortion_pct: float | None = None
    min_relative_illumination: float | None = None
    max_price: float | None = None

    def vector(self) -> Tensor:
        vals = [self.efl_target_mm, self.efl_tol_mm, self.max_f_number,
                self.max_t_stop, self.min_entrance_pupil_mm,
                self.max_component_diameter_mm, self.min_image_circle_mm,
                self.min_bfd_mm, self.max_total_length_mm,
                self.max_distortion_pct, self.min_relative_illumination,
                self.max_price]
        # value + presence bit avoids confusing unspecified with zero.
        out=[]
        for v in vals: out += [0.0 if v is None else float(v), 0.0 if v is None else 1.0]
        return torch.tensor(out, dtype=torch.float32)

@dataclass
class EvalResult:
    J_quality: float
    metrics: dict[str, float]
    feasible: bool = True
    margins: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()

    @property
    def quality_value(self) -> float:
        return -math.log(max(self.J_quality, 1e-12))

@dataclass
class SearchTarget:
    state: Candidate
    spec: DesignSpec
    surface_tokens: Tensor
    mask: Tensor
    policy: Tensor
    search_value: float
    immediate_merit: float
    feasible: float
    margins: Tensor

class Catalog(Protocol):
    def legal_actions(self, state: Candidate, spec: DesignSpec) -> Sequence[int]: ...
    def apply(self, state: Candidate, action: int) -> Candidate: ...
    def action_features(self, state: Candidate, actions: Sequence[int], spec: DesignSpec) -> Tensor: ...

class ConstraintEvaluator:
    """Converts authoritative physics metrics into explicit feasibility/margins."""
    def __call__(self, raw: EvalResult, spec: DesignSpec) -> EvalResult:
        m, margins = raw.metrics, {}
        def upper(name, value, limit):
            if limit is not None and value is not None: margins[name] = float(limit-value)
        def lower(name, value, limit):
            if limit is not None and value is not None: margins[name] = float(value-limit)
        efl=m.get("efl_mm", m.get("efl"))
        if efl is not None: margins["efl"] = spec.efl_tol_mm-abs(float(efl)-spec.efl_target_mm)
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
        violations=tuple(k for k,v in margins.items() if v < 0)
        return EvalResult(raw.J_quality,m,not violations,margins,violations)

class AlphaLenseNet(nn.Module):
    """Prescription + DesignSpec encoder; policy/value/quality/adjustment/feasibility heads."""
    def __init__(self, token_dim=8, d_model=64, nhead=4, layers=3, ff=192,
                 action_dim=32, max_surfaces=64, spec_dim=24, margin_dim=11):
        super().__init__(); self.margin_dim=margin_dim
        self.proj=nn.Linear(token_dim,d_model); self.cls=nn.Parameter(torch.zeros(1,1,d_model))
        self.pos=nn.Parameter(torch.zeros(1,max_surfaces+1,d_model))
        enc=nn.TransformerEncoderLayer(d_model,nhead,ff,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc,layers); self.norm=nn.LayerNorm(d_model)
        self.spec_encoder=nn.Sequential(nn.Linear(spec_dim,64),nn.GELU(),nn.Linear(64,d_model))
        joint=2*d_model
        self.merit=nn.Sequential(nn.Linear(joint,64),nn.GELU(),nn.Linear(64,1))
        self.value=nn.Sequential(nn.Linear(joint,64),nn.GELU(),nn.Linear(64,1))
        self.adjust=nn.Linear(joint,action_dim)
        self.feasibility=nn.Linear(joint,1); self.margins=nn.Linear(joint,margin_dim)
        self.action_key=nn.Linear(action_dim,joint,bias=False)

    def encode(self,tokens,mask,spec):
        b,n,_=tokens.shape; x=self.proj(tokens); cls=self.cls.expand(b,-1,-1)
        x=torch.cat([cls,x],1)+self.pos[:,:n+1]
        pad=torch.cat([torch.zeros(b,1,dtype=torch.bool,device=mask.device),~mask.bool()],1)
        h=self.norm(self.encoder(x,src_key_padding_mask=pad)[:,0]); c=self.spec_encoder(spec)
        return torch.cat([h,c],-1)

    def forward(self,tokens,mask,spec,action_features=None,action_mask=None):
        h=self.encode(tokens,mask,spec)
        out={"embedding":h,"merit":self.merit(h).squeeze(-1),"value":self.value(h).squeeze(-1),
             "adjustment":self.adjust(h),"feasibility_logit":self.feasibility(h).squeeze(-1),
             "margins":self.margins(h)}
        if action_features is not None:
            keys=self.action_key(action_features); logits=torch.einsum("bd,bad->ba",h,keys)/math.sqrt(h.shape[-1])
            if action_mask is not None: logits=logits.masked_fill(~action_mask.bool(),-torch.inf)
            out["policy_logits"]=logits
        return out

@dataclass
class Node:
    state: Candidate; prior: Tensor=field(default_factory=lambda:torch.empty(0)); actions: Sequence[int]=()
    N: Tensor=field(default_factory=lambda:torch.empty(0)); W: Tensor=field(default_factory=lambda:torch.empty(0))
    children: dict[int,"Node"]=field(default_factory=dict); expanded: bool=False
    @property
    def Q(self): return torch.where(self.N>0,self.W/self.N.clamp_min(1),torch.zeros_like(self.W))

class AlphaLenseMCTS:
    """Constraint-aware PUCT with selective authoritative physics grounding."""
    def __init__(self,net,catalog:Catalog,tokenize,physics,spec:DesignSpec,c_puct=1.5,device="cuda"):
        self.net,self.catalog,self.tokenize,self.physics,self.spec=net,catalog,tokenize,physics,spec
        self.constraints=ConstraintEvaluator(); self.c_puct,self.device=c_puct,device; self.cache={}

    def evaluate(self,state):
        if state not in self.cache: self.cache[state]=self.constraints(self.physics(state),self.spec)
        return self.cache[state]

    @torch.no_grad()
    def expand(self,node):
        actions=list(self.catalog.legal_actions(node.state,self.spec))
        if not actions: node.expanded=True; return -1e6
        tok,mask=self.tokenize(node.state); af=self.catalog.action_features(node.state,actions,self.spec)
        sv=self.spec.vector().to(self.device)[None]
        out=self.net(tok[None].to(self.device),mask[None].to(self.device),sv,af[None].to(self.device))
        node.actions=actions; node.prior=F.softmax(out["policy_logits"][0],-1).cpu()
        node.N=torch.zeros(len(actions)); node.W=torch.zeros(len(actions)); node.children={}; node.expanded=True
        # NN value is spec-conditioned; predicted feasibility biases cheap leaf evaluation.
        pfeas=torch.sigmoid(out["feasibility_logit"])[0].item()
        return float(out["value"][0].item())-5.0*(1.0-pfeas)

    @staticmethod
    def backed_value(r:EvalResult):
        # Feasibility-first scalar encoding for tree mechanics only. Final comparison remains lexicographic.
        if r.feasible: return r.quality_value
        severity=sum(max(0.0,-v) for v in r.margins.values())
        return -1000.0-severity

    @staticmethod
    def better(a:EvalResult,b:EvalResult|None):
        if b is None:return True
        if a.feasible!=b.feasible:return a.feasible
        if a.feasible:return a.J_quality<b.J_quality
        sa=sum(max(0.,-v) for v in a.margins.values()); sb=sum(max(0.,-v) for v in b.margins.values())
        return sa<sb

    def search(self,root_state,simulations=256,physics_topk=8,temperature=1.0):
        root=Node(root_state); self.expand(root)
        for _ in range(simulations):
            node,path=root,[]
            while node.expanded and len(node.actions):
                score=node.Q+self.c_puct*node.prior*math.sqrt(float(node.N.sum())+1)/(1+node.N)
                i=int(torch.argmax(score)); path.append((node,i))
                if i not in node.children: node.children[i]=Node(self.catalog.apply(node.state,node.actions[i]))
                node=node.children[i]
                if not node.expanded: break
            v=self.expand(node)
            for p,i in reversed(path): p.N[i]+=1;p.W[i]+=v
        order=torch.argsort(root.N,descending=True)[:min(physics_topk,len(root.actions))]
        best_result=None;best_idx=None
        for idx in order.tolist():
            s2=self.catalog.apply(root.state,root.actions[idx]); r=self.evaluate(s2); z=self.backed_value(r)
            root.N[idx]+=1;root.W[idx]+=z
            if self.better(r,best_result):best_result,best_idx=r,idx
        visits=root.N.pow(1/max(temperature,1e-6)); pi=visits/visits.sum().clamp_min(1e-12)
        # Choose among physics-grounded candidates feasibility-first, not merely highest NN visit count.
        if best_idx is None: best_idx=int(torch.argmax(root.N))
        return self.catalog.apply(root.state,root.actions[best_idx]),pi,root,best_result

class ReplayBuffer:
    def __init__(self,capacity=200_000):self.capacity,self.data=capacity,[]
    def add(self,x):self.data.append(x);self.data[:]=self.data[-self.capacity:]
    def sample(self,n):return random.sample(self.data,min(n,len(self.data)))

def train_step(net,optimizer,batch:Sequence[SearchTarget],catalog:Catalog,device="cuda",
               policy_w=1.,value_w=1.,merit_w=.5,feas_w=.5,margin_w=.25):
    tokens=torch.stack([b.surface_tokens for b in batch]).to(device); masks=torch.stack([b.mask for b in batch]).to(device)
    specs=torch.stack([b.spec.vector() for b in batch]).to(device)
    actions=[list(catalog.legal_actions(b.state,b.spec)) for b in batch]
    if len({len(a) for a in actions})!=1: raise ValueError("bucket minibatches by legal-action count")
    af=torch.stack([catalog.action_features(b.state,a,b.spec) for b,a in zip(batch,actions)]).to(device)
    out=net(tokens,masks,specs,af); pi=torch.stack([b.policy for b in batch]).to(device)
    z=torch.tensor([b.search_value for b in batch],device=device);q=torch.tensor([b.immediate_merit for b in batch],device=device)
    feas=torch.tensor([b.feasible for b in batch],device=device); margins=torch.stack([b.margins for b in batch]).to(device)
    lp=F.log_softmax(out["policy_logits"],-1); lpi=-(pi*lp).sum(-1).mean()
    lv=F.mse_loss(out["value"],z);lq=F.mse_loss(out["merit"],q)
    lf=F.binary_cross_entropy_with_logits(out["feasibility_logit"],feas);lm=F.smooth_l1_loss(out["margins"],margins)
    loss=policy_w*lpi+value_w*lv+merit_w*lq+feas_w*lf+margin_w*lm
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return {k:float(v.detach()) for k,v in {"loss":loss,"policy":lpi,"value":lv,"merit":lq,"feasibility":lf,"margins":lm}.items()}

# Integration contract
# - tokenize() reuses OPT physical surface tokens; IDs are never semantic inputs.
# - DesignSpec conditions policy/value and defines feasibility.
# - Catalog.legal_actions() enforces cheap hard geometry/procurement constraints before expansion.
# - physics() is authoritative Q4096/production evaluator and must return explicit design metrics.
# - Final ranking is lexicographic: feasible > infeasible; among feasible minimize J_quality.
# - MCTS visits train policy; best feasible physics-grounded descendants train value.
