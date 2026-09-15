"""Alpha Lense v0.1 reverse-tree policy pretraining.

Build bounded stochastic perturbation neighborhoods around known-good optical
systems, let authoritative physics relabel which moves are improvements, and
turn those improving moves into soft policy targets.  This module is adapter-
level: production prescription mutation and Q4096 evaluation are injected.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, Hashable
import math, random
import torch
import torch.nn.functional as F
from .core import DesignSpec, EvalResult, ConstraintEvaluator, MARGIN_NAMES

Tensor=torch.Tensor

@dataclass(frozen=True)
class PerturbationAction:
    """Physical mutation descriptor; multiple parameters may move together."""
    kind:str
    target:tuple[int,...]
    deltas:tuple[float,...]
    feature:tuple[float,...]
    metadata:tuple[tuple[str,str],...]=()

@dataclass(frozen=True)
class PretrainState:
    """Opaque identity plus optional ancestry; optical semantics live in tokenizer/actions."""
    key:Hashable
    depth:int=0

class PerturbationAdapter(Protocol):
    def propose(self,state:PretrainState,spec:DesignSpec,n:int,rng:random.Random)->Sequence[PerturbationAction]:...
    def apply(self,state:PretrainState,action:PerturbationAction)->PretrainState:...
    def tokenize(self,state:PretrainState)->tuple[Tensor,Tensor]:...
    def action_features(self,state:PretrainState,actions:Sequence[PerturbationAction],spec:DesignSpec)->Tensor:...

@dataclass
class RankedEval:
    state:PretrainState
    result:EvalResult
    normalized_violation:float

@dataclass
class PolicyPretrainExample:
    state:PretrainState
    spec:DesignSpec
    surface_tokens:Tensor
    mask:Tensor
    action_features:Tensor
    policy:Tensor
    immediate_merit:float
    feasible:float
    margins:Tensor
    reachable_value:float
    improving:int
    evaluated:int

@dataclass
class TreeBuildStats:
    seed:Hashable
    neighborhoods:int=0
    physics_evals:int=0
    policy_examples:int=0
    improving_edges:int=0
    frontier_sizes:list[int]=field(default_factory=list)

class NormalizedConstraintEvaluator:
    """Constraint evaluation with explicit missing-metric failure and dimensionless margins."""
    def __init__(self,scales:dict[str,float]|None=None):
        self.base=ConstraintEvaluator()
        self.scales=dict(scales or {})
    @staticmethod
    def required(spec:DesignSpec)->dict[str,bool]:
        return {
            "efl":True,
            "f_number":spec.max_f_number is not None,
            "t_stop":spec.max_t_stop is not None,
            "entrance_pupil":spec.min_entrance_pupil_mm is not None,
            "diameter":spec.max_component_diameter_mm is not None,
            "image_circle":spec.min_image_circle_mm is not None,
            "bfd":spec.min_bfd_mm is not None,
            "total_length":spec.max_total_length_mm is not None,
            "distortion":spec.max_distortion_pct is not None,
            "illumination":spec.min_relative_illumination is not None,
            "price":spec.max_price is not None,
        }
    def scale_map(self,spec:DesignSpec)->dict[str,float]:
        s={
            "efl":max(abs(spec.efl_tol_mm),1e-6),
            "f_number":max(abs(spec.max_f_number or 1.),1e-6),
            "t_stop":max(abs(spec.max_t_stop or 1.),1e-6),
            "entrance_pupil":max(abs(spec.min_entrance_pupil_mm or 1.),1e-6),
            "diameter":max(abs(spec.max_component_diameter_mm or 1.),1e-6),
            "image_circle":max(abs(spec.min_image_circle_mm or 1.),1e-6),
            "bfd":max(abs(spec.min_bfd_mm or 1.),1e-6),
            "total_length":max(abs(spec.max_total_length_mm or 1.),1e-6),
            "distortion":max(abs(spec.max_distortion_pct or 1.),1e-6),
            "illumination":max(abs(spec.min_relative_illumination or 1.),1e-6),
            "price":max(abs(spec.max_price or 1.),1e-6),
        };s.update({k:max(abs(float(v)),1e-6) for k,v in self.scales.items()});return s
    def __call__(self,raw:EvalResult,spec:DesignSpec)->RankedEval:
        r=self.base(raw,spec);req=self.required(spec);missing=[k for k,needed in req.items() if needed and k not in r.margins]
        scales=self.scale_map(spec);nm={k:float(v)/scales[k] for k,v in r.margins.items()}
        # Missing required physics is invalid, never silently feasible.
        if missing:
            nm.update({k:-1e6 for k in missing});bad=tuple(dict.fromkeys((*r.violations,*missing)))
            r=EvalResult(r.J_quality,r.metrics,False,nm,bad)
        else:
            bad=tuple(k for k,v in nm.items() if v<0);r=EvalResult(r.J_quality,r.metrics,not bad,nm,bad)
        sev=sum(max(0.,-v) for v in r.margins.values())
        return RankedEval(PretrainState("__unset__"),r,sev)

def ordered_score(r:EvalResult)->tuple[int,float,float]:
    """Higher tuple is better; feasibility dominates quality."""
    violation=sum(max(0.,-float(v)) for v in r.margins.values())
    if r.feasible:return (1,0.,-float(r.J_quality))
    return (0,-violation,-float(r.J_quality))

def scalar_utility(r:EvalResult)->float:
    """Dimensionless utility preserving feasible > infeasible for policy softmax."""
    violation=sum(max(0.,-float(v)) for v in r.margins.values())
    if r.feasible:return 100.-math.log(max(float(r.J_quality),1e-12))
    return -violation

def is_better(a:EvalResult,b:EvalResult)->bool:return ordered_score(a)>ordered_score(b)

def soft_improvement_policy(parent:EvalResult,children:Sequence[EvalResult],temperature=.25)->tuple[Tensor,list[int]]:
    """Policy mass only on physics-supported improvements; no false target otherwise."""
    good=[i for i,r in enumerate(children) if is_better(r,parent)]
    pi=torch.zeros(len(children),dtype=torch.float32)
    if not good:return pi,good
    base=scalar_utility(parent);u=torch.tensor([scalar_utility(children[i])-base for i in good],dtype=torch.float32)
    p=F.softmax(u/max(float(temperature),1e-6),dim=0)
    pi[torch.tensor(good,dtype=torch.long)]=p
    return pi,good

def margins_tensor(r:EvalResult)->Tensor:return torch.tensor([float(r.margins.get(k,-1e6)) for k in MARGIN_NAMES],dtype=torch.float32)

class ReverseTreeBuilder:
    """Bounded stochastic degradation tree with authoritative physics relabeling."""
    def __init__(self,adapter:PerturbationAdapter,physics:Callable[[PretrainState],EvalResult],constraint_scales:dict[str,float]|None=None):
        self.adapter=adapter;self.physics=physics;self.constraints=NormalizedConstraintEvaluator(constraint_scales);self.cache={}
    def evaluate(self,state:PretrainState,spec:DesignSpec)->EvalResult:
        ck=(state.key,spec)
        if ck not in self.cache:
            rr=self.constraints(self.physics(state),spec);rr.state=state;self.cache[ck]=rr.result
        return self.cache[ck]
    def neighborhood(self,parent:PretrainState,spec:DesignSpec,proposals:int,rng:random.Random,temperature:float)->tuple[PolicyPretrainExample|None,list[tuple[PretrainState,float]]]:
        actions=list(self.adapter.propose(parent,spec,proposals,rng));children=[self.adapter.apply(parent,a) for a in actions]
        if not actions:return None,[]
        pr=self.evaluate(parent,spec);cr=[self.evaluate(s,spec) for s in children];pi,good=soft_improvement_policy(pr,cr,temperature)
        # Frontier weights favor physically better states, not ancestry distance.
        scored=[(s,scalar_utility(r)) for s,r in zip(children,cr)]
        if not good:return None,scored
        tok,mask=self.adapter.tokenize(parent);af=self.adapter.action_features(parent,actions,spec)
        best=max((cr[i] for i in good),key=ordered_score)
        ex=PolicyPretrainExample(parent,spec,tok,mask,af,pi,scalar_utility(pr),float(pr.feasible),margins_tensor(pr),scalar_utility(best),len(good),len(actions))
        return ex,scored
    def build(self,seed:PretrainState,spec:DesignSpec,depth=10,proposals=100,frontier_width=256,parents_per_depth=64,temperature=.25,seed_rng=0):
        """Yield policy examples; complexity O(depth*parents*proposals), never O(proposals**depth)."""
        rng=random.Random(seed_rng);frontier=[(seed,scalar_utility(self.evaluate(seed,spec)))];stats=TreeBuildStats(seed.key)
        examples=[]
        for _level in range(depth):
            frontier.sort(key=lambda z:z[1],reverse=True)
            # Mix strong basin states with random states to avoid a greedy narrow tree.
            elite=frontier[:max(1,parents_per_depth//2)];rest=frontier[len(elite):]
            chosen=list(elite)
            if rest and len(chosen)<parents_per_depth:chosen+=rng.sample(rest,min(len(rest),parents_per_depth-len(chosen)))
            candidates=[]
            for parent,_ in chosen:
                ex,kids=self.neighborhood(parent,spec,proposals,rng,temperature);stats.neighborhoods+=1;stats.physics_evals+=len(kids)+1;candidates.extend(kids)
                if ex is not None:examples.append(ex);stats.policy_examples+=1;stats.improving_edges+=ex.improving
            # Dedupe by state identity and keep a bounded stochastic frontier.
            best={}
            for s,u in candidates:
                if s.key not in best or u>best[s.key][1]:best[s.key]=(s,u)
            vals=list(best.values());vals.sort(key=lambda z:z[1],reverse=True)
            elite_n=min(len(vals),frontier_width//2);newf=vals[:elite_n];tail=vals[elite_n:]
            if tail and len(newf)<frontier_width:newf+=rng.sample(tail,min(len(tail),frontier_width-len(newf)))
            frontier=newf;stats.frontier_sizes.append(len(frontier))
            if not frontier:break
        return examples,stats

def policy_pretrain_step(net,optimizer,batch:Sequence[PolicyPretrainExample],spec_encoder:Callable[[DesignSpec],Tensor]|None=None,device="cuda",policy_w=1.,merit_w=.5,feas_w=.5,margin_w=.25,value_w=.5):
    """Train AlphaLenseNet on reverse-tree labels. Batch examples must share action count/token shape."""
    tokens=torch.stack([x.surface_tokens for x in batch]).to(device);masks=torch.stack([x.mask for x in batch]).to(device)
    specs=torch.stack([(spec_encoder(x.spec) if spec_encoder else x.spec.vector()) for x in batch]).to(device);af=torch.stack([x.action_features for x in batch]).to(device)
    out=net(tokens,masks,specs,af);pi=torch.stack([x.policy for x in batch]).to(device)
    lpi=-(pi*F.log_softmax(out["policy_logits"],-1)).sum(-1).mean()
    q=torch.tensor([x.immediate_merit for x in batch],dtype=torch.float32,device=device);z=torch.tensor([x.reachable_value for x in batch],dtype=torch.float32,device=device);f=torch.tensor([x.feasible for x in batch],dtype=torch.float32,device=device);m=torch.stack([x.margins for x in batch]).to(device)
    lq=F.smooth_l1_loss(out["merit"],q);lv=F.smooth_l1_loss(out["value"],z);lf=F.binary_cross_entropy_with_logits(out["feasibility_logit"],f);lm=F.smooth_l1_loss(out["margins"],m)
    loss=policy_w*lpi+merit_w*lq+value_w*lv+feas_w*lf+margin_w*lm
    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);optimizer.step()
    return {k:float(v.detach()) for k,v in {"loss":loss,"policy":lpi,"merit":lq,"value":lv,"feasibility":lf,"margins":lm}.items()}
