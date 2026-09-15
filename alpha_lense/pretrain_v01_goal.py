"""Goal-directed Alpha Lense v0.1 policy pretraining.

A known-good, physics-refined lens x* defines a basin target under DesignSpec c.
Multi-scale perturbations generate starting states x.  Supervision points DIRECTLY
from each x toward x*, not through arbitrary perturbation ancestry.

Topology is part of the action space: perturbations may add/remove/split/merge
optical elements, so lens element count is learnable rather than fixed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Protocol, Sequence
import random, torch
from .core import DesignSpec, EvalResult

Tensor=torch.Tensor

TOPOLOGY_KINDS=("add_element","remove_element","split_element","merge_elements")
CONTINUOUS_KINDS=("curvature","spacing","thickness","refractive_index","abbe","asphere","aperture")

@dataclass(frozen=True)
class GoalPerturbation:
    kind:str
    target:tuple[int,...]
    deltas:tuple[float,...]=()
    payload:tuple[float,...]=()
    @property
    def changes_element_count(self)->bool:return self.kind in TOPOLOGY_KINDS

@dataclass(frozen=True)
class GoalState:
    key:Hashable
    perturbation_depth:int=0

class GoalAdapter(Protocol):
    def perturb(self,goal:GoalState,spec:DesignSpec,depth:int,rng:random.Random)->GoalState:...
    def tokenize(self,state:GoalState)->tuple[Tensor,Tensor]:...
    def direct_goal_action(self,state:GoalState,goal:GoalState,spec:DesignSpec)->Tensor:...
    def element_count(self,state:GoalState)->int:...

@dataclass
class GoalPolicyExample:
    state:GoalState
    goal:GoalState
    spec:DesignSpec
    tokens:Tensor
    mask:Tensor
    # Continuous/topological optical intention pointing directly to x*.
    direct_action:Tensor
    source_element_count:int
    target_element_count:int
    perturbation_depth:int

class GoalDirectedDatasetBuilder:
    """Generate multi-scale perturbation clouds; every label points to x*."""
    def __init__(self,adapter:GoalAdapter):self.adapter=adapter
    def build(self,goal:GoalState,spec:DesignSpec,samples:int=10000,max_depth:int=10,seed:int=0)->list[GoalPolicyExample]:
        rng=random.Random(seed);out=[]
        for _ in range(samples):
            # Broad curriculum: all distances from near to heavily perturbed.
            depth=rng.randint(1,max_depth);x=self.adapter.perturb(goal,spec,depth,rng)
            tok,mask=self.adapter.tokenize(x);a=self.adapter.direct_goal_action(x,goal,spec)
            out.append(GoalPolicyExample(x,goal,spec,tok,mask,a,self.adapter.element_count(x),self.adapter.element_count(goal),depth))
        return out

def topology_delta(example:GoalPolicyExample)->int:
    return int(example.target_element_count-example.source_element_count)

def validate_goal(goal_eval:EvalResult)->None:
    if not goal_eval.feasible:raise ValueError("goal lens must satisfy its DesignSpec after authoritative physics refinement")
