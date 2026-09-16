"""Alpha Lense v0.1 Stage0-2 pretrain-data pipeline core.

Stage0 reconstructs source prescriptions into a canonical variable-topology model.
Stage1 generates physical perturbation clouds and evaluates every candidate through
an injected authoritative prescription evaluator.
Stage2 turns physics-grounded local rankings into deterministic pretrain targets.
No model training occurs here.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
from typing import Callable,Iterable
import hashlib,json,math,random,re

@dataclass(frozen=True)
class Surface:
    radius:float
    thickness:float
    n_after:float
    v_after:float
    clear_aperture:float|None=None
    conic:float=0.0
    asphere:tuple[float,...]=()
@dataclass(frozen=True)
class Prescription:
    surfaces:tuple[Surface,...]
    stop_after:int
    source_id:str
    family:str
    design_spec:dict
    provenance:dict
    def canonical(self):
        return {'surfaces':[asdict(s) for s in self.surfaces],'stop_after':self.stop_after,'design_spec':self.design_spec}
    def optical_hash(self):return hashlib.sha256(json.dumps(self.canonical(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
@dataclass(frozen=True)
class Edit:
    kind:str; index:int; delta:float=0.; payload:tuple[float,...]=()
@dataclass
class PhysicsRecord:
    seed_hash:str; candidate_hash:str; family:str; split:str; edits:list[dict]; prescription:dict; design_spec:dict; physics:dict; feasible:bool; violation:float; rank_key:tuple

# P2P/OpticalBench files are heterogeneous. This parser accepts only explicit
# sequential surface rows. Ambiguous files are quarantined, never guessed.
FLOAT=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?'
ROW=re.compile(r'^\s*(?:S(?:URF(?:ACE)?)?\s*)?(\d+)\s+('+FLOAT+r'|INF|INFINITY)\s+('+FLOAT+r')\s+('+FLOAT+r')\s+('+FLOAT+r')(?:\s+('+FLOAT+r'))?',re.I)
def _f(x):return math.inf if x.upper() in ('INF','INFINITY') else float(x)
def parse_explicit_surfaces(text:str,source_id:str,family:str,provenance:dict)->Prescription:
    ss=[]
    for ln in text.replace('\r','\n').splitlines():
        m=ROW.match(ln)
        if not m:continue
        _,r,t,n,v,ca=m.groups();ss.append(Surface(_f(r),float(t),float(n),float(v),float(ca) if ca else None))
    if len(ss)<4:raise ValueError('no unambiguous explicit sequential surface table')
    # Air after surface is represented n=1, v=0. Stop is only accepted when explicit.
    stop=None
    for i,ln in enumerate(text.lower().splitlines()):
        if 'stop' in ln or 'aperture' in ln:
            nums=re.findall(r'\d+',ln)
            if nums:stop=int(nums[0]);break
    if stop is None:raise ValueError('stop position not explicit')
    spec={'efl_target_mm':None,'max_f_number':None,'image_circle_mm':None}
    return Prescription(tuple(ss),stop,source_id,family,spec,provenance)

def split_for_family(family:str)->str:
    x=int(hashlib.sha256(family.encode()).hexdigest()[:8],16)%100
    return 'train' if x<90 else ('val' if x<95 else 'test')

def perturb(p:Prescription,rng:random.Random,depth:int)->tuple[Prescription,list[Edit]]:
    ss=list(p.surfaces);ed=[]
    kinds=('curvature','spacing','thickness','material','aperture','add_element','remove_element','split_element','merge_elements')
    for _ in range(depth):
        k=rng.choice(kinds);i=rng.randrange(len(ss))
        if k=='curvature' and math.isfinite(ss[i].radius):
            d=rng.gauss(0,.025);ss[i]=replace(ss[i],radius=ss[i].radius*(1+d));ed.append(Edit(k,i,d))
        elif k in ('spacing','thickness'):
            d=rng.gauss(0,.04);ss[i]=replace(ss[i],thickness=max(.02,ss[i].thickness*(1+d)));ed.append(Edit(k,i,d))
        elif k=='material' and ss[i].n_after>1.01:
            dn=rng.gauss(0,.006);dv=rng.gauss(0,1.2);ss[i]=replace(ss[i],n_after=max(1.3,min(2.1,ss[i].n_after+dn)),v_after=max(15,min(95,ss[i].v_after+dv)));ed.append(Edit(k,i,dn,(dv,)))
        elif k=='aperture' and ss[i].clear_aperture:
            d=rng.gauss(0,.03);ss[i]=replace(ss[i],clear_aperture=max(.5,ss[i].clear_aperture*(1+d)));ed.append(Edit(k,i,d))
        elif k=='add_element' and len(ss)<64:
            # conservative thin singlet insertion; Stage1 evaluator decides viability.
            r=max(5.,abs(ss[i].radius) if math.isfinite(ss[i].radius) else 50.);new=[Surface(r,1.0,1.5168,64.17),Surface(-r,1.0,1.0,0.0)];ss[i:i]=new;ed.append(Edit(k,i))
        elif k=='remove_element' and len(ss)>6 and i+1<len(ss):del ss[i:i+2];ed.append(Edit(k,i))
        elif k=='split_element' and len(ss)<64:
            s=ss[i];ss[i:i+1]=[replace(s,thickness=max(.02,s.thickness*.48)),Surface(math.inf,max(.02,s.thickness*.04),1.,0.),replace(s,thickness=max(.02,s.thickness*.48))];ed.append(Edit(k,i))
        elif k=='merge_elements' and i+2<len(ss):del ss[i+1:i+3];ed.append(Edit(k,i))
    return replace(p,surfaces=tuple(ss)),ed

def constraint_rank(physics:dict,spec:dict)->tuple[bool,float,tuple]:
    margins=[]
    def add(key,target,tol=None,upper=None,lower=None):
        if target is None or key not in physics:return
        x=float(physics[key]);scale=max(abs(float(target)),1e-6)
        if tol is not None:margins.append(max(0.,abs(x-target)-tol)/scale)
        if upper is not None:margins.append(max(0.,x-upper)/max(abs(upper),1e-6))
        if lower is not None:margins.append(max(0.,lower-x)/max(abs(lower),1e-6))
    add('efl',spec.get('efl_target_mm'),tol=spec.get('efl_tol_mm'))
    add('fno',spec.get('max_f_number'),upper=spec.get('max_f_number'))
    add('min_illum',spec.get('min_relative_illumination'),lower=spec.get('min_relative_illumination'))
    add('dist_max',spec.get('max_distortion_pct'),upper=spec.get('max_distortion_pct'))
    violation=sum(margins);feasible=violation<=1e-12;J=float(physics.get('J',physics.get('merit_J',1e9)))
    return feasible,violation,(0 if feasible else 1,J if feasible else violation,J)

def generate_stage1(seed:Prescription,evaluator:Callable[[Prescription],dict],samples:int,seed_rng:int,max_depth:int=10)->list[PhysicsRecord]:
    rng=random.Random(seed_rng);items=[]
    candidates=[(seed,[])]
    for _ in range(samples):candidates.append(perturb(seed,rng,rng.randint(1,max_depth)))
    for c,ed in candidates:
        ph=evaluator(c);feas,v,rk=constraint_rank(ph,seed.design_spec);items.append(PhysicsRecord(seed.optical_hash(),c.optical_hash(),seed.family,split_for_family(seed.family),[asdict(x) for x in ed],c.canonical(),seed.design_spec,ph,feas,v,rk))
    items.sort(key=lambda x:x.rank_key)
    for i,x in enumerate(items):x.physics['local_rank']=i
    return items

def assemble_stage2(records:Iterable[PhysicsRecord])->list[dict]:
    groups={}
    for r in records:groups.setdefault(r.seed_hash,[]).append(r)
    out=[]
    for seed,rs in groups.items():
        rs=sorted(rs,key=lambda x:x.rank_key);best=rs[0]
        for r in rs:
            # Direct physics-grounded target: best candidate in the same evaluated cloud.
            out.append({'seed_hash':seed,'state_hash':r.candidate_hash,'family':r.family,'split':r.split,'state':r.prescription,'design_spec':r.design_spec,'value_target':-math.log(max(float(r.physics.get('J',r.physics.get('merit_J',1e9))),1e-12)) if r.feasible else -1000.-r.violation,'feasible_target':r.feasible,'violation_target':r.violation,'physics_target':r.physics,'policy_goal_hash':best.candidate_hash,'policy_goal':best.prescription,'policy_goal_edits':best.edits,'evaluator_config_hash':r.physics.get('config_hash')})
    return out
