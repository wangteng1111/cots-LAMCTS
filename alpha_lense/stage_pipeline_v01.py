"""Alpha Lense Stage0-2 pretrain-data pipeline core.

Stage0 reconstructs PhotonToPhotos OpticalBench prescriptions into a canonical
variable-topology representation. Stage1 generates physically perturbed clouds
and evaluates them. Stage2 turns physics-grounded rankings into pretrain targets.
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
    stop_z_mm:float|None=None
    image_z_mm:float|None=None
    source_config_index:int=0
    def canonical(self):
        return {
            'surfaces':[asdict(s) for s in self.surfaces],
            'stop_after':self.stop_after,
            'stop_z_mm':self.stop_z_mm,
            'image_z_mm':self.image_z_mm,
            'source_config_index':self.source_config_index,
            'design_spec':self.design_spec,
        }
    def optical_hash(self):
        return hashlib.sha256(json.dumps(self.canonical(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class Edit:
    kind:str; index:int; delta:float=0.; payload:tuple[float,...]=()

@dataclass
class PhysicsRecord:
    seed_hash:str; candidate_hash:str; family:str; split:str; edits:list[dict]; prescription:dict; design_spec:dict; physics:dict; feasible:bool; violation:float; rank_key:tuple

FLOAT=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?'
_NUM=re.compile(r'^'+FLOAT+r'$',re.I)
_INF={'inf','infinity','∞'}
_UNDEF={'','undefined','undef','na','n/a','none'}

def _num(x:str|None)->float|None:
    if x is None:return None
    s=str(x).strip()
    if ',' in s and '.' not in s and s.count(',')==1:
        a,b=s.split(',',1)
        if a.lstrip('+-').isdigit() and b.isdigit():s=a+'.'+b
    if s.lower() in _UNDEF:return None
    if s.lower() in _INF:return math.inf
    try:return float(s)
    except Exception:return None

def _sections(text:str)->dict[str,list[str]]:
    out={};cur=''
    for raw in text.replace('\r\n','\n').replace('\r','\n').splitlines():
        line=raw.strip('\n')
        m=re.match(r'^\s*\[([^]]+)\]\s*$',line)
        if m:
            cur=m.group(1).strip().lower();out.setdefault(cur,[]);continue
        if cur and line.strip():out[cur].append(line.rstrip())
    return out

def _tab_fields(line:str)->list[str]:
    if '\t' in line:return [x.strip() for x in line.split('\t')]
    return re.split(r'\s{2,}',line.strip())

def _series(lines:list[str])->dict[str,list[str]]:
    d={}
    for line in lines:
        p=_tab_fields(line)
        if len(p)>=2:d[p[0].strip()]=p[1:]
    return d

def _finite_series(vs:list[str]|None)->list[float|None]:
    if not vs:return []
    return [_num(x) for x in vs]

def _choose_config(vars:dict[str,list[str]])->int:
    f=_finite_series(vars.get('Focal Length'))
    for i,v in enumerate(f):
        if v is not None and math.isfinite(v):return i
    return 0

def _at(vars:dict[str,list[str]],key:str,idx:int)->float|None:
    vals=vars.get(key) or []
    if not vals:return None
    j=min(idx,len(vals)-1)
    return _num(vals[j])

def _resolve_gap(token:str,vars:dict[str,list[str]],idx:int)->float:
    v=_num(token)
    if v is not None:return float(v)
    vals=vars.get(token)
    if vals:
        x=_at(vars,token,idx)
        if x is not None and math.isfinite(x):return float(x)
    raise ValueError(f'unresolved axial distance token: {token!r}')

def _parse_aspheres(lines:list[str])->dict[int,tuple[float,tuple[float,...]]]:
    out={}
    for line in lines:
        p=_tab_fields(line)
        if len(p)<4:continue
        try:i=int(float(p[0]))
        except Exception:continue
        con=_num(p[2])
        coeff=tuple(float(x) for x in p[3:] if _num(x) is not None)
        out[i]=(0.0 if con is None else float(con),coeff)
    return out

def parse_explicit_surfaces(text:str,source_id:str,family:str,provenance:dict)->Prescription:
    """Parse native P2P OpticalBench sectioned text without guessing missing optics.

    Native lens-data columns are:
      surface-label, radius/special, axial-distance, n_after, clear-diameter, Vd
    Empty n_after means air. AS/FS rows are non-refracting axial entities and
    their distances are folded into the gap between adjacent refracting surfaces.
    """
    sec=_sections(text)
    lens=sec.get('lens data') or []
    if not lens:raise ValueError('missing [lens data] section')
    vars=_series(sec.get('variable distances') or [])
    const=_series(sec.get('constants') or [])
    config=_choose_config(vars)
    asph=_parse_aspheres(sec.get('aspherical data') or [])

    entities=[]
    numeric_order=[]
    for line in lens:
        p=_tab_fields(line)
        if len(p)<3:continue
        label=p[0].strip();rad=p[1].strip();gap=p[2].strip()
        n=p[3].strip() if len(p)>3 else ''
        aperture=p[4].strip() if len(p)>4 else ''
        vd=p[5].strip() if len(p)>5 else ''
        special=rad.upper() in {'AS','FS'} or label.upper().endswith('AS') or label.upper().endswith('FS')
        try:g=_resolve_gap(gap,vars,config)
        except ValueError:
            # Bf is often the final image-space distance and can be in variable data.
            if gap.lower()=='bf':
                bf=_at(vars,'Bf',config)
                if bf is None:raise
                g=float(bf)
            else:raise
        if special:
            kind='AS' if (rad.upper()=='AS' or label.upper().endswith('AS')) else 'FS'
            entities.append({'kind':kind,'label':label,'gap':g,'aperture':_num(aperture)})
            continue
        rr=math.inf if rad.upper()=='CG' else _num(rad)
        if rr is None:raise ValueError(f'unsupported optical entity/radius token {rad!r} at {label}')
        try:idx=float(label)
        except Exception:raise ValueError(f'unsupported surface label {label!r}')
        numeric_order.append(idx)
        nd=_num(n);ca=_num(aperture);vv=_num(vd)
        ndv=1.0 if nd is None else float(nd)
        vdv=0.0 if nd is None else (0.0 if vv is None else float(vv))
        ii=int(idx) if float(idx).is_integer() else None
        con,coef=asph.get(ii,(0.0,()))
        entities.append({'kind':'surface','label':label,'index':idx,'radius':float(rr),'gap':g,'n':ndv,'v':vdv,'aperture':None if ca is None else float(ca),'conic':con,'asphere':coef})

    phys=[e for e in entities if e['kind']=='surface']
    if len(phys)<2:raise ValueError('fewer than two refracting/physical surface rows')
    if not any(e['n']>1.01 for e in phys):raise ValueError('no glass medium found')
    # P2P files normally enumerate front-to-back. Reverse/nonmonotonic examples
    # need an explicit orientation model and remain quarantined rather than guessed.
    if any(b<=a for a,b in zip(numeric_order,numeric_order[1:])):
        raise ValueError('unsupported non-increasing surface order')

    z=0.0;positions=[];stop_z=None;stop_diam=None
    for e in entities:
        e['z']=z
        if e['kind']=='AS':
            if stop_z is not None:raise ValueError('multiple aperture stops not supported')
            stop_z=z;stop_diam=e.get('aperture')
        z+=e['gap']
    image_z=z

    # Convert physical entity positions into refracting-surface-to-surface gaps.
    surfs=[]
    ppos=[e for e in entities if e['kind']=='surface']
    for k,e in enumerate(ppos):
        if k+1<len(ppos):th=float(ppos[k+1]['z']-e['z'])
        else:th=float(image_z-e['z'])
        if th<0:raise ValueError('negative axial spacing')
        surfs.append(Surface(e['radius'],th,e['n'],e['v'],e['aperture'],e['conic'],e['asphere']))
        positions.append(float(e['z']))
    if stop_z is None:
        typ=(const.get('Type') or [''])[0].strip().upper()
        # RCL/teleconverter/close-up attachments legitimately use the main-lens stop.
        # They are reconstructable optics but cannot be authoritative standalone Q4096 seeds.
        if typ=='RCL':raise ValueError('external aperture stop: RCL/attachment is not a standalone Q4096 seed')
        raise ValueError('aperture stop (AS) position not explicit')
    stop_after=sum(1 for zz in positions if zz < stop_z-1e-9)

    efl=_at(vars,'Focal Length',config)
    fno=_at(vars,'F-Number',config)
    imh=_at(vars,'Image Height',config)
    aov=_at(vars,'Angle of View',config)
    bf=_at(vars,'Bf',config)
    total=_at(vars,'Total Length',config)
    apd=_at(vars,'Aperture Diameter',config)
    if apd is None:apd=_at(vars,'Aperture Diameter(m)',config)
    source_type=(const.get('Type') or ['standalone'])[0].strip() or 'standalone'
    spec={
        'efl_target_mm':efl,'efl_tol_mm':None,'max_f_number':fno,'source_type':source_type,
        'image_circle_mm':imh,'max_field_deg':None if aov is None else float(aov)/2.0,
        'angle_of_view_deg':aov,'bfd_mm':bf,'total_length_mm':total,
        'stop_diameter_mm':stop_diam if stop_diam is not None else apd,
        'source_configuration_count':max([len(v) for v in vars.values()] or [1]),
    }
    prov=dict(provenance)
    prov.update({'parser':'p2p_native_v02','source_config_index':config,'source_surface_rows':len(phys),'source_entity_rows':len(entities)})
    return Prescription(tuple(surfs),stop_after,source_id,family,spec,prov,float(stop_z),float(image_z),config)

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

def generate_stage1_candidates(seed:Prescription,samples:int,seed_rng:int,max_depth:int=10)->list[tuple[Prescription,list[Edit]]]:
    rng=random.Random(seed_rng);candidates=[(seed,[])]
    for _ in range(samples):candidates.append(perturb(seed,rng,rng.randint(1,max_depth)))
    return candidates

def records_from_evaluations(seed:Prescription,candidates:list[tuple[Prescription,list[Edit]]],physics:list[dict])->list[PhysicsRecord]:
    if len(candidates)!=len(physics):raise ValueError('candidate/evaluation count mismatch')
    items=[]
    for (cand,ed),ph in zip(candidates,physics):
        feas,v,rk=constraint_rank(ph,seed.design_spec)
        items.append(PhysicsRecord(seed.optical_hash(),cand.optical_hash(),seed.family,split_for_family(seed.family),[asdict(x) for x in ed],cand.canonical(),seed.design_spec,ph,feas,v,rk))
    items.sort(key=lambda x:x.rank_key)
    for i,x in enumerate(items):x.physics['local_rank']=i
    return items

def generate_stage1(seed:Prescription,evaluator:Callable[[Prescription],dict],samples:int,seed_rng:int,max_depth:int=10)->list[PhysicsRecord]:
    candidates=generate_stage1_candidates(seed,samples,seed_rng,max_depth)
    physics=[evaluator(c) for c,_ in candidates]
    return records_from_evaluations(seed,candidates,physics)

def _surface_match_cost(x:dict,y:dict)->float:
    def rel(a,b,scale=1.0):
        try:
            aa=float(a);bb=float(b)
            if not (math.isfinite(aa) and math.isfinite(bb)):return 0.0 if (math.isinf(aa) and math.isinf(bb) and (aa>0)==(bb>0)) else 2.0
            return abs(aa-bb)/max(abs(aa),abs(bb),scale)
        except Exception:return 1.0
    return min(3.0,1.2*rel(x.get('radius'),y.get('radius'),5.0)+.5*rel(x.get('thickness'),y.get('thickness'),1.0)+1.2*abs(float(x.get('n_after',1.0))-float(y.get('n_after',1.0)))+.012*abs(float(x.get('v_after',0.0))-float(y.get('v_after',0.0))))

def _structured_delta(current:dict,goal:dict)->dict:
    a=current.get('surfaces',[]);b=goal.get('surfaces',[]);na=len(a);nb=len(b)
    # Dynamic-programming sequence alignment prevents a topology insertion/removal
    # from shifting every subsequent optical surface target.
    insdel=1.15
    dp=[[0.0]*(nb+1) for _ in range(na+1)]
    bt=[[None]*(nb+1) for _ in range(na+1)]
    for i in range(1,na+1):dp[i][0]=i*insdel;bt[i][0]='delete'
    for j in range(1,nb+1):dp[0][j]=j*insdel;bt[0][j]='insert'
    for i in range(1,na+1):
        for j in range(1,nb+1):
            opts=[(dp[i-1][j-1]+_surface_match_cost(a[i-1],b[j-1]),'match'),(dp[i-1][j]+insdel,'delete'),(dp[i][j-1]+insdel,'insert')]
            dp[i][j],bt[i][j]=min(opts,key=lambda z:z[0])
    ops=[];i=na;j=nb
    while i or j:
        op=bt[i][j]
        if op=='match':
            x,y=a[i-1],b[j-1]
            def delta(k):
                xv=x.get(k);yv=y.get(k)
                if xv is None or yv is None:return None
                try:
                    xv=float(xv);yv=float(yv)
                    if not (math.isfinite(xv) and math.isfinite(yv)):return None
                    return yv-xv
                except Exception:return None
            ops.append({'op':'match','current_index':i-1,'goal_index':j-1,'radius_delta':delta('radius'),'thickness_delta':delta('thickness'),'n_after_delta':delta('n_after'),'v_after_delta':delta('v_after'),'aperture_delta':delta('clear_aperture')});i-=1;j-=1
        elif op=='delete':ops.append({'op':'delete','current_index':i-1});i-=1
        elif op=='insert':ops.append({'op':'insert','goal_index':j-1,'surface':b[j-1]});j-=1
        else:raise RuntimeError('surface alignment backtrace failed')
    ops.reverse()
    return {'surface_count_delta':nb-na,'alignment_cost':float(dp[na][nb]),'operations':ops,'topology_required':any(x['op']!='match' for x in ops)}

def _quality_value(r:PhysicsRecord)->float:
    return -math.log(max(float(r.physics.get('J',r.physics.get('merit_J',1e9))),1e-12))

def assemble_stage2(records:Iterable[PhysicsRecord])->list[dict]:
    groups={}
    for r in records:groups.setdefault(r.seed_hash,[]).append(r)
    out=[]
    for seed,rs in groups.items():
        rs=sorted(rs,key=lambda x:x.rank_key);best=rs[0];best_quality=_quality_value(best)
        for r in rs:
            merit=_quality_value(r)
            out.append({'seed_hash':seed,'state_hash':r.candidate_hash,'family':r.family,'split':r.split,'state':r.prescription,'design_spec':r.design_spec,
              'merit_target':merit,'value_target':best_quality,
              'feasible_target':r.feasible,'violation_target':r.violation,
              'reachable_feasible_target':best.feasible,'reachable_violation_target':best.violation,
              'physics_target':r.physics,'policy_goal_hash':best.candidate_hash,'policy_goal':best.prescription,
              'policy_goal_delta':_structured_delta(r.prescription,best.prescription),
              'evaluator_config_hash':r.physics.get('config_hash'),'local_rank':r.physics.get('local_rank')})
    return out
