#!/usr/bin/env python3
from __future__ import annotations
import math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import Surface,Prescription,PhysicsRecord,assemble_stage2,_structured_delta

def s(r,t,n=1.0,v=0.0):return {'radius':float(r),'thickness':float(t),'n_after':float(n),'v_after':float(v),'clear_aperture':20.0,'conic':0.0,'asphere':()}
a=[s(20,2,1.5,60),s(-30,1,1,0),s(40,3,1.6,50)]
b=[s(20,2,1.5,60),s(10,0.5,1.7,30),s(-30,1,1,0),s(40,3,1.6,50)]
d=_structured_delta({'surfaces':b},{'surfaces':a})
assert d['surface_count_delta']==-1 and d['topology_required']
assert sum(x['op']=='delete' for x in d['operations'])==1
matches=[x for x in d['operations'] if x['op']=='match']
assert len(matches)==3
# Critical alignment property: the three original surfaces align to their optical peers,
# not to same-position indices after the inserted surface.
assert [(x['current_index'],x['goal_index']) for x in matches]==[(0,0),(2,1),(3,2)]

spec={'efl_target_mm':50.0,'max_f_number':2.0}
pbest={'surfaces':a,'stop_after':1,'stop_z_mm':2.2,'image_z_mm':20.0,'source_config_index':0,'design_spec':spec}
pcur={'surfaces':b,'stop_after':1,'stop_z_mm':2.2,'image_z_mm':20.0,'source_config_index':0,'design_spec':spec}
best=PhysicsRecord('seed','best','fam','train',[],pbest,spec,{'J':5.0,'merit_J':5.0,'local_rank':0,'config_hash':'h'},True,0.0,(0,5.0))
cur=PhysicsRecord('seed','cur','fam','train',[],pcur,spec,{'J':20.0,'merit_J':20.0,'local_rank':1,'config_hash':'h'},False,0.3,(1,0.3,20.0))
out=assemble_stage2([cur,best]);m={x['state_hash']:x for x in out}
assert abs(m['cur']['merit_target']+math.log(20.0))<1e-9
assert abs(m['cur']['value_target']+math.log(5.0))<1e-9
assert m['cur']['feasible_target'] is False and m['cur']['reachable_feasible_target'] is True
assert abs(m['cur']['reachable_violation_target'])<1e-12
assert m['cur']['policy_goal_hash']=='best'
assert m['cur']['policy_goal_delta']['topology_required']
print({'status':'passed','alignment':[(x.get('op'),x.get('current_index'),x.get('goal_index')) for x in d['operations']]})
