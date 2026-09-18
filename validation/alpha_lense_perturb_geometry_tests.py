#!/usr/bin/env python3
from __future__ import annotations
import math,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import Surface,Prescription,perturb,parse_explicit_surfaces

p=Prescription(
 tuple([
  Surface(30,2,1.5,60,20),Surface(-40,4,1,0,20),
  Surface(50,3,1.6,50,18),Surface(-60,10,1,0,18),
 ]),
 2,'x','fam',{'efl_target_mm':50,'efl_tol_mm':1.5,'max_f_number':2.0},{},
 3.0,19.0,0
)
for seed in range(100):
 q,ed=perturb(p,random.Random(seed),8)
 assert len(q.surfaces)>=4 and len(q.surfaces)<=64
 assert 0<=q.stop_after<=len(q.surfaces)
 assert abs(q.image_z_mm-sum(s.thickness for s in q.surfaces))<1e-9
 if 0<q.stop_after<len(q.surfaces)+1:
  prev=sum(s.thickness for s in q.surfaces[:q.stop_after-1])
  nxt=prev+q.surfaces[q.stop_after-1].thickness
  assert prev-1e-9<=q.stop_z_mm<=nxt+1e-9
 assert all(s.thickness>=.02 for s in q.surfaces)

txt="""[constants]
[variable distances]
Focal Length	50
F-Number	2
Image Height	43.2
Bf	20
[lens data]
1	30	2	1.5	20	60
2	-40	4		20
2AS	AS	1		10
3	50	3	1.6	18	50
4	-60	20		18
"""
q=parse_explicit_surfaces(txt,'t','f',{})
assert abs(q.design_spec['efl_tol_mm']-1.5)<1e-12
print({'status':'passed','efl_tol_mm':q.design_spec['efl_tol_mm'],'seeds_checked':100})
