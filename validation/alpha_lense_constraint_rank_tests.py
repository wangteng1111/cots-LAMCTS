#!/usr/bin/env python3
from __future__ import annotations
import math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import constraint_rank
spec={'efl_target_mm':50.0,'efl_tol_mm':1.5,'max_f_number':2.0}
for ph in (
 {'J':1e9,'merit_J':1e9,'config_hash':'invalid','error':'nonfinite_cardinal'},
 {'J':1e9,'merit_J':1e9,'config_hash':'valid-but-ray-failed'},
 {'J':float('nan'),'config_hash':'h'},
):
 f,v,r=constraint_rank(ph,spec)
 assert not f and v>=1000 and r[0]==1
ok={'J':12.0,'config_hash':'h','efl':50.5,'fno':1.9}
f,v,r=constraint_rank(ok,spec);assert f and v==0 and r[0]==0
bad={'J':10.0,'config_hash':'h','efl':55.0,'fno':1.9}
f,v,r=constraint_rank(bad,spec);assert not f and v>0 and r[0]==1
print({'status':'passed','invalid_violation':1000.0,'efl_constraint_violation':v})
