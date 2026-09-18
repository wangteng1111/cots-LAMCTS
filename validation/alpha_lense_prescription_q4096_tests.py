#!/usr/bin/env python3
from __future__ import annotations
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import parse_explicit_surfaces
from evaluator.prescription_q4096 import evaluate_prescription,evaluator_signature
from evaluator import market_final_direct_q4096_v4 as E

FIXED="""[constants]
[variable distances]
Focal Length	35
Angle of View	63
F-Number	1.8
Image Height	43.2
Bf	19.32
[lens data]
1	21.084	3.50	1.7197	24.95	50.2
2	48.930	0.10		24.95
3	13.801	4.76	1.6620	18.95	57.7
4	167.720	0.88	1.6206	18.95	38
5	8.537	4.04		12.78
5AS	AS	4.51		12.472
6	-8.967	1.50	1.7850	12.47	25.9
7	-12.051	19.32		14.34
"""
ASPH="""[constants]
[variable distances]
Focal Length	16.88
Angle of View	23.3
F-Number	5.22
Image Height	7.04
d6	1.853
Bf	0
[lens data]
1	19.1828	1.10	1.83400	12.37	37.17
2	7.3466	1.75		9.27
3	-143.6780	1.00	1.80400	10.88	46.58
4	10.5443	0.80		9.36
5	10.4893	2.30	1.80518	9.89	25.43
6	120.8022	d6		9.11
7	AS	0.40		4.813
8	9.6920	2.00	1.58313	6.86	59.62
9	-22.1432	0.10		6.86
10	25.7030	2.50	1.58313	9.83	59.62
11	-13.6090	0.60		9.83
[aspherical data]
8	9.6920	-0.9643	-4.44320E-05	-1.03700E-05
"""
a=parse_explicit_surfaces(FIXED,'fixed','fam',{})
b=parse_explicit_surfaces(ASPH,'asph','fam2',{})
legacy_hash=E.CONFIG_HASH
ra=evaluate_prescription(a,device=0)
rb=evaluate_prescription(b,device=1)
assert E.CONFIG_HASH==legacy_hash
for r in (ra,rb):
 assert 'cuda' in r['backend'].lower()
 assert r['config_hash'] not in ('invalid',legacy_hash)
 assert math.isfinite(float(r['J']))
 assert r['evaluator_config']['version']=='alpha_lense_prescription_q4096_v3'
assert abs(ra['target_efl_mm']-35)<1e-9 and abs(ra['target_fno']-1.8)<1e-9 and abs(ra['max_field_deg']-31.5)<1e-9
assert abs(rb['target_efl_mm']-16.88)<1e-9 and b.surfaces[6].asphere
print(json.dumps({'status':'passed','legacy_hash':legacy_hash,'fixed':{k:ra[k] for k in ('J','efl','fno','config_hash','device','film_z_mm','stop_radius_mm','max_field_deg')},'asphere':{k:rb[k] for k in ('J','efl','fno','config_hash','device','film_z_mm','stop_radius_mm','max_field_deg')}}))
