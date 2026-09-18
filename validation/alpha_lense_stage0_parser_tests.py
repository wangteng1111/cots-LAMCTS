#!/usr/bin/env python3
from __future__ import annotations
import math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import parse_explicit_surfaces

FIXED="""[constants]
[variable distances]
Focal Length	35
Angle of View	63
F-Number	1.8
Image Height	43.2
Bf	19.32
Aperture Diameter	12.472
[lens data]
1	21.084	3.50	1.7197	24.95	50.2
2	48.930	0.10		24.95
3	13.801	4.76	1.6620	18.95	57.7
4	167.720	0.88	1.6206	18.95	38
5	8.537	4.04		12.78
5AS	AS	4.51		12.472
6	-8.967	1.50	1.7850	12.47	25.9
7	-12.051	Bf		14.34
"""
ZOOM_CG="""[constants]
[variable distances]
Focal Length	5.97	16.88
Angle of View	64.2	23.3
F-Number	2.87	5.22
Image Height	7.04	7.04
d6	13.451	1.853
Bf	0	0
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
10	CG	2.17	1.51633	9.42	64.14
11	CG	Bf		9.42
[aspherical data]
8	9.6920	-0.9643	-4.44320E-05	-1.03700E-05
"""
RCL="""[constants]
Type	RCL
[variable distances]
Focal Length	-161.55
Image Height	43.26
Bf	39.50
[lens data]
1	-803.441	1.50	1.79952	29.31	42.2
2	24.164	9.49	1.59270	29.31	35.3
3	-55.099	Bf		29.31
"""
p=parse_explicit_surfaces(FIXED,'fixed','fam',{})
assert len(p.surfaces)==7
assert abs(p.surfaces[0].n_after-1.7197)<1e-9
assert abs(p.surfaces[0].v_after-50.2)<1e-9
assert abs(p.surfaces[0].clear_aperture-24.95)<1e-9
assert abs(p.stop_z_mm-(3.50+0.10+4.76+0.88+4.04))<1e-9
assert p.design_spec['efl_target_mm']==35 and p.design_spec['max_f_number']==1.8
q=parse_explicit_surfaces(ZOOM_CG,'zoom','fam2',{})
assert q.source_config_index==0
assert len(q.surfaces)==10
assert math.isinf(q.surfaces[-2].radius) and math.isinf(q.surfaces[-1].radius)
assert q.surfaces[7].asphere and abs(q.surfaces[7].conic+0.9643)<1e-9
try:
 parse_explicit_surfaces(RCL,'rcl','fam3',{})
 raise AssertionError('RCL without standalone stop must quarantine')
except ValueError as e:
 assert 'external aperture stop' in str(e)
print({'status':'passed','fixed_surfaces':len(p.surfaces),'zoom_surfaces':len(q.surfaces),'stop_z':p.stop_z_mm})
