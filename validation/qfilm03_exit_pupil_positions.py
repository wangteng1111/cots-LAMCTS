#!/usr/bin/env python3
"""Inspect DeepLens Huygens physical secondary-source plane relative to glass,
aperture and film, for identical full-frame lens prescriptions.
"""
from __future__ import annotations
import sys,os,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deeplens import GeoLens
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,BK7
import torch
p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
out={}
for name,pp in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
 doc,_=conversion(pp,sensor_size=48.,sensor_res=9600,material_mode="bk7")
 f=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/(name+"_exit.json")
 f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(doc))
 lens=GeoLens(filename=str(f),device="cuda:0",dtype=torch.float64)
 z,r=lens.get_exit_pupil();ze,re=lens.get_entrance_pupil()
 surface_vertices=[float(s.d) for s in lens.surfaces]
 out[name]={"exit_pupil_z_mm":float(z),"exit_pupil_radius_mm":float(r),
     "entrance_pupil_z_mm":float(ze),"entrance_pupil_radius_mm":float(re),
     "surface_vertex_z_mm":surface_vertices,
     "film_z_mm":float(lens.d_sensor),
     "last_refracting_vertex_mm":max(float(s.d) for s in lens.surfaces if not getattr(s,"is_aperture",False)),
     "stop_index":int(lens.aper_idx),
     "stop_z_mm":float(lens.surfaces[lens.aper_idx].d)}
 print("EXIT_PUPIL",name,json.dumps(out[name]),flush=True)
print("DONE",json.dumps(out))
