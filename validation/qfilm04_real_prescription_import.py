"""Strict translator: pinned DeepLens real-product-named optical prescriptions
-> Qfilm0.4 geometry with EXACT per-wavelength n(lambda) from its Material.

Do not transform names into proof of real manufactured construction: many
datasets are public patent embodiments, and some differ from retail optics.
All provenance and nominal-vs-paraxial inconsistencies are recorded.
"""
from __future__ import annotations
import json,math,re
from pathlib import Path
from evaluator.qfilm02 import Config

CAMERA_SAMPLES=(
 "ef100mm_f2.8","ef35mm_f2.0","ef40mm_f2.8","ef50mm_f1.8",
 "ef85mm_f1.8","rf16mm_f2.8","rf24mm_f1.8","rf35mm_f1.8",
 "rf50mm_f1.8","sigma70mm_f2.8","yongnuo_50mm_f1.8",
)

class IncompatibleLens(ValueError): pass

def _curvature(surface):
 if "roc" in surface:
  return math.inf if float(surface["roc"])==0 else float(surface["roc"])
 c=float(surface.get("c",surface.get("(c)",0.)))
 return math.inf if abs(c)<1e-14 else 1./c

def _material_ndv(material):
 if material.lower()=="air":return 1.,0.
 mt=re.fullmatch(r"([0-9.]+)/([0-9.]+)",material)
 if mt:return float(mt.group(1)),float(mt.group(2))
 return None,None

def _probe_materials(lens,wavelengths):
 # No generic Cauchy approximation: obtain same material ior values as
 # the pinned DeepLens source prescription at every evaluation wavelength.
 return [{str(int(w)) if float(w).is_integer() else str(float(w)):
    float(s.mat2.ior(float(w)*1e-3)) for w in wavelengths}
    for s in lens.surfaces]

def import_geolens_lens(lens,source_dict,identifier,frame_mm=(36.,24.),
                        wavelengths=(430.,470.,510.,550.,590.,630.,670.)):
 # The pinned upstream defaults to 587 nm, not 550 nm. Derived EFL/pupil
 # and object-field normalization must use the SAME reference wavelength
 # as Qfilm. This changes cached cardinal properties, never the prescription.
 original_primary_nm=float(lens.primary_wvln)*1000.
 if abs(original_primary_nm-550.)>1e-9:
  lens.primary_wvln=.550
  lens.post_computation()
 rows=source_dict["surfaces"]
 if len(rows)!=len(lens.surfaces):
  raise IncompatibleLens("source/deeplens surface count mismatch")
 if len(rows)<3:raise IncompatibleLens("lens under 3 surfaces")
 z=0.;points=[];stop_z=None;stop_r=None;prev_z=None;unsupported=[]
 refrac_count=0;nn=_probe_materials(lens,wavelengths)
 for i,s in enumerate(rows):
  typ=s["type"].lower()
  if s.get("is_square",False):unsupported.append(f"square aperture surface={i}")
  if any(abs(float(v))>1e-12 for v in s.get("pos_xy",[0,0])):
   unsupported.append(f"decentered surface={i}")
  if s.get("vec_local") is not None and s["vec_local"] != [0,0,1]:
   unsupported.append(f"tilted surface={i}")
  if typ in ("aperture","stop") or bool(s.get("is_aperture",False)):
   if stop_z is not None:raise IncompatibleLens("multiple explicit stops")
   stop_z=z;stop_r=float(s["r"])
  if typ in ("spheric","spherical","aspheric"):
   refrac_count+=1
   raw_ai=list(s.get("ai",[]) or [])
   if typ=="aspheric" and raw_ai and s.get("use_ai2",True):
    a2=float(raw_ai[0]);raw_ai=raw_ai[1:]
    if abs(a2)>1e-15:
     unsupported.append(f"nonzero-legacy-a2-on-surface-{i}={a2}")
   if typ=="aspheric" and s.get("ai2") is not None and abs(float(s["ai2"]))>1e-15:
    unsupported.append(f"nonzero-ai2-on-surface-{i}")
   n2_by_lambda=nn[i]
   nbase,vd=_material_ndv(str(s.get("mat2","air")))
   if nbase is None:
    nbase=n2_by_lambda.get("550",1.);vd=50.
   entry={"radius":_curvature(s),"thickness":0., "n_after":nbase,
    "v_after":vd,"conic":float(s.get("k",0.)),
    "asphere":raw_ai if typ=="aspheric" else [],
    "clear_aperture":2.*float(s["r"]),
    "n_by_lambda_nm":n2_by_lambda,"surface_source_index":i}
   if prev_z is not None:points[-1]["thickness"]=z-prev_z
   points.append(entry);prev_z=z
  elif typ not in ("aperture","stop"):
   unsupported.append(f"unsupported surface type {typ} at {i}")
  z+=float(s.get("d_next",0.))
 if stop_z is None or stop_r is None:raise IncompatibleLens("missing explicit physical stop")
 if unsupported:raise IncompatibleLens("; ".join(unsupported))
 film_z=float(lens.d_sensor)
 if film_z<=prev_z:raise IncompatibleLens("film at/before last surface")
 points[-1]["thickness"]=film_z-prev_z
 if stop_z<=-1e-9:raise IncompatibleLens("negative stop z")
 if any(v["thickness"]<0 for v in points):raise IncompatibleLens("non-sequential surfaces")
 nominal_efl=float(source_dict.get("foclen",lens.foclen))
 real_efl=float(lens.foclen)
 fnum=float(lens.fnum)
 obj={"surfaces":points,"stop_z_mm":stop_z,"image_z_mm":film_z,
   "design_spec":{"stop_diameter_mm":2*stop_r,"efl_target_mm":real_efl},
   "source_provenance":{"identifier":identifier,"source":"pinned-vccimaging/DeepLens datasets/lenses/camera",
    "source_primary_description":source_dict.get("info"),
    "original_primary_wavelength_nm":original_primary_nm,
    "comparison_primary_wavelength_nm":550.,
    "published_foclen_mm":nominal_efl,"deeplens_paraxial_efl_mm":real_efl,
    "fnumber":fnum,"physical_frame_mm":frame_mm,
    "not_proven_actual_manufacturing_prescription":True}}
 return obj

def lens_metadata(source_dict,identifier):
 """Naming vs patent EFL consistency must be a separate explicit check."""
 m=re.search(r"(?:ef|rf|sigma|yongnuo_)([0-9]+)mm",identifier)
 nominal=float(m.group(1)) if m else None
 design=float(source_dict.get("foclen",float("nan")))
 return {"identifier":identifier,"reference":source_dict.get("info"),
   "claimed_model_focal_mm":nominal,"optical_file_focal_mm":design,
   "nominal_design_relative_difference":abs(nominal-design)/nominal if nominal else None,
   "surface_count":len(source_dict["surfaces"]),
   "aspheric_count":sum(s.get("type")=="Aspheric" for s in source_dict["surfaces"]),
   "design_is_patent_not_verified_production":True}

def extract_full_precision_lens(lens,description):
 """Preserve native in-memory Zemax read precision; avoid lossy 4dp JSON export."""
 raw={"info":description,"foclen":float(lens.foclen),
  "fnum":float(lens.fnum),"surfaces":[]}
 for i,surf in enumerate(lens.surfaces):
  tp=surf.__class__.__name__
  data={"type":tp,"r":float(surf.r),"d_next":float(surf.d_next),
        "mat2":str(getattr(surf.mat2,"name","air")),
        "is_aperture":bool(getattr(surf,"is_aperture",False) or getattr(surf,"is_stop",False)),
        "is_square":bool(getattr(surf,"is_square",False))}
  if hasattr(surf,"c"):
   data["c"]=float(surf.c)
  if hasattr(surf,"k"):
   data["k"]=float(surf.k)
  if hasattr(surf,"ai") and surf.ai is not None:
   data["ai"]=[float(x) for x in surf.ai]
   if getattr(surf,"ai2",None) is not None:
    data["ai"]=[float(surf.ai2)]+data["ai"]
    data["use_ai2"]=True
   else:
    data["use_ai2"]=False
  raw["surfaces"].append(data)
 return raw
