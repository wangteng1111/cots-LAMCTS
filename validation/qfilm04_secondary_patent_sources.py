"""Independent, named manufactured-camera products represented by independently
published Zemax patent examples. These are references to third-party files:
fetch the original files at execution time, never overwrite original metadata.

ZemaxSims explicitly states MOST prescriptions are patents, and are not optimized
or guaranteed to be the actual retail manufactured prescriptions.
"""
from __future__ import annotations
import urllib.request,hashlib
from pathlib import Path
from urllib.parse import quote
ZEMAX_SAMPLES={
"canon_ef50_f1.2l":"Canon/CanonEF50f1.2L/CanonEF50f1.2L.zmx",
"nikon_nikkor50_f1.4_ais":"Nikon/Nikkor50mmf1.4AI-S/Nikkor50mmf1.4AI-S.zmx",
"zeiss_planar_t50_f1.4":"Zeiss/PlanarT50f1.4/Zeiss50f1.4.zmx",
"leica_summicron_m50_f2":"Leica/SummicronM50f2/SummicronM50f2.zmx",
"canon_fd28_f2.8sc":"Canon/CanonFD28f2.8SC/CanonFD28f2.8SC.zmx",
"nikon_nikkor28_f2.8s":"Nikon/Nikkor28mmf2.8S/Nikkor28mmf2.8S.zmx",
"zeiss_distagon_t25_f2.8":"Zeiss/DistagonT25f2.8Contax/DistagonT25f2.8Contax.zmx",
}
ROOT="https://raw.githubusercontent.com/Amarthgul/ZemaxSims/main/"
def load_remote_zmx(name,target_dir,timeout=35):
 if name not in ZEMAX_SAMPLES:raise KeyError(name)
 path=ZEMAX_SAMPLES[name]
 url=ROOT+quote(path,safe="/")
 target_dir=Path(target_dir)
 target_dir.mkdir(parents=True,exist_ok=True)
 output=target_dir/(name+".zmx")
 with urllib.request.urlopen(url,timeout=timeout) as response:
  source=response.read()
 if not source or len(source)>1024*1024 or b"SURF " not in source:
  raise ValueError("not a complete Zemax optical file")
 output.write_bytes(source)
 return str(output),{"url":url,"bytes":len(source),
                     "sha256":hashlib.sha256(source).hexdigest(),
                     "source_kind":"published_patent_or_independent_design",
                     "same_as_actual_manufacturing_not_proven":True}
