#!/usr/bin/env python3
"""Reproduce QfilmRW's frozen 11-case fixture from the original pinned source."""
import argparse,sys,json,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.qfilm04_fine_11_cpu_gpu import import_case,DEEP_SHA
from validation.qfilm04_real_prescription_import import CAMERA_SAMPLES

def clean(x):
    if isinstance(x,float) and math.isinf(x):return str(x)
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deeplens-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if subprocess.check_output(['git','-C',str(a.deeplens_root),'rev-parse','HEAD'],text=True).strip()!=DEEP_SHA:raise RuntimeError('wrong upstream commit')
    manifest=json.loads((ROOT/'validation/fixtures/qfilm04_real11_source_sha256.json').read_text());rows=[]
    for name in CAMERA_SAMPLES:
        p,sha=import_case(a.deeplens_root,name)
        if sha!=manifest['source_sha256'][name]:raise RuntimeError('changed source prescription')
        rows.append({'name':name,'source_sha256':sha,'prescription':clean(p)})
    d={'schema':'QfilmRWReal11Prescriptions-v1','source_commit':DEEP_SHA,'conversion':'same qfilm04 strict geometry and exact tabulated material indices; infinities encoded as strings','cases':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
