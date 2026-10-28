#!/usr/bin/env python3
"""Compare only actually completed, protocol-matched fine-grid CPU/GPU data."""
import argparse,json
from pathlib import Path
import numpy as np

def compare(cpu,gpu):
    for key in ("protocol_hash","mode"):
        if cpu[key]!=gpu[key]: raise ValueError("mismatched "+key)
    if cpu["mode"]!="solve": raise ValueError("timing/preflight data are not PSF results")
    if cpu["variant"]!="Qfilm0.4-c" or gpu["variant"]!="Qfilm0.4-g":
        raise ValueError("wrong backend labels")
    out={"schema":"Qfilm04-fine-CPU-GPU-comparison-v1","protocol":cpu["protocol"],
        "cpu_status":cpu["status"],"gpu_status":gpu["status"],"cases":[],
        "physics_certified":False,"training_eligible":False}
    for name in cpu["protocol"]["names"]:
        c=next((x for x in cpu["cases"] if x["name"]==name),{})
        g=next((x for x in gpu["cases"] if x["name"]==name),{})
        if c.get("source_sha256")!=g.get("source_sha256"):
            out["cases"].append({"name":name,"status":"missing_or_mismatched_source"});continue
        rows=[]
        for cf in c.get("fields",[]):
            gf=next((f for f in g.get("fields",[]) if f["index"]==cf["index"]),{})
            for cs in cf["samples"]:
                gs=next((s for s in gf.get("samples",[]) if s["wavelength_nm"]==cs["wavelength_nm"]),{})
                if cs.get("state")!="completed" or gs.get("state")!="completed": continue
                delta=abs(np.asarray(cs["mtf"])-np.asarray(gs["mtf"]))
                rows.append({"field_h":cf["field_h"],"wavelength_nm":cs["wavelength_nm"],
                    "max_abs_mtf_difference":float(delta.max()),"mean_abs_mtf_difference":float(delta.mean())})
        expected=len(cpu["protocol"]["fields_h"])*len(cpu["protocol"]["wavelengths_nm"])
        row={"name":name,"matched_completed_psfs":len(rows),"expected_psfs":expected,
            "status":"complete" if len(rows)==expected else "incomplete","mono_comparisons":rows}
        if rows: row["max_abs_mono_mtf_difference"]=max(x["max_abs_mtf_difference"] for x in rows)
        poly=[]
        for cf in c.get("fields",[]):
            gf=next((f for f in g.get("fields",[]) if f["index"]==cf["index"]),{})
            if cf.get("complete") and gf.get("complete"):
                poly.append(float(np.max(abs(np.asarray(cf["polychromatic_mtf_ts"])-np.asarray(gf["polychromatic_mtf_ts"])))))
        if poly: row["max_abs_polychromatic_mtf_difference"]=max(poly)
        cj=c.get("quality",{}).get("J_task");gj=g.get("quality",{}).get("J_task")
        if row["status"]=="complete" and cj is not None and gj is not None: row["J_abs_difference"]=abs(cj-gj)
        out["cases"].append(row)
    out["status"]="complete" if all(c["status"]=="complete" for c in out["cases"]) else "incomplete"
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu",type=Path,required=True);ap.add_argument("--gpu",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    out=compare(json.loads(a.cpu.read_text()),json.loads(a.gpu.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(out["status"])
    return 0 if out["status"]=="complete" else 2

if __name__=="__main__": raise SystemExit(main())
