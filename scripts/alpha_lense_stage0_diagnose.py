#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,re,urllib.request
from collections import Counter,defaultdict
from pathlib import Path
from alpha_lense.stage_pipeline_v01 import parse_explicit_surfaces
from scripts.alpha_lense_collect_corpus_v01 import known_rows
NUM=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?')
def fetch(r):
 req=urllib.request.Request(r['url'],headers={'User-Agent':'AlphaLenseResearch/0.7-stage0-diagnose'})
 with urllib.request.urlopen(req,timeout=15) as x:return x.read().decode('utf-8','replace')
def features(t):
 lines=[x.rstrip() for x in t.replace('\r','\n').splitlines() if x.strip()]
 lo=t.lower()
 return {
  'lines':len(lines),'numeric_lines_ge2':sum(len(NUM.findall(x))>=2 for x in lines),
  'numeric_lines_ge4':sum(len(NUM.findall(x))>=4 for x in lines),
  'has_stop':('stop' in lo),'has_aperture':('aperture' in lo),
  'has_asphere':('asph' in lo or 'conic' in lo),'has_glass':any(k in lo for k in ('glass','abbe',' nd',' vd')),
  'markers':[k for k in ('rd','th','nd','vd','diameter','radius','thickness','surface','zoom','wavelength','efl','fno','f-number') if k in lo],
  'preview':lines[:60],
 }
def reason(err,feat):
 s=str(err)
 if 'stop position not explicit' in s:return 'missing_explicit_stop'
 if 'no unambiguous explicit sequential surface table' in s:
  if feat['numeric_lines_ge4']>=4:return 'surface_table_grammar_mismatch_numeric'
  if feat['numeric_lines_ge2']>=4:return 'surface_table_split_columns_or_labels'
  return 'insufficient_numeric_surface_rows'
 return 'fetch_or_other'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-seeds',type=int,default=200);a=ap.parse_args()
 rec=[];counts=Counter();success=0
 for i,r in enumerate(known_rows()[:a.max_seeds],1):
  try:
   t=fetch(r);feat=features(t)
   try:
    p=parse_explicit_surfaces(t,r['filename'],r['filename'].split('_Example')[0],{'url':r['url'],'lens_name':r.get('lens_name')})
    success+=1;rec.append({'status':'ok','filename':r['filename'],'lens_name':r.get('lens_name'),'surface_count':len(p.surfaces),'stop_after':p.stop_after,'features':feat})
   except Exception as e:
    rr=reason(e,feat);counts[rr]+=1
    rec.append({'status':'quarantine','filename':r['filename'],'lens_name':r.get('lens_name'),'reason':rr,'error':repr(e),'features':feat})
  except Exception as e:
   counts['fetch_error']+=1;rec.append({'status':'quarantine','filename':r['filename'],'lens_name':r.get('lens_name'),'reason':'fetch_error','error':repr(e)})
 summary={'attempted':a.max_seeds,'reconstructed':success,'quarantine':a.max_seeds-success,'reason_counts':dict(counts)}
 ad=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','.'));ad.mkdir(parents=True,exist_ok=True)
 (ad/'stage0_diagnosis.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rec)+'\n')
 (ad/'stage0_diagnosis_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
