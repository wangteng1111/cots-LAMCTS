#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,re,sys,urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import parse_explicit_surfaces
from scripts.alpha_lense_collect_corpus_v01 import known_rows
NUM=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?')
def fetch(r):
 req=urllib.request.Request(r['url'],headers={'User-Agent':'AlphaLenseResearch/0.8-stage0-diagnose'})
 with urllib.request.urlopen(req,timeout=15) as x:return x.read().decode('utf-8','replace')
def features(t):
 lines=[x.rstrip() for x in t.replace('\r','\n').splitlines() if x.strip()]
 lo=t.lower()
 return {'lines':len(lines),'numeric_lines_ge2':sum(len(NUM.findall(x))>=2 for x in lines),'numeric_lines_ge4':sum(len(NUM.findall(x))>=4 for x in lines),'has_stop':('stop' in lo),'has_aperture':('aperture' in lo),'has_asphere':('asph' in lo or 'conic' in lo),'has_glass':any(k in lo for k in ('glass','abbe',' nd',' vd')),'markers':[k for k in ('rd','th','nd','vd','diameter','radius','thickness','surface','zoom','wavelength','efl','fno','f-number') if k in lo],'preview':lines[:80]}
def reason(err,feat):
 s=str(err)
 if 'stop position not explicit' in s:return 'missing_explicit_stop'
 if 'no unambiguous explicit sequential surface table' in s:
  if feat['numeric_lines_ge4']>=4:return 'surface_table_grammar_mismatch_numeric'
  if feat['numeric_lines_ge2']>=4:return 'surface_table_split_columns_or_labels'
  return 'insufficient_numeric_surface_rows'
 return 'parse_other'
def one(r):
 try:
  t=fetch(r);feat=features(t)
  try:
   p=parse_explicit_surfaces(t,r['filename'],r['filename'].split('_Example')[0],{'url':r['url'],'lens_name':r.get('lens_name')})
   return {'status':'ok','filename':r['filename'],'lens_name':r.get('lens_name'),'surface_count':len(p.surfaces),'stop_after':p.stop_after,'features':feat}
  except Exception as e:
   return {'status':'quarantine','filename':r['filename'],'lens_name':r.get('lens_name'),'reason':reason(e,feat),'error':repr(e),'features':feat}
 except Exception as e:
  return {'status':'quarantine','filename':r['filename'],'lens_name':r.get('lens_name'),'reason':'fetch_error','error':repr(e)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-seeds',type=int,default=200);ap.add_argument('--workers',type=int,default=24);a=ap.parse_args()
 rows=known_rows()[:a.max_seeds];rec=[]
 with ThreadPoolExecutor(max_workers=a.workers) as ex:
  fut={ex.submit(one,r):r for r in rows}
  for i,f in enumerate(as_completed(fut),1):
   rec.append(f.result())
   if i%50==0:print(json.dumps({'done':i,'n':len(rows),'reconstructed':sum(x['status']=='ok' for x in rec)}),flush=True)
 rec.sort(key=lambda x:x['filename'])
 counts=Counter(x.get('reason') for x in rec if x['status']!='ok')
 summary={'attempted':len(rows),'reconstructed':sum(x['status']=='ok' for x in rec),'quarantine':sum(x['status']!='ok' for x in rec),'reason_counts':dict(counts),'workers':a.workers}
 ad=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','.'));ad.mkdir(parents=True,exist_ok=True)
 (ad/'stage0_diagnosis.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rec)+'\n')
 (ad/'stage0_diagnosis_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
