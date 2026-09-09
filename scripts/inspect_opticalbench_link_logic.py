#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, urllib.request
from pathlib import Path
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)
url='https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm'
raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read().decode('utf-8','replace')
patterns=['PatentID','ExampleID','modeldata','window.open','location','OpticalBench.htm','href','onclick','encodeURIComponent']
out={}
for pat in patterns:
    hits=[]
    for m in re.finditer(re.escape(pat),raw,re.I):
        a=max(0,m.start()-600); b=min(len(raw),m.end()+1200)
        s=raw[a:b]
        # keep unique compact snippets
        if s not in hits: hits.append(s)
    out[pat]=hits[-20:]
(ART/'opticalbench_link_logic.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
# print snippets nearest tail / script logic, not the large data array beginnings
for pat in patterns:
    print('\n###',pat,'hits',len(out[pat]))
    for s in out[pat][-3:]: print(s.replace('\n',' ')[:1800])
