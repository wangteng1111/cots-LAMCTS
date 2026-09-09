#!/usr/bin/env python3
from __future__ import annotations
import html, json, os, re, urllib.request, urllib.parse
from pathlib import Path

ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)
URL='https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm'
req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
raw=urllib.request.urlopen(req,timeout=60).read().decode('utf-8','replace')
links=re.findall(r'''(?:href|src)\s*=\s*["']([^"']+)["']''',raw,re.I)
# Keep all interesting references, including scripts and OpticalBench invocations.
interesting=[html.unescape(x) for x in links if ('OpticalBench' in x or x.lower().endswith('.js') or 'lens' in x.lower() or 'patent' in x.lower())]
# Also save snippets around likely data/table variables.
snips=[]
for pat in ['OpticalBench','patent','Example','lens','data','table','json','csv','.js']:
    for m in list(re.finditer(pat,raw,re.I))[:25]:
        a=max(0,m.start()-220); b=min(len(raw),m.end()+500)
        snips.append(raw[a:b])
out={'url':URL,'bytes':len(raw.encode()),'link_count':len(links),'interesting_links':interesting[:500],'snippets':snips[:250]}
(ART/'opticalbench_hub_inspect.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
(ART/'opticalbench_hub_raw.txt').write_text(raw)
print(json.dumps({'bytes':out['bytes'],'link_count':out['link_count'],'interesting_count':len(interesting),'first':interesting[:50]},indent=2,ensure_ascii=False),flush=True)
