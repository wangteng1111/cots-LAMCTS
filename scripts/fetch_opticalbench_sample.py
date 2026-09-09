#!/usr/bin/env python3
import os, urllib.request
from pathlib import Path
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)
url='https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/US008736971_Example03P.txt'
raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read().decode('utf-8','replace')
(ART/'sample_opticalbench.txt').write_text(raw)
print(raw[:12000],flush=True)
