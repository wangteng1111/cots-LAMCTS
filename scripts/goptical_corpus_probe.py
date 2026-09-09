#!/usr/bin/env python3
from pathlib import Path
import json, os
root=Path('/var/lib/cots-lamcts/corpora/goptical')
art=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','.')); art.mkdir(parents=True,exist_ok=True)
files=sorted((root/'data').rglob('*.txt'))
out={'root_exists':root.exists(),'data_exists':(root/'data').exists(),'n_txt':len(files),'samples':[]}
for p in files[:5]:
    raw=p.read_text(errors='replace').splitlines()
    out['samples'].append({'path':str(p.relative_to(root)),'first_lines':raw[:80]})
(art/'goptical_probe.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'root_exists':out['root_exists'],'data_exists':out['data_exists'],'n_txt':out['n_txt'],'sample_paths':[x['path'] for x in out['samples']]},indent=2))
