#!/usr/bin/env python3
"""Physically strict Broyden front-stop inversion comparison (experimental)."""
from __future__ import annotations
import os,json
from pathlib import Path
os.environ['QFILM_TEST_FUSED_INTERNAL']='1'
from validation.qfilm02_accelerated import one_case,thin_lens,rich_lens
def main():
 rows=[
  one_case('internal_5fields_32',rich_lens(),32,5),
  one_case('internal_15fields_64',rich_lens(),64,15),
  one_case('front_100fields_64',thin_lens(),64,100,(430.,470.,510.,550.,590.,630.,670.))]
 out={'schema':'qfilm02-broyden-test-v1','passed':all(r['passed'] for r in rows),'cases':rows}
 file=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_broyden_compare.json'
 file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(out,indent=2)+'\n')
 print('SUMMARY',json.dumps({'passed':out['passed'],'cases':len(rows)}),flush=True)
 if not out['passed']:raise SystemExit(2)
if __name__=='__main__':main()
