#!/usr/bin/env python3
"""Stage1/2 readiness gate.

Refuses to fabricate real-lens physics labels with the Mandler-only Q4096 pool.
This job audits Stage0 and emits the exact engineering blocker for Stage1.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage0',default='data/alpha_lense/stage0');ap.add_argument('--out',default='data/alpha_lense/stage12_gate');a=ap.parse_args();src=Path(a.stage0);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 seeds=src/'stage0_seeds.jsonl'
 n=sum(1 for x in seeds.open() if x.strip()) if seeds.exists() else 0
 report={'schema':1,'stage0_seed_count':n,'stage1_ready':False,'stage2_ready':False,'blocker':'Existing evaluator/q4096_cuda_pool.py decodes only 8-int Mandler states. Real commercial prescription perturbations require a prescription-level authoritative Q4096 adapter before physics labels/ranking can be generated.','required_next':'Implement and validate prescription_q4096 adapter against real Stage0 fixtures; then run perturbation -> batched 2xGPU Q4096 -> feasibility-first ranking -> Stage2 target assembly.','prohibited_fallback':'Do not substitute Mandler states or synthetic merit for real-lens labels.'}
 (out/'gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
