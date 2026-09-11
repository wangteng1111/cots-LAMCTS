#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENV=Path('/var/lib/cots-lamcts/venv/bin/python')
if VENV.exists() and sys.prefix != str(VENV.parent.parent):
    os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
GAPS=((.20,.29,.38,.47,.56),(7.50,8.32,9.14,9.96,10.78),(11.,12.18,13.36,14.54,15.72),(.20,.29,.38,.47,.56))
def decode(st):
    st=tuple(map(int,st));d=tuple((x//2,bool(x%2)) for x in st[:4]);g=tuple(GAPS[i][st[4+i]] for i in range(4));return d,g
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    from evaluator import market_final_direct_q4096_v4 as E
    st=json.loads(a.state);d,g=decode(st);r=E.evaluate_final_with_gaps(d,g,4096)
    Path(a.out).write_text(json.dumps({'state':st,'J':float(r['merit_J']),'config_hash':r['config_hash']}))
if __name__=='__main__':main()
