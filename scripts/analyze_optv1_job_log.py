#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, re, statistics, time
from collections import defaultdict
from pathlib import Path

ROUND_RE = re.compile(r"^ROUND\s+(\d+)\s+n\s+(\d+)\s+bestJ\s+([^\s]+)\s+best\s+(\([^\n]+?\))\s+eligible\s+(\d+)$", re.M)
Q_RE = re.compile(r"^Q\s+(init-s\d+|round(\d+)-s\d+)\s+(\d+)\s+(\([^\n]+?\))\s+([^\s]+)$", re.M)


def parse_state(s: str):
    return [int(x.strip()) for x in s.strip()[1:-1].split(',')]


def quantile(xs, q):
    if not xs: return None
    ys=sorted(xs); p=(len(ys)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    if lo==hi:return ys[lo]
    return ys[lo]*(hi-p)+ys[hi]*(p-lo)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target-job', required=True); ns=ap.parse_args()
    root=Path(os.environ.get('COTS_BRIDGE_ROOT','~/.cots_lamcts_bridge')).expanduser().resolve()
    log=root/'jobs'/ns.target_job/'job.log'
    text=log.read_text(errors='replace')
    rounds=[]
    for m in ROUND_RE.finditer(text):
        rounds.append({'round':int(m.group(1)),'n':int(m.group(2)),'bestJ':float(m.group(3)),
                       'state':parse_state(m.group(4)),'eligible':int(m.group(5))})
    qs=[]
    for m in Q_RE.finditer(text):
        try:j=float(m.group(5))
        except:continue
        label=m.group(1); rnd=None if label.startswith('init') else int(m.group(2))
        qs.append({'round':rnd,'idx':int(m.group(3)),'state':parse_state(m.group(4)),'J':j})

    improvements=[]; prev=float('inf')
    for r in rounds:
        if r['bestJ'] < prev-1e-12:
            improvements.append(r.copy()); prev=r['bestJ']
    thresholds={}
    for th in [30,25,22,21,20,19.5,19,18.5,18.25,18.2,18.1,18.0]:
        hit=next((r for r in rounds if r['bestJ']<=th),None); thresholds[str(th)]=hit

    plateaus=[]
    if rounds:
        start=rounds[0]; last=rounds[0]
        for r in rounds[1:]:
            if abs(r['bestJ']-last['bestJ'])>1e-12:
                plateaus.append({'J':last['bestJ'],'state':last['state'],'start_n':start['n'],'end_n':last['n'],
                                 'evaluations':last['n']-start['n']})
                start=r
            last=r
        plateaus.append({'J':last['bestJ'],'state':last['state'],'start_n':start['n'],'end_n':last['n'],
                         'evaluations':last['n']-start['n']})
    plateaus=sorted(plateaus,key=lambda x:x['evaluations'],reverse=True)

    bins=defaultdict(lambda:{'q':[],'eligible':[],'best':[]})
    for q in qs:
        # approximate eval bin by round; init -> 0, after init n ~= 96+16*r
        n=0 if q['round'] is None else 96+16*q['round']
        b=(n//1000)*1000; bins[b]['q'].append(q['J'])
    for r in rounds:
        b=(r['n']//1000)*1000; bins[b]['eligible'].append(r['eligible']); bins[b]['best'].append(r['bestJ'])
    bstats=[]
    for b in sorted(bins):
        d=bins[b]; x=d['q']; e=d['eligible']
        bstats.append({'bin_start':b,'q_count':len(x),'J_median':quantile(x,.5),'J_p10':quantile(x,.1),'J_p01':quantile(x,.01),
                       'J_min':min(x) if x else None,'eligible_median':quantile(e,.5),'eligible_min':min(e) if e else None,
                       'eligible_max':max(e) if e else None,'best_end':d['best'][-1] if d['best'] else None})

    # coordinate frequencies among improving best states
    coord=[]
    for k in range(4):
        counts=defaultdict(int)
        for r in improvements: counts[r['state'][k]]+=1
        coord.append(sorted(({'value':v,'count':c} for v,c in counts.items()),key=lambda z:(-z['count'],z['value']))[:10])

    final=rounds[-1] if rounds else None
    result={'schema':1,'target_job':ns.target_job,'snapshot_at':time.time(),'log_size':log.stat().st_size,
            'round_count':len(rounds),'q_count':len(qs),'final_round':final,'improvement_count':len(improvements),
            'improvements':improvements,'threshold_crossings':thresholds,'longest_plateaus':plateaus[:12],
            'per_1000':bstats,'improvement_coordinate_frequencies':coord}
    out=Path(os.environ['COTS_JOB_ARTIFACT_DIR']); out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    compact={k:result[k] for k in ['round_count','q_count','final_round','improvement_count','threshold_crossings','longest_plateaus','per_1000']}
    print(json.dumps(compact,indent=2,sort_keys=True))

if __name__=='__main__': main()
