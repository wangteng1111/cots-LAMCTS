#!/usr/bin/env python3
"""Local execution wrapper: persist child identity and actual return code."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('request');ap.add_argument('source');ap.add_argument('python');a=ap.parse_args()
    request=Path(a.request).resolve();job=request.parent;req=json.loads(request.read_text());source=Path(a.source).resolve()
    entry=(source/req['entrypoint']).resolve()
    if source not in entry.parents:raise ValueError('entrypoint escapes source')
    proc=None;stopping=[False]
    def stop(signum,frame):
        stopping[0]=True
        if proc is not None and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        def limits():
            if req.get('memory_gb',0)>0:
                import resource
                limit=req['memory_gb']*1024**3;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        proc=subprocess.Popen([a.python,'-u',str(entry),*req['args']],cwd=source,start_new_session=True,preexec_fn=limits)
        atomic(job/'worker-process.json',{'pid':proc.pid,'wrapper_pid':os.getpid(),'job_id':req['job_id'],'started_at':time.time(),
            'wrapper_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]})
        if stopping[0]:stop(signal.SIGTERM,None)
        deadline=time.monotonic()+req['timeout_sec'];stop_at=None;timed_out=False
        while proc.poll() is None:
            if time.monotonic()>=deadline:timed_out=True;stop(signal.SIGTERM,None)
            if stopping[0]:
                if stop_at is None:stop_at=time.monotonic()
                if time.monotonic()-stop_at>20:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            time.sleep(.5)
        rc=proc.wait()
        atomic(job/'worker-exit.json',{'job_id':req['job_id'],'returncode':rc,'finished_at':time.time(),
            'timed_out':timed_out,'stopped':stopping[0] and not timed_out,'code_commit':req['code_commit']})
        return 0
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            stop(signal.SIGTERM,None)
            try:proc.wait(timeout=25)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        atomic(job/'worker-exit.json',{'job_id':req['job_id'],'returncode':None,'finished_at':time.time(),
            'wrapper_error':type(exc).__name__,'code_commit':req['code_commit']})
        return 2

if __name__=='__main__':raise SystemExit(main())
