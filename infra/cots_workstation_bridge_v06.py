#!/usr/bin/env python3
"""Bridge 0.6: offline supervision, durable coalesced publication, adoption.

All HTTP and git fetch operations are background work. A network failure cannot
escape the child polling loop. Claimed IDs are never replayed. Existing jobs
are observed without killing them or inventing an exit code.
"""
from __future__ import annotations
import argparse,base64,fcntl,hashlib,json,os,re,socket,subprocess,sys,threading,time,uuid
from pathlib import Path
import cots_workstation_bridge as v3
import cots_workstation_bridge_v04 as v4
import cots_workstation_bridge_v05 as v5

VERSION='0.6.1'
OUTBOX=v3.STATE/'outbox_v06';OUTBOX.mkdir(parents=True,exist_ok=True)
LOCK=threading.RLock();LAST_SYNC=0.;NETWORK_ERROR=None
ORIGINAL_PUT=v4._ORIG_PUT_FILE
TERMINAL={'finished','failed','timed_out','stopped','rejected','recovered','recovered_unknown_exit'}
UPLOAD_MARKS={}
MANAGED=set()
ADOPT_LOCK=threading.RLock()
ORIGINAL_VALIDATE=v3.validate_request
STOP_SENT={}

def stop_adopted(processes, now=None):
    """Stop only explicitly cancelled jobs after rechecking PID identity.

    Never signal a process group inherited from the service. A reused PID or
    an unrelated Python process must not be signalled. Persist the reason
    separately; an interrupted optical calculation is never convergence.
    """
    import signal
    now=time.monotonic() if now is None else now
    for rec in processes:
        if not cached_stop(rec['job_id']):continue
        proc=Path('/proc')/str(rec['pid'])
        try:
            if (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=rec['start_ticks']:continue
            prefix=str(v3.JOBS/rec['job_id']/'src')+'/'
            if not any(a.startswith(prefix) and a.endswith('.py') for a in
                       (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')):continue
            key=(rec['pid'],rec['start_ticks'])
            first=STOP_SENT.setdefault(key,now)
            os.kill(rec['pid'],signal.SIGKILL if now-first>=20 else signal.SIGTERM)
            atomic(v3.STATE/'cancelled_adopted_v06'/f"{rec['job_id']}.json",
                {'job_id':rec['job_id'],'state':'stop_requested','observed_unix':time.time(),
                 'pid':rec['pid'],'start_ticks':rec['start_ticks'],'reason':'explicit remote stop control'})
        except (OSError,IndexError):continue

def validate_request(obj,path):
    if path.startswith('remote/prepared/bridge-v06/'):
        from qfilm_convergence_status import NEW_IDS
        if obj.get('job_id') not in NEW_IDS.values() or path!=f"remote/prepared/bridge-v06/{obj['job_id']}.json":
            raise ValueError('unapproved prepared request')
        return ORIGINAL_VALIDATE(obj,f"remote/queue/{obj['job_id']}.json")
    return ORIGINAL_VALIDATE(obj,path)

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def freshness(data,fallback=0.):
    try:
        d=json.loads(data)
        if not isinstance(d,dict):return (0,float(fallback))
        values=[d.get(k) for k in ('timestamp','updated_at','updated_unix','finished_at','finished_unix','observed_unix','claimed_at','started_at')]
        times=[float(v) for v in values if isinstance(v,(int,float))]
        return (int(d.get('state') in TERMINAL or d.get('status') in ('completed','completed_with_errors')),max(times) if times else float(fallback))
    except (ValueError,TypeError):return (0,float(fallback))

def enqueue(path,data,message):
    key=hashlib.sha256(path.encode()).hexdigest();target=OUTBOX/(key+'.json')
    now=time.time();rank=freshness(data,now)
    first_queued=now
    with LOCK:
        watermark=v3.STATE/'publish_watermarks_v06.json'
        seen=json.loads(watermark.read_text()) if watermark.exists() else {}
        if key in seen and tuple(seen[key])>rank:return
        if target.exists():
            old=json.loads(target.read_text())
            if tuple(old['freshness'])>rank:return
            first_queued=old['queued_at']
        atomic(target,{'path':path,'message':message,'data_b64':base64.b64encode(data).decode(),
            'freshness':rank,'sha256':hashlib.sha256(data).hexdigest(),'queued_at':first_queued})

def migrate_pending():
    for file in sorted(v4.PENDING.glob('*.json')):
        try:
            rec=json.loads(file.read_text());data=base64.b64decode(rec['data_b64'])
            # Replay in semantic observation order, never hashed-filename order.
            target=OUTBOX/(hashlib.sha256(rec['path'].encode()).hexdigest()+'.json')
            rank=freshness(data,rec.get('queued_at',0))
            with LOCK:
                if target.exists() and tuple(json.loads(target.read_text())['freshness'])>rank:
                    file.unlink();continue
                enqueue(rec['path'],data,rec.get('message','migrated publication'))
                saved=json.loads(target.read_text()) if target.exists() else None
                if saved and saved['sha256']==hashlib.sha256(data).hexdigest():
                    saved['freshness']=rank;atomic(target,saved)
            file.unlink()
        except (OSError,ValueError,KeyError):continue

def publish_one():
    with LOCK:
        records=[]
        for path in OUTBOX.glob('*.json'):
            try:records.append((path,json.loads(path.read_text())))
            except (OSError,ValueError):pass
    if not records:return False
    # Terminal evidence first, then newest observations; bounded one item/tick.
    records.sort(key=lambda x:(-x[1]['freshness'][0],x[1]['queued_at']))
    file,rec=records[0];data=base64.b64decode(rec['data_b64'])
    remote=v3.gh_content(rec['path'])
    skip=False
    if remote and remote.get('content'):
        existing=base64.b64decode(remote['content'])
        immutable=rec['path'].startswith('remote/finals/') or rec['path'].endswith('-terminal.json')
        if immutable and existing!=data:
            atomic(v3.STATE/'publication_conflict_v06.json',{'path':rec['path'],'observed_at':time.time()})
            skip=True
        elif freshness(existing)>tuple(rec['freshness']):skip=True
        elif existing==data:skip=True
    if not skip:ORIGINAL_PUT(rec['path'],data,rec['message'])
    with LOCK:
        marks=v3.STATE/'publish_watermarks_v06.json'
        seen=json.loads(marks.read_text()) if marks.exists() else {}
        key=file.stem;seen[key]=max(tuple(seen.get(key,(0,0))),tuple(rec['freshness']))
        atomic(marks,seen)
        if file.exists() and json.loads(file.read_text())['sha256']==rec['sha256']:file.unlink()
    return True

def publisher():
    while True:
        try:
            if not publish_one():time.sleep(1)
        except Exception as exc:
            atomic(v3.STATE/'publisher_health_v06.json',{'observed_at':time.time(),'error_type':type(exc).__name__})
            time.sleep(10)

def sync_control():
    global LAST_SYNC,NETWORK_ERROR
    while True:
        try:
            v3.run(['git','fetch','--prune','origin'],cwd=v3.MIRROR,timeout=45)
            LAST_SYNC=time.time();NETWORK_ERROR=None
        except Exception as exc:NETWORK_ERROR=type(exc).__name__
        atomic(v3.STATE/'control_sync_v06.json',{'last_success_unix':LAST_SYNC,'error_type':NETWORK_ERROR})
        time.sleep(20)

def cached_exact(sha):
    if not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('exact commit SHA required')
    try:
        v3.run(['git','cat-file','-e',sha+'^{commit}'],cwd=v3.MIRROR,timeout=5)
        v3.run(['git','merge-base','--is-ancestor',sha,'origin/main'],cwd=v3.MIRROR,timeout=5)
    except Exception:raise RuntimeError('commit not available in synchronized local mirror')
    return sha

def cached_claimed(jid):
    if v5._read_local_receipt(jid) is not None:return True
    for prefix in ('remote/claims/','remote/finals/'):
        r=subprocess.run(['git','cat-file','-e',f'origin/workstation-results:{prefix}{jid}.json'],cwd=v3.MIRROR,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
        if r.returncode==0:return True
    return False

def cached_stop(jid):
    try:return json.loads(v3.git_show('origin/main',f'remote/control/{jid}.json')).get('action')=='stop'
    except Exception:return False

def discover_processes():
    found=[];prefix=str(v3.JOBS)+'/'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            paths=[a for a in args if a.startswith(prefix) and '/src/' in a and a.endswith('.py')]
            if not paths:continue
            jid=paths[0][len(prefix):].split('/')[0]
            if jid in MANAGED:continue
            found.append({'pid':int(p.name),'job_id':jid,'script':Path(paths[0]).name,
                'start_ticks':(p/'stat').read_text().rsplit(')',1)[1].split()[19]})
        except (OSError,IndexError):continue
    return found

def upload_changed(jid):
    root=v3.JOBS/jid/'artifacts';count=0
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in v3.TEXT_ARTIFACT_SUFFIXES:continue
        try:
            stat=path.stat();mark=(stat.st_mtime_ns,stat.st_size)
            if stat.st_size>v3.MAX_ARTIFACT_BYTES or UPLOAD_MARKS.get(str(path))==mark:continue
            enqueue(f'remote/artifacts/{jid}/{path.relative_to(root).as_posix()}',path.read_bytes(),f'checkpoint {jid}')
            UPLOAD_MARKS[str(path)]=mark;count+=1
            if count>=16:break
        except OSError:continue

def observe_adopted():
    with ADOPT_LOCK:return _observe_adopted()

def _observe_adopted():
    processes=discover_processes()
    stop_adopted(processes)
    oldfile=v3.STATE/'adopted_processes_v06.json'
    previous=json.loads(oldfile.read_text()).get('processes',[]) if oldfile.exists() else []
    current_ids={p['job_id'] for p in processes}
    for jid in {p['job_id'] for p in previous}|current_ids:
        upload_changed(jid)
        if jid not in current_ids:
            exitfile=v3.JOBS/jid/'worker-exit.json'
            wrapperfile=v3.JOBS/jid/'worker-process.json'
            if not exitfile.exists() and wrapperfile.exists():
                identity=json.loads(wrapperfile.read_text());wp=Path('/proc')/str(identity.get('wrapper_pid',0))
                try:
                    if wp.is_dir() and (wp/'stat').read_text().rsplit(')',1)[1].split()[19]==identity.get('wrapper_start_ticks'):
                        processes.extend(p for p in previous if p['job_id']==jid)
                        continue
                except OSError:pass
            if exitfile.exists():
                exitdata=json.loads(exitfile.read_text());rc=exitdata.get('returncode')
                value={'schema':1,'job_id':jid,'state':'finished' if rc==0 else 'failed',**exitdata,'bridge_version':VERSION}
            else:
                cancelled=(v3.STATE/'cancelled_adopted_v06'/f'{jid}.json').exists()
                value={'schema':1,'job_id':jid,'state':'stopped' if cancelled else 'recovered_unknown_exit','returncode':None,
                    'finished_at':time.time(),'bridge_version':VERSION,'note':'adopted process ended; original exit status unavailable; inspect native scientific terminal'}
            atomic(v3.STATE/'adopted_terminals_v06'/f'{jid}.json',value)
            v3.put_json(f'remote/results/{jid}.json',value,'adopted task terminal')
            v3.put_json(f'remote/finals/{jid}.json',value,'adopted task terminal')
    atomic(oldfile,{'observed_at':time.time(),'processes':processes})
    v3.put_json('remote/health/adopted-processes.json',{'observed_unix':time.time(),
        'bridge_version':VERSION,'processes':processes},'adopted process identities')
    return processes

def adoption_loop():
    while True:
        try:
            observe_adopted()
            from qfilm_convergence_status import collect
            report=collect(v3.JOBS)
            atomic(v3.STATE/'qfilm-three-verification.json',report)
            v3.put_json('remote/health/qfilm-three-verification.json',report,'independent Qfilm terminal verification')
        except Exception as exc:print('adoption checkpoint deferred:',type(exc).__name__,flush=True)
        time.sleep(30)

def queue_paths():
    processes=observe_adopted()
    # A live remediation job must finish before launching another full test.
    if any(p['script']=='qfilm04_remediation_run.py' for p in processes):return []
    if time.time()-LAST_SYNC>300:return []
    out=v3.run(['git','ls-tree','-r','--name-only','origin/main','remote/queue','remote/prepared/bridge-v06'],cwd=v3.MIRROR,timeout=10)
    selected=os.environ.get('COTS_BRIDGE_QUEUE_INCLUDE',r'.*')
    return sorted(p for p in out.splitlines() if p.endswith('.json') and
        re.search(selected,Path(p).stem) and not cached_stop(Path(p).stem))

def run_job(req):
    jid=req['job_id'];job=v3.JOBS/jid;src=job/'src';art=job/'artifacts'
    if job.exists():raise RuntimeError('existing job directory preserved; use a new job ID')
    art.mkdir(parents=True)
    v3.run(['git','worktree','add','--detach',str(src),req['code_commit']],cwd=v3.MIRROR,timeout=120)
    atomic(job/'worker-request.json',req)
    MANAGED.add(jid)
    env=os.environ.copy();env.update(req['env']);env.update(COTS_JOB_ID=jid,COTS_JOB_DIR=str(job),
        COTS_JOB_ARTIFACT_DIR=str(art),COTS_GIT_COMMIT=req['code_commit'],PYTHONUNBUFFERED='1')
    status={**req,'state':'running','started_at':time.time(),'hostname':socket.gethostname(),'bridge_version':VERSION}
    atomic(job/'supervisor-status.json',status)
    with (job/'job.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('cots_bridge_worker.py')),
            str(job/'worker-request.json'),str(src),v3.PYTHON],cwd=src,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status['supervisor_pid']=proc.pid;next_tick=0.
    while proc.poll() is None:
        if time.monotonic()>=next_tick:
            if cached_stop(jid):proc.send_signal(__import__('signal').SIGTERM)
            status.update(updated_at=time.time(),log_tail=v3.tail(job/'job.log',40))
            try:
                atomic(job/'supervisor-status.json',status)
                v3.put_json(f'remote/results/{jid}.json',status,f'supervised {jid}')
                heartbeat=v3.agent_status(jid);heartbeat.update(control_last_sync=LAST_SYNC,outbox_count=len(list(OUTBOX.glob('*.json'))))
                v3.put_json(v3.AGENT_STATUS_PATH,heartbeat,'workstation heartbeat')
                upload_changed(jid)
            except OSError as exc:print('local checkpoint error:',type(exc).__name__,flush=True)
            next_tick=time.monotonic()+30
        time.sleep(1)
    proc.wait()
    exitfile=job/'worker-exit.json'
    exitdata=json.loads(exitfile.read_text()) if exitfile.exists() else {'returncode':None,'wrapper_error':'missing_exit_record'}
    status.update(exitdata)
    status.update(state='timed_out' if exitdata.get('timed_out') else 'stopped' if exitdata.get('stopped') else
        'finished' if exitdata.get('returncode')==0 else 'failed',finished_at=time.time())
    atomic(job/'supervisor-status.json',status)
    enqueue(f'remote/artifacts/{jid}/job.log',(job/'job.log').read_bytes()[-v3.MAX_LOG_UPLOAD_BYTES:],'terminal log')
    upload_changed(jid)
    v3.put_json(f'remote/results/{jid}.json',status,'supervised terminal')
    MANAGED.discard(jid)
    return status

def install_patches():
    v3.VERSION=v4.VERSION=v5.VERSION=VERSION
    v3.put_file=v4.resilient_put_file=enqueue
    v4.flush_pending=lambda:(0,0)
    v4.recover_legacy_orphans=observe_adopted
    v4.queue_paths=queue_paths;v3.exact_commit=cached_exact
    v3.validate_request=validate_request
    v5.already_claimed=cached_claimed
    v4.run_job=run_job

def main():
    if not v3.TOKEN:raise RuntimeError('COTS_GITHUB_TOKEN is not set')
    lock=(v3.STATE/'daemon_v06.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('another v0.6 supervisor is active')
    install_patches();migrate_pending()
    atomic(v3.STATE/'daemon_identity_v06.json',{'pid':os.getpid(),'version':VERSION,
        'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]})
    v3.put_json('remote/health/bridge-v06-start.json',{'bridge_version':VERSION,'timestamp':time.time(),
        'pid':os.getpid(),'hostname':socket.gethostname(),'preserve_existing_workers':True},'bridge v0.6 started')
    threading.Thread(target=publisher,daemon=True).start()
    threading.Thread(target=sync_control,daemon=True).start()
    threading.Thread(target=adoption_loop,daemon=True).start()
    # v0.5's durable claim and immutable final logic are retained. Bootstrap
    # network initialization is unnecessary for an existing configured mirror.
    v3.ensure_repo=lambda:None;v3.ensure_results_branch=lambda:None
    v5.daemon()

if __name__=='__main__':main()
