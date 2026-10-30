#!/usr/bin/env python3
"""Stage, test and activate v0.6 without killing existing computation children.

Staging works under the existing service sandbox. Activation requires root;
when unavailable, emit the exact staged command rather than bypassing security.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def run(argv,**kwargs):return subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,**kwargs)
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)

def activate():
    if os.geteuid()!=0:raise SystemExit('Root activation required; no privilege bypass attempted')
    runtime=Path(__file__).resolve().parents[1]
    manifest=json.loads((runtime/'manifest.json').read_text())
    for name,sha in manifest['files_sha256'].items():
        file=(runtime/name).resolve()
        if runtime not in file.parents or hashlib.sha256(file.read_bytes()).hexdigest()!=sha:raise SystemExit('staged checksum mismatch')
    result=run([sys.executable,str(runtime/'scripts/test_bridge_v06.py')],timeout=180)
    if result.returncode:raise SystemExit(result.stdout)
    drop=Path('/etc/systemd/system/cots-lamcts-bridge.service.d/zz-cots-bridge-v06.conf')
    drop.parent.mkdir(parents=True,exist_ok=True)
    backup=runtime/'activation-backup';backup.mkdir(exist_ok=True)
    if drop.exists():shutil.copy2(drop,backup/drop.name)
    content=('[Service]\nExecStart=\nExecStart=/usr/bin/python3 '+str(runtime/'infra/cots_workstation_bridge_v06.py')+
        '\nWorkingDirectory='+str(runtime/'infra')+'\nKillMode=process\n'+
        'Environment="COTS_BRIDGE_QUEUE_INCLUDE=^000-000-020-qfilmrw-"\n')
    temp=drop.with_suffix('.tmp');temp.write_text(content);os.replace(temp,drop)
    run(['systemctl','daemon-reload'],check=True)
    settings=run(['systemctl','show','cots-lamcts-bridge.service','--property=KillMode,ExecStart'],check=True).stdout
    if 'KillMode=process' not in settings or str(runtime/'infra/cots_workstation_bridge_v06.py') not in settings:
        raise SystemExit('systemd override precedence mismatch; not restarting')
    # KillMode=process preserves all existing workers in the service cgroup.
    run(['systemctl','restart','cots-lamcts-bridge.service'],check=True)
    print(run(['systemctl','show','cots-lamcts-bridge.service','--property=ActiveState,SubState,MainPID,KillMode']).stdout)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--activate',action='store_true');a=ap.parse_args()
    if a.activate:return activate()
    art=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR',str(ROOT/'artifacts/bridge_v06_stage')));art.mkdir(parents=True,exist_ok=True)
    bridge_root=Path(os.environ.get('COTS_BRIDGE_ROOT','/var/lib/cots-lamcts'))
    stamp=os.environ.get('COTS_GIT_COMMIT','local')[:12]
    runtime=bridge_root/'runtime'/('bridge-v06-'+stamp);runtime.mkdir(parents=True,exist_ok=True)
    files=['infra/cots_workstation_bridge.py','infra/cots_workstation_bridge_v04.py','infra/cots_workstation_bridge_v05.py',
        'infra/cots_workstation_bridge_v06.py','infra/cots_bridge_worker.py','infra/qfilm_convergence_status.py',
        'scripts/deploy_bridge_v06.py','scripts/test_bridge_v06.py']
    report={'schema':'Bridge06Deployment-v1','started_unix':time.time(),'status':'staging','runtime':str(runtime),'uid':os.getuid(),
        'code_commit':os.environ.get('COTS_GIT_COMMIT'),'activation_requires_root':True,'existing_workers_terminated':False}
    for name in files:
        target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    manifest={'files_sha256':{name:hashlib.sha256((runtime/name).read_bytes()).hexdigest() for name in files}}
    write(runtime/'manifest.json',manifest)
    test=run([sys.executable,str(runtime/'scripts/test_bridge_v06.py')],timeout=180)
    (art/'bridge-v06-tests.log').write_text(test.stdout);report['tests_returncode']=test.returncode
    if test.returncode:
        report['status']='tests_failed';write(art/'bridge-v06-deployment.json',report);return 2
    # Preserve all published-type native checkpoints before any activation.
    backup=runtime/'checkpoint-backup';saved=[]
    for job in (bridge_root/'jobs').glob('*qfilm*'):
        for p in (job/'artifacts').rglob('*.json'):
            if p.is_file() and p.stat().st_size<=20_000_000:
                dest=backup/job.name/p.relative_to(job/'artifacts');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
                saved.append({'job_id':job.name,'path':str(p.relative_to(job/'artifacts')),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    write(art/'checkpoint-backup-manifest.json',{'files':saved,'originals_retained':True})
    sys.path.insert(0,str(runtime/'infra'))
    from qfilm_convergence_status import collect
    write(art/'three-case-verification.json',collect(bridge_root/'jobs'))
    report['systemd_before']=run(['systemctl','show','cots-lamcts-bridge.service','--property=ActiveState,MainPID,KillMode,NoNewPrivileges,ProtectSystem']).stdout
    report['checkpoints_backed_up']=len(saved)
    report['activation_command']='sudo /usr/bin/python3 '+str(runtime/'scripts/deploy_bridge_v06.py')+' --activate'
    sudo=run(['sudo','-n','true']) if shutil.which('sudo') else None
    report['sudo_noninteractive']=sudo is not None and sudo.returncode==0
    report['status']='staged_requires_host_activation'
    write(art/'bridge-v06-deployment.json',report)
    if report['sudo_noninteractive']:
        # Delay so the old Bridge can return this staging job first. Only its
        # main process will be replaced; computation children are preserved.
        schedule=run(['sudo','-n','systemd-run','--unit=cots-bridge-v06-activate-'+stamp,'--on-active=20s',
                      '/usr/bin/python3',str(runtime/'scripts/deploy_bridge_v06.py'),'--activate'])
        report['activation_schedule_returncode']=schedule.returncode
        if schedule.returncode==0:report['status']='activation_scheduled'
        write(art/'bridge-v06-deployment.json',report)
    print(json.dumps(report),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
