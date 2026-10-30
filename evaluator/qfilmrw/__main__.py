import argparse,json
from pathlib import Path
import numpy as np
from . import evaluate,Task,Policy
from evaluator.qfilm04_io import atomic_json

def main():
    ap=argparse.ArgumentParser(description='QfilmRW ray-wave evaluator; experimental local-Fourier model')
    be=ap.add_mutually_exclusive_group(required=True);be.add_argument('-c',action='store_true');be.add_argument('-g',action='store_true')
    ap.add_argument('--prescription',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fields',type=int,default=3);ap.add_argument('--gpu-index',type=int,default=0)
    ap.add_argument('--survey',action='store_true');ap.add_argument('--max-pupil',type=int,default=1024);ap.add_argument('--grid-n',type=int,default=1024)
    a=ap.parse_args()
    if a.fields<2 or a.max_pupil<256:ap.error('at least 2 fields and max-pupil >=256 required')
    levels=[];n=128
    while n<=a.max_pupil:levels.append(n);n*=2
    p=json.loads(a.prescription.read_text())
    def checkpoint(d):
        plan=d['plan'];atomic_json(a.output.with_suffix('.checkpoints')/f"h{plan['field_h']:g}-w{plan['wavelength_nm']:g}-terminal.json",d,immutable=True)
    result=evaluate(p,fields=tuple(np.linspace(0,1,a.fields)),policy=Policy(pupils=tuple(levels),angular_grid_n=a.grid_n),
        backend='c' if a.c else 'g',gpu_index=a.gpu_index,mode='survey' if a.survey else 'verify',checkpoint=checkpoint)
    atomic_json(a.output,result)
    print(json.dumps({'version':result['variant'],'J_task':result['J_task'],'condition_count':len(result['conditions']),'elapsed_seconds':result['elapsed_seconds'],'training_eligible':False}))
if __name__=='__main__':main()
