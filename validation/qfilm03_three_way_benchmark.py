#!/usr/bin/env python3
"""Qfilm0.3 three-way GPU acceptance: physical lens, shared wavefield, TorchOptics propagation.

The full-lens comparison is Qfilm fast vs DeepLens Huygens.
TorchOptics is not a refractive-lens tracer.  To make an apples-to-apples
propagation benchmark, DeepLens ASM and TorchOptics ASM receive EXACTLY the
same complex exit-pupil field at the same wavelength, pitch and propagation z.
TorchOptics DIM is separately checked on an identical small converging field.
All results written as machine-readable JSON.  No physical label certification.
"""
from __future__ import annotations
import sys,os,time,json,types,pathlib,math,traceback,statistics
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,BK7,mtf_from_psf
from evaluator.qfilm02 import Config,evaluate as ref_eval
from evaluator.qfilm03 import ValidationConfig,evaluate as q03

TO_PATH=pathlib.Path("/var/lib/cots-lamcts/torchoptics_upstream/src/torchoptics")
TO_SHA="34fe9c40f874e53db225e703c5580befe8c523b5"
DEEP_SHA="911bb9d334fce7e20a4407c4506ac683b52ff246"
OUT=pathlib.Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","/tmp"))/"qfilm03_benchmark"
OUT.mkdir(parents=True,exist_ok=True)

def load_torchoptics_field():
    # TorchOptics currently imports its matplotlib visualization when importing
    # torchoptics.Fields. On this workstation the Ubuntu matplotlib wheel has
    # incompatible NumPy-1 ABI. Import real numerical modules with only visual
    # modules shimmed; NO physical propagation source code is changed.
    from importlib import import_module
    if not (TO_PATH/"fields.py").is_file():raise FileNotFoundError("TorchOptics upstream not bootstrapped")
    pkg=sys.modules.get("torchoptics")
    if pkg is None:
        pkg=types.ModuleType("torchoptics")
        pkg.__path__=[str(TO_PATH)]
        pkg.__package__="torchoptics"
        sys.modules["torchoptics"]=pkg
    if "torchoptics.visualization" not in sys.modules:
        vis=types.ModuleType("torchoptics.visualization")
        def no_fig(*args,**kwargs):raise RuntimeError("Matplotlib display unavailable on this machine")
        vis.visualize_tensor=no_fig;vis.animate_tensor=no_fig
        sys.modules["torchoptics.visualization"]=vis
    fg=sys.modules.get("matplotlib.figure")
    if fg is None:
        fg=types.ModuleType("matplotlib.figure")
        fg.Figure=type("Figure",(),{})
        sys.modules["matplotlib.figure"]=fg
    Field=import_module("torchoptics.fields").Field
    return Field

def time_gpu(call):
    torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode():res=call()
    torch.cuda.synchronize()
    return res,float(time.perf_counter()-t)

def crop_center(im,ks):
    n,m=im.shape[-2:]
    return im[...,n//2-ks//2:n//2-ks//2+ks,m//2-ks//2:m//2-ks//2+ks]

def norm_intensity(field,ks=None):
    p=field.abs().square()
    if ks is not None:p=crop_center(p,ks)
    p=p/p.sum().clamp(min=1.e-30)
    return p

def shared_deeplens_pupil_field(Field,device="cuda:0",sensor_res=1024,spp=1000000):
    from deeplens import GeoLens
    from deeplens.light import AngularSpectrumMethod
    import tempfile
    p=thin_lens()
    p["surfaces"][0]["sellmeier"]=BK7
    optical,_=conversion(p,sensor_size=12.8,sensor_res=sensor_res,material_mode="bk7")
    with tempfile.TemporaryDirectory() as tmp:
        path=pathlib.Path(tmp)/"lens.json"
        path.write_text(json.dumps(optical))
        lens=GeoLens(filename=str(path),device=device,dtype=torch.float64)
        wave,psfc= time_gpu(lambda:lens.pupil_field(
            points=[0.,0.,-1000000.],wvln=0.55,spp=spp,recenter=True))
        w0=w[0].contiguous()
        # Physical pupil field has shape [H,H]; same zero pad and same pixel pitch as DeepLens ASM.
        pd=torch.nn.functional.pad(w0,(sensor_res//2,)*4,mode="constant",value=0)
        z=float(lens.d_sensor)-float(lens.get_exit_pupil()[0])
        pitch=float(lens.pixel_size)
        fw=Field(pd,wavelength=0.00055,z=0,spacing=(pitch,pitch)).to(device)
        # Prewarm both methods, then repeat interleaved to remove cold overhead.
        def deep():return AngularSpectrumMethod(pd[None,None],z=z,wvln=0.55,ps=pitch,padding=False).squeeze()
        def opt():return fw.propagate_to_z(z,propagation_method="ASM",asm_pad=(0,0)).data
        deep0,_=time_gpu(deep)
        torch0,_=time_gpu(opt)
        dd=[];tt=[]
        for _ in range(3):
            a,ts=time_gpu(deep);dd.append(ts);b,ts=time_gpu(opt);tt.append(ts)
        dpsf=norm_intensity(a,ks=96);tpsf=norm_intensity(b,ks=96)
        direct_diff=(dpsf-tpsf).abs()
        fx,fy=mtf_from_psf(dpsf.cpu().numpy(),pitch,(5.,10.,20.,40.,60.,80.))
        tx,ty=mtf_from_psf(tpsf.cpu().numpy(),pitch,(5.,10.,20.,40.,60.,80.))
        max_mtf=max(np.max(np.abs(np.asarray(fx)-tx)),np.max(np.abs(np.asarray(fy)-ty)))
        return {"input":"identical DeepLens exit-pupil complex field","pupil_field_sample_spp":spp,
                "pupil_field_grid":sensor_res,"propagation_grid":2*sensor_res,
                "physical_z_mm":z,"pixel_pitch_mm":pitch,"wavelength_mm":0.00055,
                "output_psf_patch":96,"interpolation":"none",
                "deeplens_asm_median_s":statistics.median(dd),
                "torchoptics_asm_median_s":statistics.median(tt),
                "psf_max_abs_diff":float(direct_diff.max()),
                "psf_mean_abs_diff":float(direct_diff.mean()),
                "mtf_max_abs_diff":float(max_mtf),
                "deeplens_mtf_x":fx,"torchoptics_mtf_x":tx,
                "deeplens_mtf_y":fy,"torchoptics_mtf_y":ty,
                "physically_equivalent_inputs":True}

def crosscheck_asm_dim(Field,device="cuda:0",n=96):
    # Same input complex converging scalar wave; no lens prescription or ray tracing.
    # Grid support 0.48 mm with a 0.12mm circular aperture, z=6mm focal distance.
    lam=.00055;ps=.005;z=6.0;radius=.12
    k=2*math.pi/lam
    x=(torch.arange(n,device=device,dtype=torch.float64)+.5-n/2)*ps
    yy,xx=torch.meshgrid(x,x,indexing="ij")
    rr=torch.sqrt(xx.square()+yy.square())
    phase=-k*(torch.sqrt(z*z+rr.square())-z)
    wave=torch.where(rr<=radius,torch.polar(torch.ones_like(phase),phase),
                        torch.zeros_like(phase,dtype=torch.complex128))
    f=Field(wave.contiguous(),wavelength=lam,z=0.,spacing=(ps,ps)).to(device)
    def run(method):
        return f.propagate_to_z(z,propagation_method=method,
                                asm_pad=(n//2,n//2),interpolation_mode="none").data
    # TorchOptics returns padded support for ASM and original support for DIM:
    # crop to equivalent physical output window.
    results={}
    for method in ("ASM","DIM"):
        try:
            v,_=time_gpu(lambda:run(method))
            ss=[]
            for _ in range(3):
                v,t=time_gpu(lambda:run(method));ss.append(t)
            p=norm_intensity(v,ks=64)
            results[method]={"seconds_median":statistics.median(ss),
                             "psf":p.cpu().numpy(),"finite":bool(torch.isfinite(p).all())}
        except Exception as e:
            results[method]={"error":repr(e),"traceback":traceback.format_exc()[-2000:]}
    result={"field_n":n,"input_pitch_mm":ps,"wavelength_mm":lam,"distance_mm":z,
            "aperture_radius_mm":radius,"methods":{}}
    for name,d in results.items():
        result["methods"][name]={k:v for k,v in d.items() if k!="psf"}
    if all("psf" in d for d in results.values()):
        a=results["ASM"]["psf"];b=results["DIM"]["psf"]
        ma,mb=mtf_from_psf(a,ps,(5.,10.,20.,40.)),mtf_from_psf(b,ps,(5.,10.,20.,40.))
        result["psf_max_abs_diff"]=float(np.max(np.abs(a-b)))
        result["mtf_max_abs_diff"]=float(max(np.max(np.abs(np.asarray(ma[0])-mb[0])),np.max(np.abs(np.asarray(ma[1])-mb[1]))))
    return result

def main():
    records={"schema":"qfilm03-physics-3way-v1",
             "deep_source":DEEP_SHA,"torchoptics_source":TO_SHA,
             "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"cases":{}}
    report=OUT/"three_way.json"
    def write():report.write_text(json.dumps(records,indent=2)+"\n")
    for name in ("thin","internal_positive"):
        p=thin_lens() if name=="thin" else internal_stop_positive_lens()
        if name=="thin":p["surfaces"][0]["sellmeier"]=BK7
        cfg=Config(fields=3,pupil_n=128,fft_n=512,max_fields_per_chunk=3,
                   wavelengths_nm=(550.,),spectral_weights=(1.,),
                   frame_width_mm=8.,frame_height_mm=6.,
                   target_efl_mm=float(p["design_spec"]["efl_target_mm"]))
        validation=ValidationConfig(mode="verify",fields=(0.,.5,1.),
                  wavelength_nm=550.,huygens_spp=1000000,
                  sensor_resolution=3072,psf_kernel=96,base_backend="torch")
        r0,s0=time_gpu(lambda:ref_eval(p,cfg,device="cuda:0"))
        r3,s3=time_gpu(lambda:q03(p,cfg,device="cuda:0",validation=validation))
        records["cases"][name]={"baseline_seconds":s0,
           "qfilm03_total_seconds":s3,"fast_seconds":r3["timings"]["qfilm_seconds"],
           "reference_seconds":r3["timings"].get("reference_seconds"),
           "max_Q_image_abs":float(np.max(np.abs(r0["Q_image"]-r3["Q_image"]))),
           "J_diff":abs(float(r0["quality"]["J_task"])-float(r3["quality"]["J_task"])),
           "qfilm03_version":r3["version"],"status":r3["reference"]["status"],
           "max_abs_mtf_discrepancy":r3["reference"].get("mtf_max_abs"),
           "per_field_discrepancy":[x["mtf_max_abs"] for x in r3["reference"].get("fields",[])],
           "physics_certified":r3["certified"],"training_eligible":r3["training_eligible"],
           "efl_delta":r3["reference"].get("efl_abs_difference_mm"),
           "error":r3["reference"].get("error")}
        print("QFILM03",name,json.dumps(records["cases"][name]),flush=True);write()
    try:
        Field=load_torchoptics_field()
        records["torchoptics_same_wavefront"]=shared_deeplens_pupil_field(Field)
        print("SAME_WAVEFIELD",json.dumps(records["torchoptics_same_wavefront"]),flush=True);write()
        records["torchoptics_asm_dim"]=crosscheck_asm_dim(Field)
        print("ASM_DIM",json.dumps(records["torchoptics_asm_dim"]),flush=True);write()
    except Exception as e:
        records["torchoptics_error"]={"error":repr(e),"traceback":traceback.format_exc()[-4500:]}
        print("TORCHOPTICS_ERROR",repr(e),flush=True);write()
    records["passed_qfilm03"]=all(c["status"]=="checked_subset" and c["max_Q_image_abs"]<1.e-7
                                    for c in records["cases"].values())
    records["passed_torchoptics"]=("torchoptics_same_wavefront" in records and
                                  np.isfinite(records["torchoptics_same_wavefront"]["mtf_max_abs_diff"]))
    write()
    return 0 if records["passed_qfilm03"] and records["passed_torchoptics"] else 2
if __name__=="__main__":raise SystemExit(main())
