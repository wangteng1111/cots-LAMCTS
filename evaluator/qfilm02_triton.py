"""Experimental fused Triton ray tracer for Qfilm0.2.

This is *not* a replacement for the validated reference until equivalence tests pass.
One program spans a vector of pupil rays for potentially different fields / colors;
surface intersections and Newton iterations execute entirely on device.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import triton
import triton.language as tl
from evaluator.qfilm02 import _surfaces_at
from evaluator.optical_primitives import front_matrix

@triton.jit
def _sag(r, c, conic, a4, a6, a8, a10):
    rr=r*r
    arg=1.-(1.+conic)*c*c*rr
    sq=tl.sqrt(tl.maximum(arg,1e-14))
    sag=c*rr/(1.+sq)
    ds=c*r/sq
    r4=rr*rr; r6=r4*rr; r8=r4*r4; r10=r8*rr
    sag=sag+a4*r4+a6*r6+a8*r8+a10*r10
    ds=ds+(4.*a4)*r*rr+(6.*a6)*r*rr*rr+(8.*a8)*r*r6+(10.*a10)*r*r8
    return sag,ds,arg>1e-14

@triton.jit
def _step(z,x,y,dz,dx,dy,opl,valid,params,wi,ss,NS:tl.constexpr,USE_OPL:tl.constexpr):
    loc=wi*NS+ss
    zs=tl.load(params+loc*12+0)
    c=tl.load(params+loc*12+1)
    conic=tl.load(params+loc*12+2)
    n1=tl.load(params+loc*12+3)
    n2=tl.load(params+loc*12+4)
    aper=tl.load(params+loc*12+5)
    a4=tl.load(params+loc*12+6)
    a6=tl.load(params+loc*12+7)
    a8=tl.load(params+loc*12+8)
    a10=tl.load(params+loc*12+9)
    t=(zs-z)/tl.maximum(dz,1e-12)
    for _ in range(12):
        xx=x+t*dx; yy=y+t*dy
        r=tl.sqrt(xx*xx+yy*yy)
        sag,ds,ok=_sag(r,c,conic,a4,a6,a8,a10)
        radial=(xx*dx+yy*dy)/tl.maximum(r,1e-14)
        f=z+t*dz-zs-sag
        df=dz-ds*radial
        good=tl.abs(df)>1e-12
        t=t-tl.where(good,f/tl.where(good,df,1.),0.)
        valid=valid & ok & good
    xx=x+t*dx; yy=y+t*dy; zz=z+t*dz
    rr=tl.sqrt(xx*xx+yy*yy)
    sag_final,ds,ok=_sag(rr,c,conic,a4,a6,a8,a10)
    # Consistent with reference: 0.1 nm surface intersection residual.
    residual=zz-zs-sag_final
    valid=valid & ok & (tl.abs(residual)<=1.e-7) & (t>0.) & (t==t) & (tl.abs(t)<1.e16) & (rr<=aper*.5+1e-9)
    nx=-ds*xx/tl.maximum(rr,1e-14)
    ny=-ds*yy/tl.maximum(rr,1e-14)
    nz=tl.full(nx.shape,1.,tl.float64)
    norm=tl.sqrt(nx*nx+ny*ny+nz*nz)
    nx=nx/norm;ny=ny/norm;nz=nz/norm
    dot=nx*dx+ny*dy+nz*dz
    nx=tl.where(dot>0.,-nx,nx);ny=tl.where(dot>0.,-ny,ny);nz=tl.where(dot>0.,-nz,nz)
    eta=n1/n2
    ci=-(nx*dx+ny*dy+nz*dz)
    root2=1.-eta*eta*(1.-ci*ci)
    valid=valid & (root2>=0.)
    k=eta*ci-tl.sqrt(tl.maximum(root2,0.))
    dx2=eta*dx+k*nx;dy2=eta*dy+k*ny;dz2=eta*dz+k*nz
    nn=tl.sqrt(dx2*dx2+dy2*dy2+dz2*dz2)
    dx2=dx2/tl.maximum(nn,1e-14);dy2=dy2/tl.maximum(nn,1e-14);dz2=dz2/tl.maximum(nn,1e-14)
    if USE_OPL:
        opl=opl+n1*tl.sqrt((xx-x)*(xx-x)+(yy-y)*(yy-y)+(zz-z)*(zz-z))
    return zz+1e-9*dz2,xx+1e-9*dx2,yy+1e-9*dy2,dz2,dx2,dy2,opl,valid

@triton.jit
def _at_stop(sx,sy,angcos,angsin,params,wi,ZS:tl.constexpr,ZSTOP:tl.constexpr,NS:tl.constexpr,NFRONT:tl.constexpr):
    z=tl.full(sx.shape,ZS,tl.float64)
    x=sx;y=sy;dz=angcos;dx=angsin;dy=tl.full(sx.shape,0.,tl.float64)
    opl=tl.full(sx.shape,0.,tl.float64)
    valid=tl.full(sx.shape,True,tl.int1)
    for ss in range(NFRONT):
        z,x,y,dz,dx,dy,opl,valid=_step(z,x,y,dz,dx,dy,opl,valid,params,wi,ss,NS,False)
    t=(ZSTOP-z)/tl.maximum(dz,1e-12)
    return x+t*dx,y+t*dy,valid & (t>=0.)

@triton.jit
def _fused_rays(U,V,ANGLES,PARAMS,OUT_X,OUT_Y,OUT_Z,OUT_OPL,OUT_FX,OUT_FY,OUT_VALID,
                NP:tl.constexpr,NF:tl.constexpr,NW:tl.constexpr,NS:tl.constexpr,NFRONT:tl.constexpr,
                ZS:tl.constexpr,ZSTOP:tl.constexpr,STOPR:tl.constexpr,FILM:tl.constexpr,START_A:tl.constexpr,START_B:tl.constexpr,START_D:tl.constexpr,BLOCK:tl.constexpr):
    j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    # Include exactly one additional pupil sample per field/wavelength for chief ray.
    group_size=NP+1
    wi=tl.minimum(j//(NF*group_size),NW-1)
    fi=(j//group_size)%NF
    pi=j%group_size
    mask=j<NW*NF*group_size
    uf=tl.load(U+pi,mask=mask,other=0.)
    vf=tl.load(V+pi,mask=mask,other=0.)
    angle=tl.load(ANGLES+fi,mask=mask,other=0.)
    cosa=tl.cos(angle);sina=tl.sin(angle)
    tarx=STOPR*uf;tary=STOPR*vf
    if NFRONT>0 and tl.abs(START_A)>1e-6:
        sx=(tarx-(START_A*START_D+START_B)*sina/tl.maximum(cosa,1e-12))/START_A
        sy=tary/START_A
    else:
        sx=tarx-(ZSTOP-ZS)*sina/tl.maximum(cosa,1e-12)
        sy=tary
    valid=tl.full((BLOCK,),True,tl.int1)
    if NFRONT>0:
        # One finite-difference Jacobian followed by inverse-Broyden style secant updates.
        # Reduces optical front traces per ray from 6*3+1=19 to 3+6=9.
        fx,fy,v0=_at_stop(sx,sy,cosa,sina,PARAMS,wi,ZS,ZSTOP,NS,NFRONT)
        h=1e-3
        fxx,fyx,vx=_at_stop(sx+h,sy,cosa,sina,PARAMS,wi,ZS,ZSTOP,NS,NFRONT)
        fxy,fyy,vy=_at_stop(sx,sy+h,cosa,sina,PARAMS,wi,ZS,ZSTOP,NS,NFRONT)
        j00=(fxx-fx)/h;j10=(fyx-fy)/h
        j01=(fxy-fx)/h;j11=(fyy-fy)/h
        valid=valid & v0 & vx & vy
        for _ in range(6):
            det=j00*j11-j01*j10
            good=tl.abs(det)>1e-12
            ex=fx-tarx;ey=fy-tary
            stepx=tl.where(good,(j11*ex-j01*ey)/tl.where(good,det,1.),0.)
            stepy=tl.where(good,(-j10*ex+j00*ey)/tl.where(good,det,1.),0.)
            sxnew=sx-stepx;synew=sy-stepy
            fnew,gnew,vnew=_at_stop(sxnew,synew,cosa,sina,PARAMS,wi,ZS,ZSTOP,NS,NFRONT)
            dxq=sxnew-sx;dyq=synew-sy
            df=fnew-fx;dg=gnew-fy
            rx=df-j00*dxq-j01*dyq
            ry=dg-j10*dxq-j11*dyq
            scale=tl.maximum(dxq*dxq+dyq*dyq,1e-24)
            j00=j00+rx*dxq/scale
            j01=j01+rx*dyq/scale
            j10=j10+ry*dxq/scale
            j11=j11+ry*dyq/scale
            valid=valid & good & vnew
            sx=sxnew;sy=synew;fx=fnew;fy=gnew
        valid=valid & ((fx-tarx)*(fx-tarx)+(fy-tary)*(fy-tary)<1e-8)
    z=tl.full((BLOCK,),ZS,tl.float64)
    x=sx;y=sy;dz=cosa;dx=sina;dy=tl.full((BLOCK,),0.,tl.float64)
    opl=dx*sx+dz*z
    for ss in range(NS):
        if ss==NFRONT:
            t=(ZSTOP-z)/tl.maximum(dz,1e-12)
            px=x+t*dx;py=y+t*dy
            valid=valid & (px*px+py*py<=(STOPR+1e-8)*(STOPR+1e-8))
        z,x,y,dz,dx,dy,opl,valid=_step(z,x,y,dz,dx,dy,opl,valid,PARAMS,wi,ss,NS,True)
    t=(FILM-z)/tl.maximum(dz,1e-12)
    valid=valid & (t>0.) & (t==t) & (tl.abs(t)<1e16)
    xf=x+t*dx;yf=y+t*dy
    tl.store(OUT_X+j,x,mask=mask)
    tl.store(OUT_Y+j,y,mask=mask)
    tl.store(OUT_Z+j,z,mask=mask)
    tl.store(OUT_OPL+j,opl,mask=mask)
    tl.store(OUT_FX+j,xf,mask=mask)
    tl.store(OUT_FY+j,yf,mask=mask)
    tl.store(OUT_VALID+j,valid,mask=mask)

def trace_wavelengths(src, waves, angles, uv, stop_z, stop_r, film_z):
    """Return tuple (z,x,y,opl,fx,fy,valid) with shape [wave,field,pupil+chief]."""
    if not angles.is_cuda or not uv.is_cuda:
        raise RuntimeError('Triton tracer needs CUDA tensors')
    device=angles.device
    angles=angles.flatten().contiguous().to(torch.float64)
    uv=uv.to(dtype=torch.float64,device=device).contiguous()
    npup=uv.shape[0]
    uf=torch.cat((uv[:,0],torch.zeros(1,device=device,dtype=torch.float64)))
    vf=torch.cat((uv[:,1],torch.zeros(1,device=device,dtype=torch.float64)))
    surfaces=[_surfaces_at(src,lam) for lam in waves]
    ns=len(surfaces[0]);nf=len(angles);nw=len(waves)
    nfront=sum(s.z<stop_z-1e-9 for s in surfaces[0])
    zstart=min(float(stop_z),surfaces[0][0].z)-20.
    fm=front_matrix(surfaces[0],float(stop_z)) if nfront>0 else None
    A0=float(fm[0,0]) if fm is not None else 1.
    B0=float(fm[0,1]) if fm is not None else 0.
    D0=float(surfaces[0][0].z-zstart)
    arr=np.zeros((nw,ns,12),np.float64)
    for w,surfs in enumerate(surfaces):
        for s,sf in enumerate(surfs):
            ap=getattr(sf,'clear_aperture',None)
            asp=tuple(sf.asphere)
            if len(asp)>4 and any(abs(x)>0 for x in asp[4:]):
                raise ValueError('Triton v1 supports up to A10 asphere')
            arr[w,s,:]=[sf.z, 0. if math.isinf(sf.R) else 1./sf.R,
                        sf.conic,sf.n1,sf.n2,1e16 if ap is None else float(ap),
                        *(list(asp[:4])+[0.]*(4-min(4,len(asp)))),0.,0.]
    par=torch.as_tensor(arr.ravel().copy(),device=device,dtype=torch.float64)
    n=nw*nf*(npup+1)
    out=[torch.empty(n,dtype=torch.float64,device=device) for _ in range(6)]
    valid=torch.empty(n,dtype=torch.bool,device=device)
    _fused_rays[(triton.cdiv(n,128),)](uf,vf,angles,par,*out,valid,
        npup,nf,nw,ns,nfront,zstart,float(stop_z),float(stop_r),float(film_z),A0,B0,D0,128,num_warps=4)
    # Kernel output order is X,Y,Z,OPL,FX,FY. Public API is Z,X,Y,OPL,FX,FY.
    return tuple(t.reshape(nw,nf,npup+1) for t in (out[2],out[0],out[1],out[3],out[4],out[5],valid))