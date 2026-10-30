"""Reference-sphere local Fourier model, inverse-map field reconstruction, CZT.

The source-dependent Hessian and higher spherical terms are NOT exact here.
The direct RS1 probe is mandatory; it is a discrepancy estimate, not a global
error proof. No general thick-lens accuracy claim follows from a passed probe.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .rays import RWError,mapping_derivatives

@torch.inference_mode()
def czt_axis(data,m,q0,dq,x0,dx):
    """Sum_j data[j] exp(-2pi i (q0+j*dq)*(x0+k*dx)), last axis.

    Bluestein convolution uses unit-modulus float64 chirps, complex128 FFTs.
    Physical cell-center origins are explicit; no implicit fftshift is used.
    """
    n=data.shape[-1];length=1<<(n+m-2).bit_length();dev=data.device
    j=torch.arange(n,device=dev,dtype=torch.float64)
    k=torch.arange(m,device=dev,dtype=torch.float64)
    theta=-2*math.pi*dq*dx
    polar=lambda p:torch.polar(torch.ones_like(p),p)
    a=data*polar(theta*j*j/2-2*math.pi*dq*x0*j)
    b=torch.zeros(length,dtype=torch.complex128,device=dev)
    b[:m]=polar(-theta*k*k/2)
    if n>1:
        t=torch.arange(-n+1,0,device=dev,dtype=torch.float64);b[-n+1:]=polar(-theta*t*t/2)
    out=torch.fft.ifft(torch.fft.fft(a,n=length)*torch.fft.fft(b),n=length)[...,:m]
    return out*polar(theta*k*k/2-2*math.pi*q0*(x0+k*dx))


def fixed_sampling_field(grid,q0,dq,pixel_n,pitch_um):
    pitch=pitch_um*.001;x0=-(pixel_n-1)*pitch/2
    a=czt_axis(grid,pixel_n,q0[0],dq[0],x0,pitch)
    return czt_axis(a.T,pixel_n,q0[1],dq[1],x0,pitch).T

@torch.inference_mode()
def reference_source(state):
    lam=state.wavelength_nm*1e-6;L=state.film_z_mm-state.exit_z_mm
    dx=state.center_mm[0]-state.x;dy=state.center_mm[1]-state.y
    r=torch.sqrt(L*L+dx*dx+dy*dy)
    qx=-dx/(lam*r);qy=-dy/(lam*r)
    path=state.opl+r
    piston=float(path[state.active][0])
    phase=(path-piston)*(2*math.pi/lam)
    # Exact RS1 prefactor evaluated at the reference sensor point.
    coefficient=state.amplitude_area*(L/(lam*r*r))*(1+1j*lam/(2*math.pi*r))*torch.polar(torch.ones_like(phase),phase)
    coefficient=torch.where(state.active,coefficient,0j)
    residual=0.
    for dim in (0,1):
        a=[slice(None)]*2;b=a.copy();a[dim]=slice(1,None);b[dim]=slice(None,-1);a,b=tuple(a),tuple(b)
        good=state.active[a]&state.active[b]
        if bool(good.any()):residual=max(residual,float((path[a]-path[b])[good].abs().max())/lam)
    return qx,qy,coefficient,piston,residual

@torch.inference_mode()
def reconstruct_pupil(state,grid_n):
    """Reconstruct density in direction-cosine frequency coordinates.

    This is an inverse map, not forward splatting. Interpolation never joins
    across an invalid ray stencil. q-Jacobian converts coefficient quadrature
    into a density exactly once. Newton nonconvergence stays outside the pupil.
    """
    qx,qy,coefficient,piston,phase_guard=reference_source(state)
    n=state.x.shape[0];du=2/n
    derivatives,jac,mask,folded=mapping_derivatives(qx,qy,state.active,du)
    qxu,qxv,qyu,qyv=derivatives
    if folded:raise RWError('UNSUPPORTED_ANGULAR_MAPPING','direction-cosine map folds within a connected pupil component')
    if int(mask.sum())<32:raise RWError('SOURCE_SAMPLING_FAILURE','angular map under-resolved')
    density=torch.where(mask,coefficient/(jac.abs()*du**2).clamp_min(1e-300),0j)
    padding=(max(float(qxu[mask].abs().max()),float(qxv[mask].abs().max()))*du, max(float(qyu[mask].abs().max()),float(qyv[mask].abs().max()))*du)
    mins=[float(v[mask].min())-pad for v,pad in zip((qx,qy),padding)];maxs=[float(v[mask].max())+pad for v,pad in zip((qx,qy),padding)]
    dq=tuple((b-a)/(grid_n-4) for a,b in zip(mins,maxs))
    if min(dq)<=0:raise RWError('UNSUPPORTED_ANGULAR_MAPPING','collapsed angular support')
    q0=tuple(a-1.5*d for a,d in zip(mins,dq))
    tx=q0[0]+torch.arange(grid_n,device=qx.device,dtype=torch.float64)*dq[0]
    ty=q0[1]+torch.arange(grid_n,device=qx.device,dtype=torch.float64)*dq[1]
    yy,xx=torch.meshgrid(ty,tx,indexing='ij')
    # Least-squares affine initialization of q(u,v) over transmitted rays.
    u=(torch.arange(n,device=qx.device,dtype=torch.float64)+.5)*du-1
    vv,uu=torch.meshgrid(u,u,indexing='ij');basis=torch.stack((uu[mask],vv[mask],torch.ones_like(uu[mask])),dim=1)
    fit=torch.linalg.lstsq(basis,torch.stack((qx[mask],qy[mask]),1)).solution
    inv=torch.linalg.inv(fit[:2,:].T)
    delta=torch.stack((xx-fit[2,0],yy-fit[2,1]),-1)
    coords=delta@inv.T
    # Newton differentiates the interpolated coordinate map itself. The
    # one-sided, masked Jacobian above belongs ONLY to physical field density;
    # zeroing its outside entries would give Newton a discontinuous derivative.
    nxv,nxu=torch.gradient(qx,spacing=(du,du));nyv,nyu=torch.gradient(qy,spacing=(du,du))
    stack=torch.stack((qx,qy,nxu,nxv,nyu,nyv,mask.double(),density.real,density.imag))[None]
    stack=torch.nan_to_num(stack,nan=0.,posinf=0.,neginf=0.)
    def coordinate_map(uv):
        # Exact derivative of the bilinear coordinate interpolant. Interpolating
        # finite-difference derivatives is inaccurate next to clipped samples.
        pos=(uv+1)*n/2-.5
        ix=torch.floor(pos[...,0]).long().clamp(0,n-2);iy=torch.floor(pos[...,1]).long().clamp(0,n-2)
        wx=pos[...,0]-ix;wy=pos[...,1]-iy
        values=[];gradients=[]
        for q in (stack[0,0],stack[0,1]):
            a=q[iy,ix];b=q[iy,ix+1];c=q[iy+1,ix];d=q[iy+1,ix+1]
            values.append(a*(1-wx)*(1-wy)+b*wx*(1-wy)+c*(1-wx)*wy+d*wx*wy)
            gradients.extend((((b-a)*(1-wy)+(d-c)*wy)*n/2,((c-a)*(1-wx)+(d-b)*wx)*n/2))
        return (*values,*gradients)
    for _ in range(16):
        qxx,qyy,a,b,c,d=coordinate_map(coords);det=a*d-b*c
        ex=qxx-xx;ey=qyy-yy
        safe=torch.where(det.abs()>1e-14,det,1.)
        step=torch.stack(((d*ex-b*ey)/safe,(-c*ex+a*ey)/safe),-1).clamp(-.2,.2)
        coords=(coords-step).clamp(-1.5,1.5)
    sampled=F.grid_sample(stack,coords[None],mode='bilinear',padding_mode='border',align_corners=False)[0]
    err=torch.maximum((sampled[0]-xx).abs()/dq[0],(sampled[1]-yy).abs()/dq[1])
    inside=(coords.abs().amax(-1)<1-1/n)&(sampled[6]>=.5)
    good=inside & (err<1e-5)
    bad_count=int((inside & ~good).sum())
    bad_interior=int((inside & ~good & (sampled[6]>1-1e-9)).sum())
    unsupported_queries=0
    # A clipped boundary cell can contain an inverse root outside the known
    # transmitted domain. Exclude it and report its area; pupil refinement
    # tests the resulting aperture quadrature. Interior root failures remain
    # errors. No claim of a rigorous boundary-energy bound is made.
    if bad_interior>max(8,int(inside.sum())*.001):
        # Robust fallback only for the few roots that did not converge. A
        # simplex may use only neighboring source cells; no bridging clipped
        # holes, disconnected pupils, or multi-valued branches is permitted.
        from scipy.spatial import Delaunay
        failed=inside & ~good
        ids=torch.nonzero(mask.flatten()).flatten().cpu().numpy()
        points=torch.stack((qx[mask],qy[mask]),-1).cpu().numpy()
        tri=Delaunay(points)
        query=torch.stack((xx[failed],yy[failed]),-1).cpu().numpy()
        simplex=tri.find_simplex(query);sel=np.maximum(simplex,0)
        vertices=tri.simplices[sel];native=ids[vertices]
        neighboring=(np.ptp(native//n,axis=1)<=1)&(np.ptp(native%n,axis=1)<=1)&(simplex>=0)
        unsupported_queries=int((~neighboring).sum())
        transform=tri.transform[sel];bary=np.einsum('nij,nj->ni',transform[:,:2,:],query-transform[:,2,:])
        bary=np.column_stack((bary,1-bary.sum(1)))
        weights=torch.as_tensor(bary,device=qx.device,dtype=torch.float64)
        source_ids=torch.as_tensor(native,device=qx.device,dtype=torch.long)
        recovered=(density.flatten()[source_ids]*weights).sum(-1)
        values=torch.complex(sampled[7],sampled[8])/sampled[6].clamp_min(.5)
        values[failed]=torch.where(torch.as_tensor(neighboring,device=qx.device),recovered,0j)
        recovered_mask=torch.zeros_like(good);recovered_mask[failed]=torch.as_tensor(neighboring,device=qx.device)
        # Queries outside the hull, or in triangles that bridge a clipped
        # source gap, have no supported transmitted patch. They are outside
        # this discretized pupil, not failed roots inside its known domain.
        inside[failed]=torch.as_tensor(neighboring,device=qx.device)
        # Remaining roots are explicitly excluded and stay in the ledger.
        good |= recovered_mask
        bad_count=int((inside & ~good).sum())
        bad_interior=int((inside & ~good & (sampled[6]>1-1e-9)).sum())
        if bad_interior>max(8,int(inside.sum())*.001):raise RWError('FIELD_RECONSTRUCTION_FAILURE',f'interior inverse-map residual exceeds tolerance: {bad_interior}/{int(inside.sum())} cells')
    else:values=torch.complex(sampled[7],sampled[8])/sampled[6].clamp_min(.5)
    grid=torch.where(good,values*dq[0]*dq[1],0j)
    return {'grid':grid,'q0':q0,'dq':dq,'piston_mm':piston,'residual_phase_neighbor_cycles':phase_guard,
            'active_angular_cells':int(good.sum()),'inverse_map_failed_cells':bad_count,
            'excluded_boundary_angular_area':(bad_count-bad_interior+unsupported_queries)*dq[0]*dq[1],
            'unsupported_source_patch_queries':unsupported_queries,
            'inverse_map_failed_interior_cells':bad_interior,
            'density_convention':'RS1 reference-point coefficient / angular cell area, integrated once',
            'max_unaliased_half_window_mm':min(.5/d for d in dq)}

@torch.inference_mode()
def windowed_observables(field,pitch_um,frequencies,retain=False):
    intensity=field.abs().square();n=field.shape[0];pitch=pitch_um*.001
    px=intensity.sum(0);py=intensity.sum(1);total=px.sum()
    if not bool(torch.isfinite(total)) or float(total)<=0:raise RWError('NUMERICAL_FAILURE','nonpositive/nonfinite intensity')
    coords=(torch.arange(n,device=field.device,dtype=torch.float64)-(n-1)/2)*pitch
    freqs=torch.as_tensor(frequencies,device=field.device,dtype=torch.float64)
    kernel=torch.exp(-2j*math.pi*freqs[:,None]*coords[None,:])
    otf=torch.stack((kernel@px.to(torch.complex128),kernel@py.to(torch.complex128)),-1)/total
    border=coords.abs()>=n*pitch/2-.020
    border_flux=(intensity[border].sum()+intensity[:,border].sum()-intensity[border][:,border].sum())/total
    peak=int(torch.argmax(intensity))
    out={'otf':otf.cpu().numpy(),'raw_integral':float(total)*pitch*pitch,'raw_projection_x':px.cpu().numpy(),
         'raw_projection_y':py.cpu().numpy(),'border_20um_fraction':float(border_flux),'peak_yx':(peak//n,peak%n),
         'pixel_n':n,'pitch_um':pitch_um,'support_mm':n*pitch,'otf_domain':'windowed','tail_bound':None}
    if retain:out['intensity']=intensity.cpu().numpy()
    return out

@torch.inference_mode()
def pupil_autocorrelation(pupil,frequencies):
    """Full-plane local-model OTF via overlap quadrature at requested shifts.

    This observable has a separate objective version. A cropped PSF OTF cannot
    be replaced with it unless an independent outside-energy bound is supplied.
    """
    g=pupil['grid'];norm=g.abs().square().sum();out=[]
    for freq in frequencies:
        axes=[]
        for axis,spacing in ((1,pupil['dq'][0]),(0,pupil['dq'][1])):
            shift=freq/spacing;lo=int(math.floor(shift));frac=shift-lo;v=0j
            for offset,weight in ((lo,1-frac),(lo+1,frac)):
                if offset>=g.shape[axis] or weight==0:continue
                if offset==0:term=norm
                elif axis==1:term=(g[:,:-offset]*g[:,offset:].conj()).sum()
                else:term=(g[:-offset]*g[offset:].conj()).sum()
                v=v+weight*term
            axes.append(complex(v/norm))
        out.append(axes)
    return np.asarray(out)

@torch.inference_mode()
def direct_rs1(state,offsets_mm,piston_mm,source_block=65536):
    """Sparse exact-distance reference for the SAME boundary/transport state."""
    active=state.active;x=state.x[active];y=state.y[active];opl=state.opl[active];a=state.amplitude_area[active]
    xy=torch.as_tensor(offsets_mm,dtype=torch.float64,device=x.device)
    L=state.film_z_mm-state.exit_z_mm;lam=state.wavelength_nm*1e-6
    result=torch.zeros(len(xy),device=x.device,dtype=torch.complex128)
    for first in range(0,len(x),source_block):
        sl=slice(first,first+source_block)
        dx=xy[:,0,None]+state.center_mm[0]-x[None,sl];dy=xy[:,1,None]+state.center_mm[1]-y[None,sl]
        r=torch.sqrt(L*L+dx*dx+dy*dy)
        phase=(opl[None,sl]+r-piston_mm)*(2*math.pi/lam)
        coeff=a[None,sl]*(L/(lam*r*r))*(1+1j*lam/(2*math.pi*r))
        result+=(torch.exp(1j*phase)*coeff).sum(-1)
    return result

@torch.inference_mode()
def reference_probe(state,pupil,field,pitch_um,peak_yx):
    n=field.shape[0];c=(n-1)/2;pitch=pitch_um*.001
    # Center, core, peak, intermediate ring and corners. Sparse checks cannot
    # certify unsampled pixels or an omitted tail, even if all checks pass.
    ids={(n//2,n//2),tuple(peak_yx)}
    for d in (2,8,max(1,n//16),max(1,n//4)):
        for dy,dx in ((0,d),(d,0),(0,-d),(-d,0)):
            ids.add((max(0,min(n-1,n//2+dy)),max(0,min(n-1,n//2+dx))))
    ids.update(((0,0),(0,n-1),(n-1,0),(n-1,n-1)))
    ids=sorted(ids);xy=[((ix-c)*pitch,(iy-c)*pitch) for iy,ix in ids]
    exact=direct_rs1(state,xy,pupil['piston_mm'])
    approx=torch.stack([field[iy,ix] for iy,ix in ids])
    # Common reference curvature restored with its actual transformed operator.
    # Here it is ONLY a declared local Taylor approximation in detector coords.
    L=state.film_z_mm-state.exit_z_mm
    a=state.active;cx=float(state.x[a].median());cy=float(state.y[a].median())
    vx=state.center_mm[0]-cx;vy=state.center_mm[1]-cy;r=math.sqrt(L*L+vx*vx+vy*vy)
    hxx=1/r-vx*vx/r**3;hyy=1/r-vy*vy/r**3;hxy=-vx*vy/r**3
    xy_t=torch.tensor(xy,device=field.device,dtype=torch.float64)
    quadratic=(hxx*xy_t[:,0]**2+2*hxy*xy_t[:,0]*xy_t[:,1]+hyy*xy_t[:,1]**2)/2
    approx*=torch.exp(2j*math.pi*quadratic/(state.wavelength_nm*1e-6))
    den=float(torch.linalg.vector_norm(exact));diff=float(torch.linalg.vector_norm(approx-exact))/max(den,1e-300)
    return {'relative_complex_field_l2':diff,'sample_count':len(ids),'pixel_yx':ids,
            'exact_real':exact.real.cpu().numpy(),'exact_imag':exact.imag.cpu().numpy(),
            'approx_real':approx.real.cpu().numpy(),'approx_imag':approx.imag.cpu().numpy(),
            'kind':'reference_discrepancy','global_bound':False,'operator':'exact-distance scalar RS1 on same ray bundle'}
