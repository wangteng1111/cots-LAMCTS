"""Versioned task and numerical contracts for the QfilmRW ray-wave solver."""
from dataclasses import dataclass, asdict
import hashlib,json,math

VERSION='QfilmRW-0.1.0'
FREQS=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
WAVES=(430.,470.,510.,550.,590.,630.,670.)
WEIGHTS=(.075,.125,.16,.28,.16,.125,.075)

@dataclass(frozen=True)
class Task:
    pitch_um:float=.25
    support_mm:float=.64
    frequencies:tuple=FREQS
    otf_domain:str='windowed'
    frame_mm:tuple=(36.,24.)
    kernel:str='rs1'
    def validate(self):
        if self.otf_domain not in ('windowed','full_plane'):raise ValueError('explicit OTF domain required')
        if self.kernel!='rs1':raise ValueError('only declared scalar RS1 boundary convention supported')
        if not all(math.isfinite(v) and v>0 for v in (self.pitch_um,self.support_mm,*self.frame_mm)):raise ValueError('invalid task geometry')
        if self.pixel_n<16 or abs(self.pixel_n*self.pitch_um*.001-self.support_mm)>1e-10:raise ValueError('nonintegral PSF window')
        if not self.frequencies or any(not math.isfinite(f) or f<0 or f>=500/self.pitch_um for f in self.frequencies):raise ValueError('invalid frequencies')
    @property
    def pixel_n(self):return int(round(self.support_mm/(self.pitch_um*.001)))
    @property
    def signature(self):return digest(asdict(self))

@dataclass(frozen=True)
class Policy:
    pupils:tuple=(128,256,512,1024)
    angular_grid_n:int=1024
    pupil_otf_tolerance:float=.002
    integral_relative_tolerance:float=.005
    grid_otf_tolerance:float=.002
    support_otf_tolerance:float=.002
    pitch_otf_tolerance:float=.001
    joint_otf_tolerance:float=.005
    residual_phase_target_cycles:float=.25
    reference_relative_l2_tolerance:float=.005
    max_seconds:float=7200.
    max_live_elements:int=300_000_000
    def validate(self):
        if len(self.pupils)<2 or tuple(sorted(set(self.pupils)))!=self.pupils or min(self.pupils)<16 or any(n%2 for n in self.pupils):raise ValueError('invalid pupil schedule')
        if self.angular_grid_n<32 or self.max_seconds<=0 or self.max_live_elements<1024:raise ValueError('invalid resources')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
