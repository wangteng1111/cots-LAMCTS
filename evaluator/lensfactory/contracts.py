"""Frozen configuration contract for the Lens Factory single-candidate solver."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LensFactoryConfig:
    fields: tuple = (0.0, 0.5, 1.0)
    wavelengths_nm: tuple = (470.0, 550.0, 630.0)
    spectral_weights: tuple = (1.0, 1.0, 1.0)
    frame_mm: tuple = (36.0, 24.0)

    stage1_metric: str = "spot"
    stage1_pupil_n: int = 48
    stage1_maxiter: int = 12
    uniform_gap_seeds_mm: tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    local_grid_steps_mm: tuple = (0.5, 0.2, 0.1)
    gap_bounds_mm: tuple = (0.0, 30.0)
    bfl_search_half_width_mm: float = 20.0
    efl_tolerance_fraction: float = 0.03
    efl_tolerance_floor_mm: float = 0.05

    stage2_pupil_n: int = 24
    stage2_detector_n: int = 64
    stage2_pitch_um: float = 1.0
    stage2_pixel_block: int = 16
    stage2_maxiter: int = 3
    stage2_maxfun: int = 40
    mtf_max_frequency_lpmm: float = 100.0
    coarse_mtf_points: int = 11
    final_mtf_points: int = 21

    final_pupils: tuple = (64, 128, 256, 512, 1024)
    final_supports_mm: tuple = (0.256, 0.384, 0.512, 0.768)
    final_pitch_um: float = 0.5
    final_pixel_block: int = 64
    final_max_pair_evaluations: int = 150_000_000_000_000
    max_bad_jacobian_fraction: float = 0.02

    def validate(self):
        if self.stage1_metric not in ("spot", "opd"):
            raise ValueError("stage1_metric must be 'spot' or 'opd'")
        if len(self.fields) < 1 or any(not math.isfinite(float(x)) or not 0 <= x <= 1 for x in self.fields):
            raise ValueError("fields must be finite normalized image heights in [0,1]")
        if tuple(sorted(set(self.wavelengths_nm))) != tuple(self.wavelengths_nm):
            raise ValueError("wavelengths_nm must be distinct and ascending")
        if not self.wavelengths_nm or any(not math.isfinite(float(x)) or x <= 0 for x in self.wavelengths_nm):
            raise ValueError("invalid wavelengths")
        if len(self.spectral_weights) != len(self.wavelengths_nm) or any(
            not math.isfinite(float(x)) or x < 0 for x in self.spectral_weights
        ) or sum(self.spectral_weights) <= 0:
            raise ValueError("invalid spectral weights")
        if len(self.frame_mm) != 2 or any(not math.isfinite(float(x)) or x <= 0 for x in self.frame_mm):
            raise ValueError("frame_mm must contain positive width and height")
        if self.stage1_pupil_n < 16 or self.stage2_pupil_n < 16 or self.stage2_detector_n < 16:
            raise ValueError("pupil and detector grids must be at least 16")
        if self.stage1_maxiter < 0 or self.stage2_maxiter < 0 or self.stage2_maxfun < 1:
            raise ValueError("invalid optimizer budget")
        if self.stage2_pixel_block < 1 or self.final_pixel_block < 1:
            raise ValueError("pixel blocks must be positive")
        if self.stage2_pitch_um <= 0 or self.final_pitch_um <= 0:
            raise ValueError("detector pitch must be positive")
        if self.gap_bounds_mm[0] < 0 or self.gap_bounds_mm[1] <= self.gap_bounds_mm[0]:
            raise ValueError("invalid nonnegative air-gap bounds")
        if self.bfl_search_half_width_mm <= 0:
            raise ValueError("BFL search half-width must be positive")
        if self.mtf_max_frequency_lpmm <= 0 or self.coarse_mtf_points < 3 or self.final_mtf_points < 3:
            raise ValueError("invalid MTF frequency grid")
        if self.mtf_max_frequency_lpmm >= 500.0 / self.stage2_pitch_um:
            raise ValueError("coarse MTF range exceeds detector Nyquist")
        if self.mtf_max_frequency_lpmm >= 500.0 / self.final_pitch_um:
            raise ValueError("final MTF range exceeds detector Nyquist")
        if len(self.final_pupils) < 2 or any(b != 2*a for a, b in zip(self.final_pupils, self.final_pupils[1:])):
            raise ValueError("final_pupils must be a doubling sequence")
        if tuple(sorted(set(self.final_supports_mm))) != tuple(self.final_supports_mm) or len(self.final_supports_mm) < 2:
            raise ValueError("final_supports_mm must be strictly increasing")
        for support in self.final_supports_mm:
            pixels = support / (self.final_pitch_um * 0.001)
            if abs(pixels - round(pixels)) > 1e-6 or pixels < 16:
                raise ValueError("each final support must contain an integer number of pixels")
        if self.final_max_pair_evaluations < 1:
            raise ValueError("final_max_pair_evaluations must be positive")
        if not 0 <= self.max_bad_jacobian_fraction <= 0.1:
            raise ValueError("invalid Jacobian threshold")

    @property
    def coarse_frequencies(self):
        n = self.coarse_mtf_points
        return tuple(self.mtf_max_frequency_lpmm * i / (n - 1) for i in range(n))

    @property
    def final_frequencies(self):
        n = self.final_mtf_points
        return tuple(self.mtf_max_frequency_lpmm * i / (n - 1) for i in range(n))

    def as_dict(self):
        return asdict(self)
