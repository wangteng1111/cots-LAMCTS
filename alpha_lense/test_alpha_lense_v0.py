import torch
from alpha_lense.alpha_lense_v0 import DesignSpec, EvalResult, ConstraintEvaluator, AlphaLenseMCTS

def test_feasible_t3_80():
    spec=DesignSpec(efl_target_mm=80,efl_tol_mm=2,max_f_number=3.0,min_entrance_pupil_mm=26,
                    min_image_circle_mm=43,min_bfd_mm=10,max_total_length_mm=100,max_distortion_pct=3)
    raw=EvalResult(1.2,{"efl_mm":80.5,"f_number":2.9,"entrance_pupil_mm":27.5,"image_circle_mm":44,
                        "bfd_mm":12,"total_length_mm":90,"distortion_pct":1.5})
    r=ConstraintEvaluator()(raw,spec)
    assert r.feasible and not r.violations

def test_great_mtf_does_not_beat_wrong_focal_length():
    spec=DesignSpec(80,2,max_f_number=3)
    good=ConstraintEvaluator()(EvalResult(5.0,{"efl_mm":80,"f_number":3}),spec)
    wrong=ConstraintEvaluator()(EvalResult(0.01,{"efl_mm":50,"f_number":1.4}),spec)
    assert good.feasible and not wrong.feasible
    assert AlphaLenseMCTS.better(good,wrong)

def test_spec_vector_has_presence_bits():
    v=DesignSpec(80,2,max_f_number=None,max_t_stop=3).vector()
    assert v.numel()==24
    assert v[5].item()==0.0  # max_f_number absent
    assert v[7].item()==1.0  # max_t_stop present
