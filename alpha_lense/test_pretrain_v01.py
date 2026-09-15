import torch
from alpha_lense.core import DesignSpec, EvalResult
from alpha_lense.pretrain_v01 import NormalizedConstraintEvaluator, soft_improvement_policy, ordered_score

def test_feasible_dominates_better_image_quality_but_wrong_spec():
    spec=DesignSpec(80,2,max_f_number=3)
    ce=NormalizedConstraintEvaluator()
    good=ce(EvalResult(5.,{"efl":80.,"fno":3.}),spec).result
    wrong=ce(EvalResult(.01,{"efl":50.,"fno":1.4}),spec).result
    assert good.feasible and not wrong.feasible
    assert ordered_score(good)>ordered_score(wrong)

def test_missing_required_metric_is_invalid():
    spec=DesignSpec(80,2,max_f_number=3,max_distortion_pct=3)
    r=NormalizedConstraintEvaluator()(EvalResult(1.,{"efl":80.,"fno":2.8}),spec).result
    assert not r.feasible
    assert "distortion" in r.violations

def test_reverse_policy_uses_physics_improvements_only():
    p=EvalResult(10.,{},True,{})
    better=EvalResult(5.,{},True,{})
    worse=EvalResult(20.,{},True,{})
    infeasible=EvalResult(.001,{},False,{"efl":-2.})
    pi,good=soft_improvement_policy(p,[better,worse,infeasible],temperature=.2)
    assert good==[0]
    assert torch.allclose(pi,torch.tensor([1.,0.,0.]))

def test_multiple_improvements_form_soft_policy():
    p=EvalResult(10.,{},True,{})
    a=EvalResult(8.,{},True,{})
    b=EvalResult(5.,{},True,{})
    pi,good=soft_improvement_policy(p,[a,b],temperature=.5)
    assert good==[0,1]
    assert abs(float(pi.sum())-1.)<1e-6
    assert pi[1]>pi[0]>0
