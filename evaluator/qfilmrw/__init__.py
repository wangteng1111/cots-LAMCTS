"""QfilmRW: explicit ray/field state, local Fourier solver, independent evidence."""
from .contracts import VERSION,Task,Policy
from .rays import trace_lens,RWError,RayState
from .solver import solve_condition,assemble_score
from .api import evaluate
