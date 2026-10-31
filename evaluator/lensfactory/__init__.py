"""Lens Factory-style single-candidate continuous optical solver."""
from .contracts import LensFactoryConfig
from .solver import VERSION, evaluate, inspect_prescription

__all__ = ["LensFactoryConfig", "VERSION", "evaluate", "inspect_prescription"]
