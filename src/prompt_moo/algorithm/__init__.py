"""Algorithm implementations: OPRO, GPO, TextGrad.

Each module contains the algorithm-specific Optimizer, GradientComputer,
and LossComputer subclasses co-located with the algorithm class for cohesion.
"""

from .gpo import GPO
from .opro import OPRO
from .textgrad import TextGrad

__all__ = [
    "OPRO",
    "GPO",
    "TextGrad",
]
