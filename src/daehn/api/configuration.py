"""DAEHN defaults for the common PURIN solver configuration."""

from dataclasses import dataclass, field
from purin.solvers.config import ImplicitGradientConfig as ImplicitGradientConfig
from purin.solvers.config import LinearSolverConfig as _Linear
from purin.solvers.config import ProjectionConfig as _Projection


@dataclass(frozen=True)
class LinearSolverConfig(_Linear):
    """DAE frontend defaults for PURIN's linear-system solver configuration."""

    max_iterations: int = 50


@dataclass(frozen=True)
class ProjectionConfig(_Projection):
    """DAE frontend defaults for projection tolerance, iteration and gradient policy."""

    linear_solver: _Linear = field(default_factory=LinearSolverConfig)
