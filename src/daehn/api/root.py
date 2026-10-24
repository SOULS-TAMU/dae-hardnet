"""Public DAEHN resources supplied to the common OHAGI engine."""

from daehn.api import (
    network as nn,
    projection as opt,
    problem as problems,
    differential as dae,
)
from daehn.api.problem import (
    Problem,
    SymbolicProblem,
    FunctionalProblem,
    MatrixProblem,
    ParametricLP,
    ParametricQP,
)
from daehn.api.configuration import (
    ProjectionConfig,
    LinearSolverConfig,
    ImplicitGradientConfig,
)
from daehn.api.model import DAEHardNet, ProjectedNeuralModel
from daehn.api.network import MLPBackbone, FlaxMLPBackbone
from ohagi.training.steps import Adam, apply_updates
from ohagi.training.state import (
    AdamState,
    BuildReport,
    CompiledProjectedGraphs,
    PredictionOutput,
    ProjectedMetrics,
)
from purin.solvers.implicit import (
    make_implicit_projection_layer,
    ImplicitProjectionLayer,
)
from purin.solvers.types import ProjectionLayerOutput
from purin.execution.runtime import configure_cache as configure_compilation_cache
from ohagi import training as train, losses
from ohagi.execution import runtime
from ohagi._utils import tree
from daehn.serialization import save_predictor, load_predictor

__version__ = "0.5.3"
__all__ = [
    "nn",
    "opt",
    "problems",
    "dae",
    "train",
    "losses",
    "runtime",
    "tree",
    "Problem",
    "SymbolicProblem",
    "FunctionalProblem",
    "MatrixProblem",
    "ParametricLP",
    "ParametricQP",
    "ProjectionConfig",
    "LinearSolverConfig",
    "ImplicitGradientConfig",
    "DAEHardNet",
    "ProjectedNeuralModel",
    "MLPBackbone",
    "FlaxMLPBackbone",
    "Adam",
    "AdamState",
    "BuildReport",
    "PredictionOutput",
    "ProjectedMetrics",
    "CompiledProjectedGraphs",
    "ProjectionLayerOutput",
    "ImplicitProjectionLayer",
    "make_implicit_projection_layer",
    "apply_updates",
    "configure_compilation_cache",
    "save_predictor",
    "load_predictor",
]
