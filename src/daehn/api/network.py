"""DAEHN backbone defaults over the shared functional and Flax backbones."""

from dataclasses import dataclass
import jax
from jax.nn import *  # noqa: F403 - compatibility facade mirrors upstream exports
from ohagi.nn.mlp import MLPBackbone as _MLP
from ohagi.nn.adapters.flax_functional import FlaxMLPBackbone as FlaxMLPBackbone
from ohagi.training.state import (
    BuildReport as BuildReport,
    CompiledProjectedGraphs as CompiledProjectedGraphs,
    PredictionOutput as PredictionOutput,
    ProjectedMetrics as ProjectedMetrics,
)
from purin.solvers.types import ProjectionLayerOutput as ProjectionLayerOutput
from purin.solvers.implicit import ImplicitProjectionLayer as ImplicitProjectionLayer


@dataclass(frozen=True)
class MLPBackbone(_MLP):
    """DAE backbone dimensions and normalization over the shared MLP implementation."""

    activation: object = jax.nn.tanh
