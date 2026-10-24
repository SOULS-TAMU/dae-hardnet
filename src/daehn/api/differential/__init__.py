"""Continuous DAE formulation and discretization frontend for DAEHN."""

from daehn.api.differential.symbols import Coordinate
from daehn.api.differential.symbols import DifferentialState
from daehn.api.differential.symbols import AlgebraicState
from daehn.api.differential.symbols import Parameter
from daehn.api.differential.derivatives import DerivativeSpec
from daehn.api.differential.derivatives import derivative_closure
from daehn.api.differential.equations import EquationBlock
from daehn.api.differential.system import DAESystem
from daehn.api.differential.scaling import ScalingConfig
from daehn.api.differential.compiler import compile_dae
from daehn.api.differential.compiler import CompiledDAE
from daehn.api.differential.compiler import PreparedDAEProjection
from daehn.api.differential.reconstruction import FullModelDerivativePlan
from daehn.api.differential.reconstruction import ProjectedDAEModel
from daehn.api.differential.reconstruction import extract_network_derivatives
from daehn.api.data import DAEData
from daehn.api.fit import DAETrainer
from daehn.api.fit import DAETrainingConfig
from daehn.api.fit import DAETrainingResult
from daehn.api.logging import DAETrainingLogger
from daehn.api.objectives import physical_residual_blocks
from daehn.api.objectives import physical_violation_metrics
from daehn.api.objectives import dae_metrics
from daehn.api.differential.discretization import TaylorNeighborhood
from daehn.api.differential.discretization import ForwardDifference
from daehn.api.differential.discretization import CentralDifference
from daehn.api.differential.conditions import Condition
from daehn.api.differential.conditions import InitialCondition
from daehn.api.differential.conditions import BoundaryCondition
from daehn.api.differential.conditions import sample_condition_coordinates
from daehn.api.differential.conditions import condition_mse
from daehn.api.differential.conditions import hard_initial_transform
from daehn.api.differential.conditions import homogeneous_box_transform

__all__ = [
    "Coordinate",
    "DifferentialState",
    "AlgebraicState",
    "Parameter",
    "DerivativeSpec",
    "derivative_closure",
    "EquationBlock",
    "DAESystem",
    "ScalingConfig",
    "compile_dae",
    "CompiledDAE",
    "PreparedDAEProjection",
    "FullModelDerivativePlan",
    "ProjectedDAEModel",
    "extract_network_derivatives",
    "DAEData",
    "DAETrainer",
    "DAETrainingConfig",
    "DAETrainingResult",
    "DAETrainingLogger",
    "physical_residual_blocks",
    "physical_violation_metrics",
    "dae_metrics",
    "TaylorNeighborhood",
    "ForwardDifference",
    "CentralDifference",
    "Condition",
    "InitialCondition",
    "BoundaryCondition",
    "sample_condition_coordinates",
    "condition_mse",
    "hard_initial_transform",
    "homogeneous_box_transform",
]
