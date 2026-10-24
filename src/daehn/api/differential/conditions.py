"""Initial and boundary conditions, sampling, losses, and hard transforms."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class Condition:
    """Target function prescribed on a fixed coordinate location."""

    coordinate_index: int
    location: float
    target: Callable
    kind: str = "dirichlet"


@dataclass(frozen=True)
class InitialCondition(Condition):
    """Condition prescribing state values at an initial coordinate location."""

    kind: str = "initial"


@dataclass(frozen=True)
class BoundaryCondition(Condition):
    """Condition prescribing state values on a coordinate boundary."""

    kind: str = "dirichlet"


def sample_condition_coordinates(condition, lower, upper, count: int, *, seed: int = 0):
    """Sample a box and pin one coordinate to the condition's location."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if count < 1:
        raise ValueError("count must be positive")
    if condition.coordinate_index < 0 or condition.coordinate_index >= lower.size:
        raise ValueError("condition coordinate index is out of range")
    rng = np.random.default_rng(seed)
    points = rng.uniform(lower, upper, size=(count, lower.size))
    points[:, condition.coordinate_index] = condition.location
    return points


def condition_mse(predictions, targets):
    """Compute mean squared condition error after checking that array shapes match."""
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("condition prediction and target shapes must match")
    return jnp.mean((predictions - targets) ** 2)


def hard_initial_transform(
    coordinates,
    raw_values,
    target_values,
    *,
    coordinate_index: int,
    location: float = 0.0,
):
    """Enforce an initial value by multiplying the raw correction by coordinate distance."""
    factor = coordinates[..., coordinate_index : coordinate_index + 1] - float(location)
    return target_values + factor * raw_values


def homogeneous_box_transform(
    coordinates, raw_values, *, spatial_indices, lower, upper
):
    """Enforce zero boundary values with products of distances to box faces."""
    factor = jnp.ones(coordinates.shape[:-1] + (1,), dtype=coordinates.dtype)
    for index, lo, hi in zip(spatial_indices, lower, upper, strict=True):
        x = coordinates[..., index : index + 1]
        factor = factor * (x - float(lo)) * (float(hi) - x)
    return factor * raw_values
