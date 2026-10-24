"""Differential definitions, physical scaling and boundary transforms."""

import numpy as np
import pytest
import jax.numpy as jnp
from daehn.dae import (
    Coordinate,
    DifferentialState,
    AlgebraicState,
    Parameter,
    DerivativeSpec,
    TaylorNeighborhood,
)
from daehn.api.differential.derivatives import derivative_closure
from daehn.api.differential.scaling import (
    ScalingConfig,
    derivative_physical_scale,
    safe_row_scales,
)
from daehn.api.differential.conditions import (
    InitialCondition,
    sample_condition_coordinates,
    condition_mse,
    hard_initial_transform,
    homogeneous_box_transform,
)


@pytest.mark.parametrize(
    "cls", [Coordinate, DifferentialState, AlgebraicState, Parameter]
)
@pytest.mark.parametrize("name", ["", "a b", "4x"])
def test_invalid_symbols(cls, name):
    with pytest.raises(ValueError):
        cls(name)


@pytest.mark.parametrize(
    "cls", [Coordinate, DifferentialState, AlgebraicState, Parameter]
)
def test_symbol_scale_and_name(cls):
    assert cls(" y ").name == "y"
    with pytest.raises(ValueError):
        cls("y", scale=0)


def test_coordinate_bounds():
    with pytest.raises(ValueError):
        Coordinate("t", lower=1, upper=0)


@pytest.mark.parametrize("multi", [(), (0,), (-1,), (0, 0)])
def test_invalid_derivative_orders(multi):
    with pytest.raises(ValueError):
        DerivativeSpec(DifferentialState("y"), multi)


def test_derivative_closure_and_scales():
    y = DifferentialState("y", scale=2)
    d = DerivativeSpec(y, (3, 0))
    mixed = DerivativeSpec(y, (1, 1))
    assert d.order == 3 and d.multi_factorial == 6 and d.pure_coordinate == 0
    assert mixed.pure_coordinate is None
    closure = derivative_closure([d, mixed], coordinate_count=2)
    assert {x.multi_index for x in closure} == {(1, 0), (2, 0), (3, 0), (1, 1)}
    np.testing.assert_allclose(derivative_physical_scale(d, (0.5, 0.2), 2), 96)
    with pytest.raises(ValueError):
        derivative_closure([d], coordinate_count=1)
    with pytest.raises(ValueError):
        derivative_physical_scale(d, (0, 0.2), 2)


@pytest.mark.parametrize(
    "config",
    [
        {"residual_mode": "bad"},
        {"metric": "bad"},
        {"minimum_scale": 0},
        {"maximum_scale": 0},
    ],
)
def test_invalid_scaling(config):
    with pytest.raises(ValueError):
        ScalingConfig(**config)


def test_row_scaling():
    np.testing.assert_array_equal(
        safe_row_scales([[2, -3], [0, 0]], ScalingConfig()), [3, 1e-12]
    )
    np.testing.assert_array_equal(
        safe_row_scales([[2, -3]], ScalingConfig(residual_mode="none")), [1]
    )
    assert safe_row_scales(np.empty((0, 2)), ScalingConfig()).shape == (0,)


@pytest.mark.parametrize(
    "kwargs", [{"order": 0}, {"deltas": (-1,)}, {"radii": ()}, {"aggregation": "bad"}]
)
def test_invalid_taylor_options(kwargs):
    with pytest.raises(ValueError):
        TaylorNeighborhood(**({"deltas": (0.1,)} | kwargs))


def test_conditions_and_hard_transforms():
    condition = InitialCondition(0, 0, lambda x: x)
    a = sample_condition_coordinates(condition, [0, 0], [1, 1], 5, seed=3)
    np.testing.assert_array_equal(a[:, 0], 0)
    np.testing.assert_array_equal(
        a, sample_condition_coordinates(condition, [0, 0], [1, 1], 5, seed=3)
    )
    with pytest.raises(ValueError):
        sample_condition_coordinates(condition, [0], [1], 0)
    with pytest.raises(ValueError):
        sample_condition_coordinates(InitialCondition(2, 0, lambda x: x), [0], [1], 2)
    assert float(condition_mse(jnp.ones((2, 1)), jnp.zeros((2, 1)))) == 1
    with pytest.raises(ValueError):
        condition_mse(jnp.zeros(2), jnp.zeros(3))
    x = jnp.array([[0.0], [0.5], [1.0]])
    np.testing.assert_array_equal(
        hard_initial_transform(
            x, jnp.ones_like(x), jnp.ones_like(x) * 2, coordinate_index=0
        ),
        [[2], [2.5], [3]],
    )
    np.testing.assert_array_equal(
        homogeneous_box_transform(
            x, jnp.ones_like(x), spatial_indices=[0], lower=[0], upper=[1]
        ),
        [[0], [0.25], [0]],
    )
