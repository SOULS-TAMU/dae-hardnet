"""Nonlinear solver selection, physical residuals and complete-model derivatives."""

from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from daehn.dae import (
    Coordinate,
    DifferentialState,
    DerivativeSpec,
    EquationBlock,
    DAESystem,
    TaylorNeighborhood,
    ForwardDifference,
    CentralDifference,
    compile_dae,
)
from daehn.api.differential.reconstruction import (
    FullModelDerivativePlan,
    extract_network_derivatives,
    _central_weights,
)
from daehn.api.objectives import physical_violation_metrics, dae_metrics
from daehn.api.fit import DAETrainingConfig, _aggregate, _fixed_batches


def system(nonlinear=False):
    state = DifferentialState("y")
    return DAESystem(
        coordinates=(Coordinate("t"),),
        differential_states=(state,),
        derivatives=(DerivativeSpec(state, (1,)),),
        differential_equations=EquationBlock(
            lambda c, s, d, p: jnp.array([d[0] + s[0] ** (2 if nonlinear else 1)]),
            size=1,
            affine_in_projection_variables=not nonlinear,
        ),
        algebraic_equalities=EquationBlock(affine_in_projection_variables=True),
        algebraic_inequalities=EquationBlock(affine_in_projection_variables=True),
        name="coverage",
    )


def test_complete_derivative_polynomial_and_autodiff():
    state = DifferentialState("y")
    specs = (DerivativeSpec(state, (1,)), DerivativeSpec(state, (2,)))
    x = jnp.array([[0.2], [0.6]])

    def apply(p, x):
        return jnp.array([x[0] ** 3])

    actual = extract_network_derivatives(apply, None, x, specs, 1)
    np.testing.assert_allclose(
        actual,
        np.column_stack((3 * np.asarray(x[:, 0]) ** 2, 6 * np.asarray(x[:, 0]))),
        atol=1e-12,
    )
    assert extract_network_derivatives(apply, None, x, [], [state]).shape == (2, 0)
    plan = FullModelDerivativePlan.build(specs, [state], (0.01,))
    model = SimpleNamespace(
        forward=lambda p, c, t: ({"states": c**3}, None, None, None)
    )
    actual = plan.evaluate(model, None, x, jnp.empty((2, 0)))
    np.testing.assert_allclose(
        actual[:, 0], 3 * np.asarray(x[:, 0]) ** 2 + 0.0001, atol=1e-10
    )
    np.testing.assert_allclose(actual[:, 1], 6 * np.asarray(x[:, 0]), atol=1e-10)
    with pytest.raises(ValueError):
        FullModelDerivativePlan.build(specs, [state], (0,))
    with pytest.raises(ValueError):
        FullModelDerivativePlan.build(specs, [state], (0.1, 0.2))
    with pytest.raises(ValueError):
        _central_weights(-1)
    np.testing.assert_array_equal(_central_weights(0)[1], [1])


@pytest.mark.coverage
@pytest.mark.parametrize(
    "method",
    [
        "auto",
        "sparse",
        "fb",
        "fischer-burmeister-newton",
        "fischer-burmeister-reduced",
        "primal-dual-interior-point",
    ],
)
def test_nonlinear_projection_paths(method):
    compiled = compile_dae(
        system(True), discretization=TaylorNeighborhood((0.01,), order=1)
    )
    projection = compiled.prepare_projection(batch_size=1, method=method)
    context = compiled.pack_context(
        jnp.array([[0.2]]), jnp.empty((1, 0)), jnp.array([[0.8]]), jnp.array([[[0.79]]])
    )
    target = compiled.build_target(jnp.array([[0.8]]), jnp.array([[-0.01]]))
    value = projection(target, context)
    assert np.isfinite(value).all()
    residuals = jax.vmap(compiled.physical_residuals)(context, value)
    assert np.max(np.abs(residuals[0])) < 1e-4
    with pytest.raises(ValueError):
        compiled.prepare_projection(batch_size=1, method="linear")
    with pytest.raises(ValueError):
        compiled.prepare_projection(batch_size=1, method="bad")


def test_physical_metrics_and_targets():
    compiled = compile_dae(
        system(), discretization=TaylorNeighborhood((0.01,), order=1)
    )
    c = jnp.array([[0.0], [1.0]])
    states = jnp.array([[1.0], [2.0]])
    derivatives = -states
    params = jnp.empty((2, 0))
    metrics = physical_violation_metrics(
        compiled.system, c, states, derivatives, params
    )
    assert metrics["violation"] == 0
    assert (
        physical_violation_metrics(
            compiled.system, c, states, derivatives, params, mask=jnp.array([1, 0])
        )["violation"]
        == 0
    )
    neighbors = states[:, None, :] - 0.01 * states[:, None, :]
    context = compiled.pack_context(c, params, states, neighbors)
    target = compiled.build_target(states, -0.01 * states)
    value = compiled.prepare_projection(batch_size=2, method="auto")(target, context)
    assert dae_metrics(compiled, context, value, targets=states)["state_mse"] >= 0
    assert (
        compiled.parameter_size == context.shape[1]
        and compiled.variable_size == value.shape[1]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"pretrain_epochs": -1},
        {"batch_size": 0},
        {"eta": -1},
        {"derivative_weight": -1},
        {"full_model_derivative_steps": ()},
        {"full_model_derivative_steps": (0,)},
    ],
)
def test_training_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        DAETrainingConfig(**kwargs)


def test_metric_aggregation_and_empty_batches():
    with pytest.raises(RuntimeError):
        _aggregate([])
    assert _aggregate([({"loss": 2, "max_x": 2}, 1), ({"loss": 4, "max_x": 4}, 3)]) == {
        "loss": 3.5,
        "max_x": 4,
    }
    assert (
        list(
            _fixed_batches(
                np.empty((0, 1)),
                np.empty((0, 0)),
                np.empty((0, 1)),
                np.empty((0, 1)),
                batch_size=2,
            )
        )
        == []
    )


@pytest.mark.parametrize(
    "strategy",
    [
        TaylorNeighborhood((0.1, 0.2)),
        ForwardDifference((0.1, 0.2)),
        CentralDifference((0.1, 0.2)),
    ],
)
def test_invalid_discretization_dimensions(strategy):
    with pytest.raises(ValueError):
        compile_dae(system(), discretization=strategy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"function": lambda c, s, d, p: jnp.ones(1), "size": -1},
        {"function": 3},
        {"size": 1, "dependencies": ()},
    ],
)
def test_equation_block_validation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        EquationBlock(**kwargs)
