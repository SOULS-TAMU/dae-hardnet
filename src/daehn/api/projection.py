"""Projection definitions supplied by PURIN, with DAEHN default options."""

import jax.numpy as jnp
from daehn.api.configuration import (
    ProjectionConfig,
    LinearSolverConfig as LinearSolverConfig,
    ImplicitGradientConfig as ImplicitGradientConfig,
)
from purin.solvers.linear_equality.solver import (
    LinearEqualityProjection as LinearEqualityProjection,
    PreparedLinearEqualityProjection as PreparedLinearEqualityProjection,
)
from purin.solvers.fischer_burmeister.dense_semismooth import (
    FischerBurmeisterProjection,
)
from purin.projections.implicit_distance import (
    SparseProjection,
    PreparedProjection as PreparedProjection,
)
from purin.solvers.api import prepare_projection as _prepare


def prepare_projection(
    problem, *, method="fb", config=None, batch_size, dtype=jnp.float64, backend="cpu"
):
    """Prepare the selected PURIN projection using DAE frontend configuration."""
    if method == "linear":
        return problem.prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    if method == "sparse":
        return SparseProjection(problem, config or ProjectionConfig()).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
    if method == "fb":
        if config is not None:
            raise TypeError(
                "Configure FischerBurmeisterProjection directly, or select a named shared backend"
            )
        return FischerBurmeisterProjection(problem).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
    return _prepare(
        problem,
        method=method,
        config=config,
        batch_size=batch_size,
        dtype=dtype,
        backend=backend,
    )


def registered_methods():
    """Return the projection methods available through the DAE frontend."""
    return (
        "linear",
        "sparse",
        "fb",
        "fischer-burmeister-newton",
        "fischer-burmeister-reduced",
        "primal-dual-interior-point",
    )
