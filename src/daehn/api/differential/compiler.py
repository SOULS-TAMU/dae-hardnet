"""Compile differential definitions into PURIN projection problems."""

from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from typing import Any
import jax
import numpy as np
from daehn.api.problem import FunctionalProblem
from daehn.api.projection import ProjectionConfig
from daehn.api.projection import ImplicitGradientConfig
from daehn.api.projection import LinearEqualityProjection
from daehn.api.projection import SparseProjection
from daehn.api.projection import FischerBurmeisterProjection
from hashlib import sha256
import json
import warnings
from daehn.api.differential.scaling import ScalingConfig
from daehn.api.differential.scaling import safe_row_scales


@dataclass(frozen=True)
class ProjectionLayout:
    """Slices identifying derivative and algebraic components of projection variables."""

    derivative_slice: slice
    algebraic_slice: slice
    size: int

    def unpack(self, values):
        """Split the final axis into derivative and algebraic projection values."""
        values = jnp.asarray(values)
        return (values[..., self.derivative_slice], values[..., self.algebraic_slice])


@dataclass(frozen=True)
class ContextLayout:
    """Slices and dimensions for coordinates, parameters and neighboring states."""

    coordinate_slice: slice
    parameter_slice: slice
    center_state_slice: slice
    neighbor_state_slice: slice
    coordinate_count: int
    parameter_count: int
    differential_state_count: int
    neighborhood_count: int
    size: int

    def pack(self, coordinates, parameters, center_states, neighbor_states):
        """Concatenate context components, flattening neighborhood and state axes."""
        coordinates = jnp.asarray(coordinates)
        center_states = jnp.asarray(center_states)
        parameters = jnp.asarray(parameters)
        neighbor_states = jnp.asarray(neighbor_states)
        batch_shape = coordinates.shape[:-1]
        flat_neighbors = neighbor_states.reshape(batch_shape + (-1,))
        return jnp.concatenate(
            (coordinates, parameters, center_states, flat_neighbors), axis=-1
        )

    def unpack(self, context):
        """Recover coordinates, parameters, center states and neighborhood states."""
        context = jnp.asarray(context)
        coords = context[..., self.coordinate_slice]
        parameters = context[..., self.parameter_slice]
        center = context[..., self.center_state_slice]
        neighbors = context[..., self.neighbor_state_slice].reshape(
            context.shape[:-1]
            + (self.neighborhood_count, self.differential_state_count)
        )
        return (coords, parameters, center, neighbors)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Selected projection method and compiled DAE dimension diagnostics."""

    selected_method: str
    condition_estimate: float
    equality_rows: int
    projection_variables: int


@dataclass(frozen=True)
class CompiledDAE:
    """Differential residuals, discretization, scaling and reconstruction resources."""

    system: Any
    discretization: Any
    projection_layout: Any
    context_layout: Any
    residual_scales: Any
    variable_physical_scales: Any
    raw_equalities: Any
    raw_inequalities: Any
    scaled_equalities: Any
    scaled_inequalities: Any
    reconstruct: Any
    build_target: Any
    decode: Any
    linear_matrix: Any | None
    signature: str
    diagnostics: ProjectionDiagnostics

    @property
    def parameter_size(self) -> int:
        """Return the packed projection-context width."""
        return int(self.context_layout.size)

    @property
    def variable_size(self) -> int:
        """Return the derivative-plus-algebraic projection width."""
        return int(self.projection_layout.size)

    @property
    def is_explicit_linear(self) -> bool:
        """Report whether the compiled equations support explicit linear projection."""
        return (
            self.linear_matrix is not None
            and self.system.algebraic_inequalities.size == 0
        )

    def pack_context(self, coordinates, parameters, center_states, neighbor_states):
        """Pack physical coordinates, parameters and state neighborhoods for projection."""
        return self.context_layout.pack(
            coordinates, parameters, center_states, neighbor_states
        )

    def physical_residuals(self, context, normalized_variables):
        """Evaluate unscaled physical equality and inequality residuals."""
        return (
            self.raw_equalities(context, normalized_variables),
            self.raw_inequalities(context, normalized_variables),
        )

    def prepare_projection(
        self,
        *,
        batch_size: int,
        method: str = "auto",
        config: ProjectionConfig | None = None,
        dtype=jnp.float64,
        backend: str = "cpu",
    ):
        """Select and prepare a PURIN solver for this compiled differential problem."""
        selected = str(method).lower()
        if selected == "auto":
            selected = "linear" if self.is_explicit_linear else "sparse"
        if config is None:
            config = ProjectionConfig(
                tolerance=1e-09,
                max_iterations=100,
                implicit_gradient=ImplicitGradientConfig(
                    differentiate_problem_parameters=True
                ),
            )
        if selected == "linear":
            if not self.is_explicit_linear:
                raise ValueError(
                    "explicit linear projection is not available for this compiled DAE"
                )
            rows = int(self.linear_matrix.shape[0])
            layer = LinearEqualityProjection(
                A=np.asarray(self.linear_matrix),
                b0=np.zeros((rows,), dtype=np.float64),
                b_parameter_map=np.eye(rows, dtype=np.float64),
            ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
            return PreparedDAEProjection(
                self, layer, "linear", config, dtype, backend, int(batch_size)
            )
        problem = FunctionalProblem(
            parameter_size=self.parameter_size,
            variable_size=self.variable_size,
            objective=lambda theta, y: jnp.asarray(0.0, dtype=y.dtype),
            equalities=self.scaled_equalities,
            inequalities=self.scaled_inequalities,
            equality_size=self.system.equality_size,
            inequality_size=self.system.algebraic_inequalities.size,
            name=f"{self.system.name}_compiled",
        )
        if selected == "sparse":
            layer = SparseProjection(problem, config=config).prepare(
                batch_size=batch_size, dtype=dtype, backend=backend
            )
        elif selected in {"fb", "fischer_burmeister"}:
            layer = FischerBurmeisterProjection(
                problem,
                tolerance=config.tolerance,
                max_iterations=config.max_iterations,
                line_search=config.backtracking,
                gradient_policy=config.implicit_gradient.policy,
                inexact_gradient_tolerance=config.implicit_gradient.inexact_tolerance,
                adjoint_gradient_tolerance=config.implicit_gradient.adjoint_tolerance,
                differentiate_problem_parameters=config.implicit_gradient.differentiate_problem_parameters,
            ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
        elif selected in {
            "fischer-burmeister-newton",
            "fischer-burmeister-reduced",
            "primal-dual-interior-point",
        }:
            from daehn.api.projection import prepare_projection
            from purin.solvers.config import FischerBurmeisterConfig
            from purin.solvers.config import ProjectionConfig as SharedConfig
            from purin.solvers.config import ImplicitGradientConfig as SharedImplicit

            implicit = SharedImplicit(**vars(config.implicit_gradient))
            config_type = (
                SharedConfig
                if selected == "primal-dual-interior-point"
                else FischerBurmeisterConfig
            )
            if type(config).__module__.startswith(("ohagi.solvers.", "purin.solvers.")):
                shared_config = config
            else:
                from purin.solvers.config import LinearSolverConfig as SharedLinear

                shared_linear = SharedLinear(**vars(config.linear_solver))
                options = dict(
                    tolerance=config.tolerance,
                    max_iterations=config.max_iterations,
                    implicit_gradient=implicit,
                    linear_solver=shared_linear,
                    transpose_mode=config.transpose_mode,
                )
                if selected == "primal-dual-interior-point":
                    options.update(
                        mu=config.mu,
                        backtracking=config.backtracking,
                        max_backtracking_steps=config.max_backtracking_steps,
                        backtracking_factor=config.backtracking_factor,
                        interior_margin=config.interior_margin,
                        store_residual_history=config.store_residual_history,
                    )
                shared_config = config_type(**options)
            layer = prepare_projection(
                problem,
                method=selected,
                config=shared_config,
                batch_size=batch_size,
                dtype=dtype,
                backend=backend,
            )
        else:
            raise ValueError("unknown projection method")
        return PreparedDAEProjection(
            self, layer, selected, config, dtype, backend, int(batch_size)
        )


@dataclass(frozen=True, eq=False)
class PreparedDAEProjection:
    """Prepared PURIN projection coupled to DAE scaling and context resources."""

    compiled: CompiledDAE
    layer: Any
    method: str
    config: ProjectionConfig
    dtype: Any
    backend: str
    batch_size: int

    def _parameters(self, context):
        if self.method == "linear":
            zeros = jnp.zeros(
                (context.shape[0], self.compiled.variable_size), dtype=context.dtype
            )
            return -jax.vmap(self.compiled.scaled_equalities)(context, zeros)
        return context

    def __call__(self, target, context):
        return self.layer(target, self._parameters(context))

    def with_info(self, target, context):
        """Project DAE variables and return the corresponding solver diagnostics."""
        return self.layer.with_info(target, self._parameters(context))


def compile_dae(
    system, *, discretization, scaling: ScalingConfig | None = None
) -> CompiledDAE:
    """Compile differential equations into reusable PURIN projection resources.

    Args:
        system: ``DAESystem`` defining coordinates, states, derivatives and
            differential/algebraic residual blocks.
        discretization: Neighborhood strategy, such as Taylor, forward
            difference or central difference, providing a ``compile`` method.
        scaling: Optional ``ScalingConfig``; defaults to the standard scaling.

    Returns:
        ``CompiledDAE`` containing packed context/variable layouts, residual
        evaluators, physical reconstruction and projection preparation resources.

    Notes:
        Differential processing belongs to this frontend. The compiled problem
        is supplied to PURIN; training remains in the shared OHAGI engine.
    """
    (
        algebraic_scales,
        context_layout,
        derivative_scales,
        disc,
        nalg,
        nderiv,
        ndiff,
        projection_layout,
        scaling,
        variable_scales,
    ) = _prepare_dae_layouts(discretization, scaling, system)

    def reconstruct(context, normalized_variables):
        parts = context_layout.unpack(context)
        dbar, abar = projection_layout.unpack(normalized_variables)
        physical_derivatives = dbar * derivative_scales
        differential_states = disc.reconstruct_differential_states(parts, dbar)
        algebraic_states = abar * algebraic_scales
        states = jnp.concatenate((differential_states, algebraic_states), axis=-1)
        return (states, physical_derivatives)

    def raw_equalities(context, normalized_variables):
        coords, params, _center, _neighbors = context_layout.unpack(context)
        states, derivatives = reconstruct(context, normalized_variables)
        differential = system.differential_equations.function(
            coords, states, derivatives, params
        )
        algebraic = system.algebraic_equalities.function(
            coords, states, derivatives, params
        )
        return jnp.concatenate(
            (jnp.atleast_1d(differential), jnp.atleast_1d(algebraic))
        )

    def raw_inequalities(context, normalized_variables):
        coords, params, _center, _neighbors = context_layout.unpack(context)
        states, derivatives = reconstruct(context, normalized_variables)
        return jnp.atleast_1d(
            system.algebraic_inequalities.function(coords, states, derivatives, params)
        )

    (
        A0,
        row_scales,
        row_scales_j,
        scaled_equalities,
        scaled_inequalities,
        zero_variables,
    ) = _prepare_dae_residual_scales(
        context_layout,
        projection_layout,
        raw_equalities,
        raw_inequalities,
        scaling,
        system,
    )

    cond, linear_matrix = _prepare_dae_linear_system(
        A0,
        context_layout,
        row_scales,
        scaled_equalities,
        scaling,
        system,
        zero_variables,
    )

    def build_target(center_all_states, normalized_derivative_anchor=None):
        center_all_states = jnp.asarray(center_all_states)
        if normalized_derivative_anchor is None:
            normalized_derivative_anchor = jnp.zeros(
                center_all_states.shape[:-1] + (nderiv,), dtype=center_all_states.dtype
            )
        alg = (
            center_all_states[..., ndiff : ndiff + nalg] / algebraic_scales
            if nalg
            else jnp.empty(center_all_states.shape[:-1] + (0,), center_all_states.dtype)
        )
        return jnp.concatenate((normalized_derivative_anchor, alg), axis=-1)

    def decode(context, normalized_variables):
        states, derivatives = reconstruct(context, normalized_variables)
        return {
            "states": states,
            "derivatives": derivatives,
            "normalized_variables": normalized_variables,
        }

    signature_payload = {
        "system": system.name,
        "states": system.state_names,
        "derivatives": [(d.state.name, d.multi_index) for d in disc.derivatives],
        "discretization": disc.signature,
        "scaling": scaling.__dict__,
    }
    signature = sha256(
        json.dumps(signature_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    diagnostics = ProjectionDiagnostics(
        "linear" if linear_matrix is not None else "sparse",
        cond,
        system.equality_size,
        projection_layout.size,
    )
    return CompiledDAE(
        system,
        disc,
        projection_layout,
        context_layout,
        row_scales_j,
        variable_scales,
        raw_equalities,
        raw_inequalities,
        scaled_equalities,
        scaled_inequalities,
        reconstruct,
        build_target,
        decode,
        linear_matrix,
        signature,
        diagnostics,
    )


def _prepare_dae_linear_system(
    A0, context_layout, row_scales, scaled_equalities, scaling, system, zero_variables
):
    """Detect and prepare the existing explicit affine projection system."""
    all_affine = (
        system.differential_equations.affine_in_projection_variables
        and system.algebraic_equalities.affine_in_projection_variables
        and system.algebraic_inequalities.affine_in_projection_variables
    )
    linear_matrix = None
    cond = float("nan")
    if all_affine and system.algebraic_inequalities.size == 0:
        linear_matrix = np.asarray(A0 / row_scales[:, None], dtype=np.float64)
        probe_context = jnp.linspace(
            0.1, 0.9, max(1, context_layout.size), dtype=jnp.float64
        )[: context_layout.size]
        A1 = np.asarray(
            jax.jacfwd(scaled_equalities, argnums=1)(probe_context, zero_variables)
        )
        if not np.allclose(linear_matrix, A1, rtol=1e-09, atol=1e-10):
            linear_matrix = None
        elif (
            linear_matrix.shape[0]
            and np.linalg.matrix_rank(linear_matrix) < linear_matrix.shape[0]
        ):
            warnings.warn(
                "compiled affine equality matrix is row-rank deficient; using nonlinear sparse projection",
                RuntimeWarning,
            )
            linear_matrix = None
        else:
            cond = (
                float(np.linalg.cond(linear_matrix @ linear_matrix.T))
                if linear_matrix.shape[0]
                else 1.0
            )
            if cond > scaling.condition_warning:
                warnings.warn(
                    f"scaled projection matrix condition estimate is {cond:.3e}",
                    RuntimeWarning,
                )
    return cond, linear_matrix


def _prepare_dae_residual_scales(
    context_layout, projection_layout, raw_equalities, raw_inequalities, scaling, system
):
    """Prepare physical residual row scales and normalized residual callables."""
    zero_context = jnp.zeros((context_layout.size,), dtype=jnp.float64)
    zero_variables = jnp.zeros((projection_layout.size,), dtype=jnp.float64)
    if system.equality_size:
        A0 = np.asarray(
            jax.jacfwd(raw_equalities, argnums=1)(zero_context, zero_variables),
            dtype=np.float64,
        )
        row_scales = safe_row_scales(A0, scaling)
    else:
        A0 = np.zeros((0, projection_layout.size), dtype=np.float64)
        row_scales = np.ones((0,), dtype=np.float64)
    ineq_scales = np.ones((system.algebraic_inequalities.size,), dtype=np.float64)
    row_scales_j = jnp.asarray(row_scales)
    ineq_scales_j = jnp.asarray(ineq_scales)

    def scaled_equalities(context, normalized_variables):
        return raw_equalities(context, normalized_variables) / row_scales_j

    def scaled_inequalities(context, normalized_variables):
        return raw_inequalities(context, normalized_variables) / ineq_scales_j

    return (
        A0,
        row_scales,
        row_scales_j,
        scaled_equalities,
        scaled_inequalities,
        zero_variables,
    )


def _prepare_dae_layouts(discretization, scaling, system):
    """Prepare discretization and packed context/variable layouts."""
    scaling = ScalingConfig() if scaling is None else scaling
    disc = discretization.compile(system, scaling)
    nderiv = len(disc.derivatives)
    nalg = len(system.algebraic_states)
    projection_layout = ProjectionLayout(
        slice(0, nderiv), slice(nderiv, nderiv + nalg), nderiv + nalg
    )
    ncoords = len(system.coordinates)
    nparams = len(system.parameters)
    ndiff = len(system.differential_states)
    nneigh = int(disc.offsets.shape[0])
    start = 0
    coordinate_slice = slice(start, start + ncoords)
    start += ncoords
    parameter_slice = slice(start, start + nparams)
    start += nparams
    center_slice = slice(start, start + ndiff)
    start += ndiff
    neighbor_slice = slice(start, start + nneigh * ndiff)
    start += nneigh * ndiff
    context_layout = ContextLayout(
        coordinate_slice,
        parameter_slice,
        center_slice,
        neighbor_slice,
        ncoords,
        nparams,
        ndiff,
        nneigh,
        start,
    )
    derivative_scales = jnp.asarray(disc.derivative_physical_scales)
    algebraic_scales = jnp.asarray(
        [s.scale for s in system.algebraic_states], dtype=jnp.float64
    )
    variable_scales = jnp.concatenate((derivative_scales, algebraic_scales))
    return (
        algebraic_scales,
        context_layout,
        derivative_scales,
        disc,
        nalg,
        nderiv,
        ndiff,
        projection_layout,
        scaling,
        variable_scales,
    )
