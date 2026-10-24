"""Taylor and finite-difference processing of differential terms."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from hashlib import sha256
from math import prod
import json
import jax.numpy as jnp
import numpy as np
from daehn.api.differential.scaling import derivative_physical_scale


@dataclass(frozen=True)
class CompiledDiscretization:
    """Neighborhood offsets, derivative scales and compiled reconstruction callables."""

    name: str
    offsets: Any
    derivatives: tuple
    derivative_physical_scales: Any
    reconstruct_differential_states: Any
    estimate_derivatives: Any
    signature: str


class DiscretizationStrategy:
    """Interface for compiling a differential system's neighborhood discretization."""

    def compile(self, system, scaling):
        """Compile reconstruction and derivative estimators for a system and scaling."""
        raise NotImplementedError


@dataclass(frozen=True)
class TaylorNeighborhood(DiscretizationStrategy):
    """Taylor neighborhood definition with coordinate steps, radii and aggregation policy."""

    deltas: tuple[float, ...]
    order: int = 2
    radii: tuple[float, ...] = (1.0,)
    aggregation: str = "equal_average"

    def __post_init__(self):
        if self.order < 1:
            raise ValueError("Taylor order must be positive")
        if any((float(v) <= 0 for v in self.deltas)):
            raise ValueError("Taylor deltas must be positive")
        if not self.radii or any((float(v) <= 0 for v in self.radii)):
            raise ValueError("Taylor radii must be positive")
        if self.aggregation not in {"equal_average", "weighted_average"}:
            raise ValueError("unsupported Taylor aggregation")

    def compile(self, system, scaling):
        """Build Taylor reconstruction coefficients and neighborhood derivative estimators."""
        if len(self.deltas) != len(system.coordinates):
            raise ValueError("one Taylor delta is required per coordinate")
        derivatives = system.derivative_closure
        state_index = {
            state.name: i for i, state in enumerate(system.differential_states)
        }
        offsets, physical_scales, weights = _taylor_neighborhood_geometry(
            self, derivatives
        )
        coeff, derivative_state, ndiff_states = _taylor_reconstruction_coefficients(
            self, derivatives, offsets, physical_scales, state_index, system
        )

        def reconstruct(context_parts, normalized_derivatives):
            _coords, _params, _center, neighbors = context_parts
            recs = []
            for n in range(offsets.shape[0]):
                current = neighbors[..., n, :]
                correction = jnp.zeros(current.shape, dtype=current.dtype)
                for d in range(len(derivatives)):
                    correction = correction.at[..., derivative_state[d]].add(
                        coeff[n, d] * normalized_derivatives[..., d]
                    )
                recs.append(current - correction)
            return jnp.sum(
                jnp.stack(recs, axis=-2) * jnp.asarray(weights)[..., None], axis=-2
            )

        state_columns, state_pseudoinverses = _taylor_estimation_resources(
            coeff, derivative_state, ndiff_states, offsets
        )

        def estimate(center, neighbors):
            output = jnp.zeros(
                center.shape[:-1] + (len(derivatives),), dtype=center.dtype
            )
            for state, columns in enumerate(state_columns):
                if not len(columns):
                    continue
                rhs = neighbors[..., :, state] - center[..., state, None]
                pinv = jnp.asarray(state_pseudoinverses[state], dtype=center.dtype)
                values = jnp.einsum("dn,...n->...d", pinv, rhs)
                output = output.at[..., jnp.asarray(columns)].set(values)
            return output

        signature_data = {
            "name": "taylor_neighborhood",
            "deltas": self.deltas,
            "order": self.order,
            "radii": self.radii,
            "derivatives": [(d.state.name, d.multi_index) for d in derivatives],
        }
        signature = sha256(
            json.dumps(signature_data, sort_keys=True).encode()
        ).hexdigest()
        return CompiledDiscretization(
            name="taylor_neighborhood",
            offsets=jnp.asarray(offsets),
            derivatives=derivatives,
            derivative_physical_scales=jnp.asarray(physical_scales),
            reconstruct_differential_states=reconstruct,
            estimate_derivatives=estimate,
            signature=signature,
        )


@dataclass(frozen=True)
class ForwardDifference(DiscretizationStrategy):
    """Forward finite-difference neighborhood defined by coordinate step sizes."""

    deltas: tuple[float, ...]

    def compile(self, system, scaling):
        """Build forward-difference offsets and derivative estimation callables."""
        del scaling
        if len(self.deltas) != len(system.coordinates):
            raise ValueError("one forward delta is required per coordinate")
        derivatives = system.derivative_closure
        offsets = []
        offset_lookup = {}
        for coord, delta in enumerate(self.deltas):
            for multiple in (1, 2):
                value = np.zeros((len(self.deltas),), dtype=np.float64)
                value[coord] = multiple * float(delta)
                offset_lookup[coord, multiple] = len(offsets)
                offsets.append(value)
        offsets = np.asarray(offsets)
        scales = np.asarray(
            [
                derivative_physical_scale(d, self.deltas, d.state.scale)
                for d in derivatives
            ]
        )
        state_index = {s.name: i for i, s in enumerate(system.differential_states)}

        def reconstruct(parts, normalized_derivatives):
            del normalized_derivatives
            return parts[2]

        def estimate(center, neighbors):
            values = []
            for spec in derivatives:
                coord = spec.pure_coordinate
                state = state_index[spec.state.name]
                if coord is None:
                    values.append(jnp.asarray(0.0, center.dtype))
                    continue
                h = float(self.deltas[coord])
                y0 = center[..., state]
                y1 = neighbors[..., offset_lookup[coord, 1], state]
                if spec.order == 1:
                    physical = (y1 - y0) / h
                elif spec.order == 2:
                    y2 = neighbors[..., offset_lookup[coord, 2], state]
                    physical = (y2 - 2.0 * y1 + y0) / (h * h)
                else:
                    physical = jnp.asarray(0.0, center.dtype)
                values.append(physical / scales[len(values)])
            return jnp.stack(values, axis=-1)

        signature = sha256(
            json.dumps(
                {"name": "forward_difference", "deltas": self.deltas}, sort_keys=True
            ).encode()
        ).hexdigest()
        return CompiledDiscretization(
            "forward_difference",
            jnp.asarray(offsets),
            derivatives,
            jnp.asarray(scales),
            reconstruct,
            estimate,
            signature,
        )


@dataclass(frozen=True)
class CentralDifference(DiscretizationStrategy):
    """Symmetric finite-difference neighborhood defined by coordinate step sizes."""

    deltas: tuple[float, ...]

    def compile(self, system, scaling):
        """Build central-difference offsets and derivative estimation callables."""
        del scaling
        if len(self.deltas) != len(system.coordinates):
            raise ValueError("one central delta is required per coordinate")
        derivatives = system.derivative_closure
        offsets = []
        lookup = {}
        for coord, delta in enumerate(self.deltas):
            for sign in (-1, 1):
                value = np.zeros((len(self.deltas),), dtype=np.float64)
                value[coord] = sign * float(delta)
                lookup[coord, sign] = len(offsets)
                offsets.append(value)
        offsets = np.asarray(offsets)
        scales = np.asarray(
            [
                derivative_physical_scale(d, self.deltas, d.state.scale)
                for d in derivatives
            ]
        )
        state_index = {s.name: i for i, s in enumerate(system.differential_states)}

        def reconstruct(parts, normalized_derivatives):
            del normalized_derivatives
            return parts[2]

        def estimate(center, neighbors):
            vals = []
            for i, spec in enumerate(derivatives):
                coord = spec.pure_coordinate
                state = state_index[spec.state.name]
                if coord is None:
                    physical = jnp.asarray(0.0, center.dtype)
                else:
                    h = float(self.deltas[coord])
                    ym = neighbors[..., lookup[coord, -1], state]
                    yp = neighbors[..., lookup[coord, 1], state]
                    if spec.order == 1:
                        physical = (yp - ym) / (2 * h)
                    elif spec.order == 2:
                        physical = (yp - 2 * center[..., state] + ym) / (h * h)
                    else:
                        physical = jnp.asarray(0.0, center.dtype)
                vals.append(physical / scales[i])
            return jnp.stack(vals, axis=-1)

        signature = sha256(
            json.dumps(
                {"name": "central_difference", "deltas": self.deltas}, sort_keys=True
            ).encode()
        ).hexdigest()
        return CompiledDiscretization(
            "central_difference",
            jnp.asarray(offsets),
            derivatives,
            jnp.asarray(scales),
            reconstruct,
            estimate,
            signature,
        )


def _taylor_estimation_resources(coeff, derivative_state, ndiff_states, offsets):
    """Prepare state-specific derivative columns and their pseudoinverses."""
    state_columns = []
    state_pseudoinverses = []
    for state in range(ndiff_states):
        columns = np.flatnonzero(derivative_state == state).astype(np.int32)
        state_columns.append(columns)
        if columns.size:
            state_pseudoinverses.append(
                np.linalg.pinv(coeff[:, columns], rcond=1e-12).astype(np.float64)
            )
        else:
            state_pseudoinverses.append(np.empty((0, len(offsets)), dtype=np.float64))
    return state_columns, state_pseudoinverses


def _taylor_reconstruction_coefficients(
    self, derivatives, offsets, physical_scales, state_index, system
):
    """Construct Taylor coefficients in the declared derivative/state order."""
    coeff = np.zeros((len(offsets), len(derivatives)), dtype=np.float64)
    for n, offset in enumerate(offsets):
        for d, spec in enumerate(derivatives):
            if spec.order > self.order:
                continue
            raw = prod(
                (float(offset[i]) ** power for i, power in enumerate(spec.multi_index))
            )
            raw /= float(spec.multi_factorial)
            coeff[n, d] = raw * physical_scales[d]
    derivative_state = np.asarray(
        [state_index[s.state.name] for s in derivatives], dtype=np.int32
    )
    ndiff_states = len(system.differential_states)
    return coeff, derivative_state, ndiff_states


def _taylor_neighborhood_geometry(self, derivatives):
    """Construct neighborhood offsets, weights and derivative scales."""
    offsets = []
    neighborhood_coordinates = []
    for coord, delta in enumerate(self.deltas):
        for radius in self.radii:
            value = np.zeros((len(self.deltas),), dtype=np.float64)
            value[coord] = float(delta) * float(radius)
            offsets.append(value)
            neighborhood_coordinates.append((coord, float(radius)))
    offsets = np.asarray(offsets, dtype=np.float64)
    weights = np.full((len(offsets),), 1.0 / max(1, len(offsets)), dtype=np.float64)
    physical_scales = np.asarray(
        [
            derivative_physical_scale(spec, self.deltas, spec.state.scale)
            for spec in derivatives
        ],
        dtype=np.float64,
    )
    return offsets, physical_scales, weights
