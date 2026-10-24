"""Shared OHAGI JAX model serialization with DAEHN identity."""

from ohagi.inference import Predictor
from ohagi.serialization.predictor import (
    save_predictor as _save,
    load_predictor,
    make_network_apply,
)
from ohagi.serialization.problem import save_problem, load_problem
from ohagi.export.weights import save_pytree, load_pytree


def save_predictor(
    directory,
    *,
    network_spec,
    params,
    input_size,
    output_size,
    batch_size,
    projection_spec=None,
    problem=None,
    input_transform=None,
    dtype="float64",
    backend="cpu",
    metadata=None,
    predictor_factory=None,
    factory_config=None,
    _inference_spec=None,
):
    """Save a prepared DAEHN predictor in the shared OHAGI JAX format."""
    options = dict(locals())
    # Normalize public DAEHN definition names to PURIN algorithm definitions.
    spec = dict(projection_spec or {"type": "none"})
    if spec.get("type") == "sparse":
        from daehn.api.configuration import ProjectionConfig

        spec["type"] = "implicit-distance"
        config = dict(spec.get("config", {}))
        linear = dict(config.get("linear_solver", {}))
        linear.setdefault(
            "max_iterations", ProjectionConfig().linear_solver.max_iterations
        )
        config["linear_solver"] = linear
        spec["config"] = config
    elif spec.get("type") == "fischer_burmeister":
        spec["type"] = "dense-semismooth"
    options["projection_spec"] = spec
    return _save(framework="daehn", **options)


__all__ = [
    "Predictor",
    "save_predictor",
    "load_predictor",
    "save_problem",
    "load_problem",
    "save_pytree",
    "load_pytree",
    "make_network_apply",
]
