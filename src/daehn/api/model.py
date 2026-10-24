"""DAEHN model adapter to the shared OHAGI projected trainer."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ohagi.training.projected import ProjectedTrainer

if TYPE_CHECKING:
    from ohagi.inference import Predictor


class ProjectedNeuralModel(ProjectedTrainer):
    """DAE frontend combining differential resources with OHAGI's projected trainer."""

    framework = "daehn"

    def _inference_spec(self):
        from daehn.serialization.artifacts import daehn_graph

        return daehn_graph(self)

    def export_inference(self, metadata_path=None):
        """Export the DAE predictor for metadata-only single-instance JAX inference."""
        from ohagi.inference import export

        path = metadata_path or getattr(self, "_inference_metadata_path", None)
        if path is None:
            raise ValueError("Provide a destination metadata.json path.")
        result = export(path, **self._inference_spec())
        self._inference_metadata_path = result["metadata_path"]
        return result

    @staticmethod
    def load_inference(metadata_path, *, device="cpu"):
        """Prepare a shared JAX predictor from a metadata.json path."""
        from ohagi.inference import load

        return load(metadata_path, device=device)

    def save_predictor(
        self,
        directory,
        *,
        network_spec: dict[str, Any] | None = None,
        projection_spec: dict[str, Any] | None = None,
        problem=None,
        input_transform: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        predictor_factory: str | None = None,
        factory_config: dict[str, Any] | None = None,
    ):
        """Write this model's prepared predictor in the shared JAX format."""
        self._require_built()
        assert self.prepared_projection is not None
        from daehn.serialization import save_predictor
        from pathlib import Path

        self._inference_metadata_path = str(
            Path(directory).expanduser().resolve() / "metadata.json"
        )

        serialized_network_spec = dict(network_spec or {"type": "jax_graph"})
        if isinstance(self.parameters, dict) and "backbone" in self.parameters:
            serialized_network_spec.setdefault("parameter_path", ["backbone"])

        return save_predictor(
            directory,
            network_spec=serialized_network_spec,
            params=self.parameters,
            input_size=self.input_dim,
            output_size=len(self.prepared_projection.compiled.variable_names),
            batch_size=self.batch_size,
            projection_spec=projection_spec,
            problem=problem,
            input_transform=input_transform,
            dtype="float64",
            backend=self.prepared_projection.backend,
            metadata=metadata,
            predictor_factory=predictor_factory,
            factory_config=factory_config,
            _inference_spec=self._inference_spec(),
        )


class DAEHardNet:
    """Load a shared OHAGI JAX bundle for repeated DAE model inference.

    Use ``DAEHardNet().load(metadata_path)`` to prepare the predictor once.
    Training continues to use the existing ``DAETrainer`` and
    ``ProjectedNeuralModel`` interfaces.
    """

    def __init__(self):
        """Create an unloaded inference model."""
        self._predictor: Predictor | None = None

    def load(self, metadata_path, *, device="cpu"):
        """Load and warm the shared predictor, returning this model.

        Keep the portable program referenced by ``metadata.json`` alongside
        it. A failed load leaves any previously loaded predictor intact.
        """
        self._predictor = ProjectedNeuralModel.load_inference(
            metadata_path, device=device
        )
        return self

    def _require_predictor(self):
        if self._predictor is None:
            raise RuntimeError(
                "Load a model with load(metadata_path) before prediction."
            )
        return self._predictor

    def predict(self, values):
        """Predict one instance and return synchronized NumPy values."""
        return self._require_predictor().predict(values)

    def predict_jax(self, values):
        """Predict one instance and keep the output on its JAX device."""
        return self._require_predictor().predict_jax(values)

    __call__ = predict
