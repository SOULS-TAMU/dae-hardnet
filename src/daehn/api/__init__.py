"""Problem-specific DAEHN adapters to OHAGI and PURIN."""

from importlib import import_module

__all__ = [
    "nn",
    "opt",
    "problems",
    "dae",
    "train",
    "losses",
    "runtime",
    "tree",
    "Problem",
    "SymbolicProblem",
    "FunctionalProblem",
    "MatrixProblem",
    "ParametricLP",
    "ParametricQP",
    "ProjectionConfig",
    "LinearSolverConfig",
    "ImplicitGradientConfig",
    "DAEHardNet",
    "ProjectedNeuralModel",
    "MLPBackbone",
    "FlaxMLPBackbone",
    "Adam",
    "AdamState",
    "BuildReport",
    "PredictionOutput",
    "ProjectedMetrics",
    "CompiledProjectedGraphs",
    "ProjectionLayerOutput",
    "ImplicitProjectionLayer",
    "make_implicit_projection_layer",
    "apply_updates",
    "configure_compilation_cache",
    "save_predictor",
    "load_predictor",
]


def __getattr__(name):
    if name not in __all__ and name != "__version__":
        raise AttributeError(name)
    root = import_module("daehn.api.root")
    if name in root.__all__ or name == "__version__":
        return getattr(root, name)
    raise AttributeError(name)
