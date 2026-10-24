"""DAEHN identity for the shared OHAGI inference graph capture."""

from ohagi.serialization.predictor import model_graph


def daehn_graph(model):
    """Capture the trained DAE predictor and physical reconstruction as a JAX graph."""
    return model_graph(model, framework="daehn")
