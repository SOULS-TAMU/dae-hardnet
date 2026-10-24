"""Differential hard-constrained learning through OHAGI and PURIN."""

import sys as _sys
from importlib import import_module as _module
from .api.root import *  # noqa: F403 - compatibility facade mirrors upstream exports
from .api.root import __all__ as __all__, __version__ as __version__

# Current public benchmark paths point directly to their implementations.
# No legacy implementation tree or import hook is installed.
for _name, _target in {
    "dae": "daehn.api.differential",
    "dae.model": "daehn.api.differential.reconstruction",
    "dae.conditions": "daehn.api.differential.conditions",
    "dae.discretization": "daehn.api.differential.discretization",
    "opt": "daehn.api.projection",
    "opt.layers": "daehn.api.projection",
    "opt.layers.projection": "daehn.api.projection",
    "train": "ohagi.training",
    "train.projected_neural": "daehn.api.root",
    "train.steps": "ohagi.training.steps",
    "nn": "daehn.api.network",
    "problems": "daehn.api.problem",
    "losses": "ohagi.losses",
    "runtime": "ohagi.execution.runtime",
    "tree": "ohagi._utils.tree",
}.items():
    _sys.modules[__name__ + "." + _name] = _module(_target)
del _name, _target
