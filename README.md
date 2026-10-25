# DAEHN

[![CI](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Validation](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/validation.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/validation.yml?query=branch%3Adevelopment)
[![CodeQL](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/codeql.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/codeql.yml?query=branch%3Adevelopment)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/dae-hardnet/codex/coverage-development/coverage.svg)](https://github.com/SOULS-TAMU/dae-hardnet/tree/codex/coverage-development)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/SOULS-TAMU/dae-hardnet/blob/development/LICENSE)
[![Tested Python: 3.10–3.14](https://img.shields.io/badge/Python-3.10%E2%80%933.14-blue?logo=python&logoColor=white)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![CPU platforms: Linux, Windows, macOS](https://img.shields.io/badge/CPU_platforms-Linux%20%7C%20Windows%20%7C%20macOS-blue)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Dependency review](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/dependency-review.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/dependency-review.yml?query=branch%3Adevelopment)

A JAX framework for hard-constrained learning of differential-algebraic
systems, including differential processing, discretization and reconstruction.

The frontend defines the problem and task policy.
[OHAGI](https://github.com/SOULS-TAMU/Ohagi) provides the shared learning and
inference engine; [PURIN](https://github.com/SOULS-TAMU/Purin) provides projection
algorithms and sparse kernels. Existing benchmark APIs remain supported.

## Install

Install the development version with its tested shared dependencies:

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@9ab708b470c4d9359e57f7bd214b0b5515aa4115"
python -m pip install "ohagi @ git+https://github.com/SOULS-TAMU/Ohagi.git@e28f687e840d6840b259e687323029c3cc49bd59"
python -m pip install "git+https://github.com/SOULS-TAMU/dae-hardnet.git@development"
```

See [installation options](docs/INSTALL.md) for editable installs and optional
Flax backbones/Orbax checkpoints.

## Python versions and platforms

Python **3.10–3.14** is tested on **Linux, Windows and macOS**, with additional
Linux ARM64 and pinned JAX baseline checks. Package installation and optional
Flax/Orbax integrations are tested across all three OSes. These are CPU checks;
accelerator support follows the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Basic API

### Differential problem

This small example learns the decay equation `dy/dt + y = 0` using state and
derivative observations. Other systems can include algebraic equations,
physical parameters and initial/boundary conditions.

```python
import jax.numpy as jnp
import numpy as np
from daehn import MLPBackbone
from daehn.dae import (
    Coordinate, DifferentialState, DerivativeSpec, EquationBlock,
    DAESystem, TaylorNeighborhood, compile_dae, DAETrainer, DAETrainingConfig,
)

strategy = TaylorNeighborhood(deltas=(0.01,), order=1)
state = DifferentialState("y")
system = DAESystem(
    coordinates=(Coordinate("t"),),
    differential_states=(state,),
    derivatives=(DerivativeSpec(state, (1,)),),
    differential_equations=EquationBlock(
        lambda coordinates, states, derivatives, parameters:
            jnp.array([derivatives[0] + states[0]]),
        size=1,
        affine_in_projection_variables=True,
    ),
    algebraic_equalities=EquationBlock(affine_in_projection_variables=True),
    algebraic_inequalities=EquationBlock(affine_in_projection_variables=True),
    name="decay",
)
compiled = compile_dae(system, discretization=strategy)
trainer = DAETrainer(
    compiled=compiled,
    prepared_projection=compiled.prepare_projection(
        batch_size=2, method="linear"
    ),
    backbone=MLPBackbone(hidden_sizes=(4,)),
    config=DAETrainingConfig(
        epochs=2,
        pretrain_epochs=1,
        batch_size=2,
        hidden_sizes=(4,),
        seed=7,
        print_every=0,
    ),
)
x = np.linspace(0, 1, 8)[:, None]
p = np.empty((8, 0))
y = np.exp(-x)
result = trainer.fit(x[:4], p[:4], y[:4], -y[:4], x[4:], p[4:], y[4:], -y[4:])
decoded, *_ = trainer.model.forward(
    result.parameters, jnp.asarray(x[:2]), jnp.asarray(p[:2])
)
prediction = decoded["states"]
```

See [problem definitions](docs/PROBLEM.md) for discretization and backbone options.

## Load and predict

The inference facade accepts a saved deployment bundle:

```python
from daehn import DAEHardNet

model = DAEHardNet().load("deployment/metadata.json")
prediction = model.predict([0.7])
```

`DAEHardNet` is the loading facade; differential training uses `DAETrainer`.
See [export and inference](docs/INFERENCE.md) to export the example above with
an explicit physical-state output graph.

The shared loader is available to both frameworks:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
prediction = predictor.predict([0.7])
```

Pass the metadata path and keep its referenced `.jax_export` file with the
bundle. Load once and reuse the predictor. `predict` returns synchronized NumPy
output; `predict_jax` returns a JAX array. See [inference](docs/INFERENCE.md).

## Data convention

Coordinates, physical parameters, states and derivatives are separate arrays
with samples on the first axis. Columns follow the declared/compiled ordering.
Use `(n_samples, 0)` for absent physical parameters.

## Examples and documentation

- [Install](docs/INSTALL.md): dependencies, optional integrations and packaging.
- [Problem definitions](docs/PROBLEM.md): modeling, backbones and training.
- [Inference](docs/INFERENCE.md): export, model bundles and repeated prediction.
- [Version notes](docs/VERSION.md) and [changelog](CHANGELOG.md): development changes.
- [Contributing](CONTRIBUTING.md) and [validation policy](.github/validation/README.md): development checks.

CI requires **90% combined statement and branch coverage**, cognitive complexity
at most **15** per function and the documented type-check policy. API snapshots,
numerical regression and exact checkpoint-resume checks protect compatibility.

## Citation and license

Use [CITATION.bib](CITATION.bib) for BibTeX or GitHub's **Cite this repository**
menu. Licensed under [Apache 2.0](LICENSE); attribution is retained in [NOTICE](NOTICE).
