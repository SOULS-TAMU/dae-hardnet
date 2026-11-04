# DAE-HardNet

**Learn differential-algebraic systems with differentiable hard constraint projections.**

[![CI](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml?query=branch%3Amain)
[![Validation](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/validation.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/validation.yml?query=branch%3Amain)
[![CodeQL](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/codeql.yml?query=branch%3Amain)
[![Dependency review](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/dependency-review.yml/badge.svg?branch=main)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/dependency-review.yml?query=branch%3Amain)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/dae-hardnet/codex/coverage-main/coverage.svg)](https://github.com/SOULS-TAMU/dae-hardnet/tree/codex/coverage-main)

[![Python 3.10–3.14](https://img.shields.io/badge/Python-3.10%E2%80%933.14-3776AB?logo=python&logoColor=white)](.github/workflows/ci.yml)
[![CPU platforms](https://img.shields.io/badge/CPU-Linux%20%7C%20Windows%20%7C%20macOS-555555)](.github/workflows/ci.yml)
[![JAX](https://img.shields.io/badge/Powered_by-JAX-8A2BE2)](https://docs.jax.dev/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Citation: BibTeX](https://img.shields.io/badge/Cite-BibTeX-6F4E37)](CITATION.bib)

Define coordinates, states, derivatives and algebraic constraints, then pass the
processed problem to OHAGI's shared learning engine and PURIN's projection solvers.

- **Differential processing:** declared derivative specifications, discretization
  strategies and reconstruction of physical states and derivatives.
- **Hard algebraic constraints:** projected predictions under the selected solver's
  numerical tolerances.
- **Flexible backbones:** functional JAX networks and optional Flax adapters,
  with optional Orbax training checkpoints through OHAGI.
- **Reusable deployment:** a shared JAX bundle and the `DAEHardNet().load(...)` facade.

[Installation](#installation) · [Basic API](#basic-api) · [Load and predict](#load-and-predict) · [Documentation](#documentation) · [Contributing](#contributing) · [Citation](#citation)

## Installation

This README documents the **main branch (version 2.0.0)**. Install its source packages
in dependency order with Python 3.10 or newer:

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@development"
python -m pip install "ohagi @ git+https://github.com/SOULS-TAMU/Ohagi.git@development"
python -m pip install "daehn @ git+https://github.com/SOULS-TAMU/dae-hardnet.git@main"
```

For editable installs and optional Flax/Orbax integrations, see
[installation options](docs/INSTALL.md).

## Basic API

### Differential problem

This example learns `dy/dt + y = 0` from state and derivative observations.
It defines the system, prepares the differential projection, trains, and
reconstructs the predicted state:

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

The example uses a first-order Taylor neighborhood and an affine projection.
See [problem definitions](docs/PROBLEM.md) for differential specifications,
algebraic equations, discretization and backbone choices. `DAETrainer` handles
training; `DAEHardNet` is the deployment loading facade.

## Load and predict

Load a previously exported physical-state deployment bundle:

```python
from daehn import DAEHardNet

model = DAEHardNet().load("deployment/metadata.json")
prediction = model.predict([0.7])
```

For differential models, export the desired physical-state output graph explicitly;
[the inference guide](docs/INFERENCE.md) shows how to export the example above.

The same deployment bundle also works through OHAGI:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
for value in [0.4, 0.7, 1.0]:
    prediction = predictor.predict([value])
```

Keep the referenced `.jax_export` file with `metadata.json`. Load once and reuse
the predictor. `predict` returns synchronized NumPy values; `predict_jax` keeps
outputs on the JAX device. Deployment uses the shared JAX pathway.

## Data convention

Coordinates, physical parameters, states and derivatives are separate arrays,
with samples on the first axis and columns in declared/compiled order. Use
`(n_samples, 0)` when no physical parameters are present.

## The HardNet stack

- **[KKTHN](https://github.com/SOULS-TAMU/kkt-hardnet/tree/main),
  [DAEHN](https://github.com/SOULS-TAMU/dae-hardnet/tree/main) and
  [NLPOPTNet](https://github.com/SOULS-TAMU/nlpoptnet)** supply problem formulations
  and framework-specific processing.
- **[OHAGI](https://github.com/SOULS-TAMU/Ohagi/tree/development)** composes models,
  runs the shared learning engine, and handles checkpoints and JAX deployment.
- **[PURIN](https://github.com/SOULS-TAMU/Purin/tree/development)** owns projection
  problems, numerical solvers, sparse kernels and projection differentiation.

## Compatibility and validation

The CPU CI matrix covers **Python 3.10–3.14** on Linux, Windows and macOS,
plus Linux ARM64 on Python 3.13. It exercises current compatible JAX releases
and JAX 0.6.2 on Python 3.12. See the [workflow](.github/workflows/ci.yml)
for exact environments; accelerator installation follows the
[JAX guide](https://docs.jax.dev/en/latest/installation.html).

- **CI:** tests, distribution builds and installation checks.
- **Validation:** public APIs, numerical references, documentation and code quality.
- **Coverage:** at least **90% combined statement and branch coverage**; the badge
  links to the measured report for this branch.
- **Security:** CodeQL analysis and dependency review.

The frontend validation policy also checks API compatibility, checkpoint resume,
types, dependency boundaries and cognitive complexity. See
[validation details](.github/validation/README.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/INSTALL.md) | Source installs, optional integrations and packaging |
| [Problem definitions](docs/PROBLEM.md) | Modeling, backbones and training |
| [Inference](docs/INFERENCE.md) | Export, loading and repeated prediction |
| [Version notes](docs/VERSION.md) | Current development behavior |
| [Changelog](CHANGELOG.md) | Recorded changes |

## Contributing

Bug reports, examples and focused pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, numerical checks and compatibility
requirements. Keep generated results, environments and checkpoints out of Git.

## Citation

Use [CITATION.bib](CITATION.bib) for a copyable BibTeX entry or GitHub's
**Cite this repository** menu for [citation metadata](CITATION.cff).

## License

[Apache License 2.0](LICENSE). Authorship and upstream attribution are recorded
in [AUTHORS](AUTHORS) and [NOTICE](NOTICE).

[Changelog](CHANGELOG.md) · [Report an issue](https://github.com/SOULS-TAMU/dae-hardnet/issues)
