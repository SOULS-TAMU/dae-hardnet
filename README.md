# DAEHN

[![CI](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/ci.yml?query=branch%3Adevelopment)
[![Validation](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/validation.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/validation.yml?query=branch%3Adevelopment)
[![CodeQL](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/codeql.yml/badge.svg?branch=development)](https://github.com/SOULS-TAMU/dae-hardnet/actions/workflows/codeql.yml?query=branch%3Adevelopment)
[![Coverage](https://raw.githubusercontent.com/SOULS-TAMU/dae-hardnet/codex/coverage-development/coverage.svg)](https://github.com/SOULS-TAMU/dae-hardnet/tree/codex/coverage-development)

Hard-constrained learning for differential-algebraic systems in JAX.

DAEHN defines the differential problem and translates it into inputs for the
shared **OHAGI** learning engine and **PURIN** projection library. It contains no
separate backbone implementation, optimization solver, or numerical training
engine.

## Install

Install this development version with its tested shared dependencies:

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@9ab708b470c4d9359e57f7bd214b0b5515aa4115"
python -m pip install "ohagi @ git+https://github.com/SOULS-TAMU/Ohagi.git@e28f687e840d6840b259e687323029c3cc49bd59"
python -m pip install "git+https://github.com/SOULS-TAMU/dae-hardnet.git@development"
```

Python 3.10 or newer is required; dependency availability also depends on the
Python version and platform. Flax backbones and Orbax checkpoints are optional:

```shell
python -m pip install "ohagi[flax,orbax]"
```

## Workflow

1. Define coordinates, differential/algebraic states, parameters, equations,
   and initial or boundary conditions.
2. Choose Taylor neighborhoods, forward differences, or central differences.
3. Compile the differential terms into a PURIN projection problem.
4. Supply a JAX backbone and the problem-specific loss to OHAGI.
5. Train, predict, and reconstruct physical states and derivatives.

The public differential interface remains available through:

```python
from daehn.dae import (
    DAESystem, Coordinate, DifferentialState, AlgebraicState,
    DerivativeSpec, EquationBlock, TaylorNeighborhood,
    ForwardDifference, CentralDifference, compile_dae,
    DAETrainer, DAETrainingConfig,
)
```

`DAETrainer` supplies differential residuals, derivative supervision,
reconstruction, and phase decisions to OHAGI's shared steps and epoch engine.
The benchmark scripts retain their existing problem, training, and data APIs.

## Backbones and projected models

`daehn.MLPBackbone` uses OHAGI's functional JAX implementation with DAEHN's
existing `tanh` default. `daehn.FlaxMLPBackbone` is the optional Flax adapter.
A custom backbone may supply compatible `init` and `apply` functions.

`daehn.ProjectedNeuralModel` is a small frontend adapter to
`ohagi.training.projected.ProjectedTrainer`. Build, optimizer updates, batch
execution, prediction, and checkpoint transport belong to OHAGI.

For a built projected model, optional Orbax checkpointing retains the optimizer
state as well as model parameters:

```python
model.save_checkpoint("checkpoints/latest")
model.restore_checkpoint("checkpoints/latest")
```

## Export and repeated inference

Export a trained projected model once:

```python
model.export_inference("deployment/metadata.json")
```

The DAEHN facade supports loading with only the metadata path:

```python
from daehn import DAEHardNet

model = DAEHardNet().load("deployment/metadata.json")
prediction = model.predict(single_input)
# model(single_input) is equivalent; predict_jax keeps outputs on device.
```

The shared loader remains available directly:

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
prediction = predictor(single_input)
```

Keep the referenced `.jax_export` file alongside `metadata.json`. The loader
warms one JAX signature for repeated single-instance calls. Reuse the predictor;
loading and compilation are one-time costs. `predict_jax` returns a JAX array;
`predict` returns a synchronized NumPy array.

The only saved-model format is the shared `ohagi_model_v1` JAX bundle. Its
parameter archive uses `ohagi_pytree_npz_v1`; inference uses
`ohagi_jax_inference_v1` and `ohagi_jax_program_v1`. Deployment does not require
DAEHN, Flax, Orbax, or problem factories. Training checkpoints remain separate.
There are no legacy model readers or native export backends.

## Source organization

```text
src/daehn/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── root.py
│   ├── configuration.py
│   ├── model.py
│   ├── network.py
│   ├── problem.py
│   ├── projection.py
│   ├── data.py
│   ├── fit.py
│   ├── objectives.py
│   ├── logging.py
│   └── differential/
│       ├── __init__.py
│       ├── symbols.py
│       ├── system.py
│       ├── equations.py
│       ├── derivatives.py
│       ├── conditions.py
│       ├── discretization.py
│       ├── compiler.py
│       ├── scaling.py
│       └── reconstruction.py
└── serialization/
    ├── __init__.py
    └── artifacts.py
```

The flat API files contain DAEHN defaults, problem adapters, and differential
training policy. `differential/` contains the mathematical processing unique to
DAEHN. Serialization supplies frontend identity to OHAGI's common writer and
graph capture. Current benchmark paths such as `daehn.dae`, `daehn.opt`, and
`daehn.train.projected_neural` refer directly to these current implementations.

There is no `_aliases.py`, legacy import hook, compatibility-only implementation
tree, or duplicate `network/`, `problems/`, `projection/`, `training_compat/`,
`utils/`, or `experimental/` subtree in DAEHN.

## Benchmarks and development

The adjacent `daehn-benchmarks` directory contains the existing case studies.
See the [validation policy](.github/validation/README.md) for short-run comparisons and regression
checks. Source restructuring preserves benchmark scripts and their call APIs.

Packaging uses the `src` layout and PEP 517 configuration in `pyproject.toml`.
See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md),
[CITATION.cff](CITATION.cff), and [NOTICE](NOTICE).

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Original authorship and applicable upstream notices are retained in [NOTICE](NOTICE).

## Compatibility and checks

Development CI tests Python 3.10–3.14 on Linux, Windows and macOS, plus Linux
ARM64 and a pinned JAX baseline. Accelerator support follows JAX's available
wheels; these checks exercise CPU execution. Coverage combines statements and
branches across the whole frontend. Optional Flax and Orbax integrations are
tested separately. See the [validation policy](.github/validation/README.md)
for numerical tolerances, current quality baselines and performance limitations.
