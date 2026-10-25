# Differential problems and training

## Workflow

1. Define coordinates, differential/algebraic states and physical parameters.
2. Declare derivatives, equations and initial/boundary conditions.
3. Choose a discretization and call `compile_dae`.
4. Prepare the projection and supply a backbone to `DAETrainer`.
5. Fit observations and reconstruct physical states and derivatives.

DAEHN compiles differential terms into a PURIN problem. OHAGI executes the
shared backbone and training engine. `DAEHardNet` loads inference bundles;
use `DAETrainer` for differential problem training.

## Discretization

| Strategy (from `daehn.dae`) | Representation |
|---|---|
| `TaylorNeighborhood` | Local Taylor states and derivatives |
| `ForwardDifference` | One-sided finite differences |
| `CentralDifference` | Centered finite differences |

Select neighborhood sizes and coordinate scales for the physical problem.
Projection enforces the discretized constraints to solver tolerance; it does
not eliminate discretization error. Mark equation blocks affine only when they
are affine in projection variables. Nonlinear blocks require a compatible solver.

## Data and outputs

```python
result = trainer.fit(
    coordinates_train, parameters_train, states_train, derivatives_train,
    coordinates_validation, parameters_validation,
    states_validation, derivatives_validation,
)
```

Arrays have matching sample counts within each split. Columns follow system
coordinates/parameters and compiled state/derivative ordering. Use
`np.empty((n_samples, 0))` for absent parameters. Keep units and scaling consistent.
The README gives a complete small example. Model reconstruction returns groups
including `states` and `derivatives`.

## Backbones and checkpoints

`MLPBackbone(hidden_sizes=(32, 32))` uses functional JAX with the existing `tanh`
default. `FlaxMLPBackbone(hidden_sizes=(32, 32))` is the optional Flax adapter.
Custom backbones supply compatible `init` and `apply` functions.

For a built `ProjectedNeuralModel`, `model.save_checkpoint(path)` and
`model.restore_checkpoint(path)` retain parameters and optimizer state through
Orbax. Rebuild the same problem, backbone, optimizer and batching configuration
before restoration. Its `fit(x, y, epochs=remaining_epochs)` interface differs
from the differential trainer's eight-array signature above.

[Back to README](../README.md)
