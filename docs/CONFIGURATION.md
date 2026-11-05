# Configuration and loss

## Training

`DAETrainingConfig` holds batching, optimizer, loss and phase settings. Its API
page lists actual defaults.

| Setting | Role |
|---|---|
| `epochs`, `pretrain_epochs`, `eta` | Horizon and raw-to-projected phase policy |
| `batch_size`, `learning_rate`, `seed` | Batching, optimizer and initialization |
| `derivative_weight` | Observed derivative error |
| `derivative_consistency_weight` | Network/reconstructed derivative consistency |
| `violation_weight`, `consistency_weight` | Physical residual and consistency penalties |
| `initial_weight`, `boundary_weight` | Observation-based condition penalties |
| `full_model_derivative_steps` | Steps for physical-model derivative evaluation |

Loss penalties and hard projection have different roles. Do not describe a
penalized condition as a hard architectural condition. Hard-transform utilities
are separate and require a compatible backbone/problem.

## Differential representation

`DerivativeSpec(state, (1,))` declares the first derivative for one coordinate;
multiple coordinates use a derivative multi-index. `TaylorNeighborhood`,
`ForwardDifference` and `CentralDifference` process terms before `compile_dae`.

`EquationBlock` returns a residual vector in declared order. Mark it affine only
when it is affine in the projection variables. The linear shortcut requires all
relevant blocks to satisfy its conditions; otherwise use an automatic/general
solver. Tight projection convergence does not remove discretization error.

Use `MLPBackbone` or optional `FlaxMLPBackbone`. Prepared projections have fixed
batch shape. `trainer.predict` handles prediction batching; direct
`trainer.model.forward` calls must obey the prepared shape.
