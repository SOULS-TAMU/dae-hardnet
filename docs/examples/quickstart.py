import jax
import numpy as np
from daehn import MLPBackbone
from daehn.dae import (
    Coordinate,
    DifferentialState,
    DerivativeSpec,
    EquationBlock,
    DAESystem,
    TaylorNeighborhood,
    compile_dae,
    DAETrainer,
    DAETrainingConfig,
)

jax.config.update("jax_enable_x64", True)

# 1. Data: time, observed state, observed derivative, physical parameters.
t = np.linspace(0, 1, 8)[:, None]
y = np.exp(-t)
dy = -y
p = np.empty((len(t), 0))  # This equation has no physical parameters.

# 2. Problem: declare y(t), dy/dt, and the residual dy/dt + y.
state = DifferentialState("y")
system = DAESystem(
    coordinates=(Coordinate("t"),),
    differential_states=(state,),
    derivatives=(DerivativeSpec(state, (1,)),),
    differential_equations=EquationBlock(lambda t, y, dy, p: dy + y, size=1),
)

# 3. Projection: process the differential equation using a Taylor neighborhood.
compiled = compile_dae(system, discretization=TaylorNeighborhood(deltas=(0.01,)))
projection = compiled.prepare_projection(batch_size=2)

# 4. Network and training: use the first four samples for training.
trainer = DAETrainer(
    compiled=compiled,
    prepared_projection=projection,
    backbone=MLPBackbone(hidden_sizes=(8,)),
    config=DAETrainingConfig(
        epochs=2,
        pretrain_epochs=1,
        batch_size=2,
        hidden_sizes=(8,),
        print_every=0,
    ),
)
result = trainer.fit(
    train_coordinates=t[:4],
    train_parameters=p[:4],
    train_targets=y[:4],
    train_derivative_targets=dy[:4],
    validation_coordinates=t[4:],
    validation_parameters=p[4:],
    validation_targets=y[4:],
    validation_derivative_targets=dy[4:],
)

# 5. Predict the physical state y at a new time.
_, prediction, *_ = trainer.predict(result.parameters, [[0.7]], np.empty((1, 0)))
print(prediction)

# Check the declared physical/output contract and model transport.

assert prediction.shape == (1, 1) and np.isfinite(prediction).all()
import jax.numpy as jnp

decoded, *_ = trainer.model.forward(
    result.parameters, jnp.asarray(t[:2]), jnp.asarray(p[:2])
)
np.testing.assert_allclose(decoded["states"] + decoded["derivatives"], 0.0, atol=1e-6)
from ohagi.inference import export, load


def predict_states(inputs):
    decoded, *_ = trainer.model.forward(
        result.parameters, jnp.repeat(inputs, 2, axis=0), jnp.empty((2, 0))
    )
    return decoded["states"][:1]


export(
    "deployment/metadata.json",
    predict_states,
    input_size=1,
    output_size=1,
    framework="daehn",
)
np.testing.assert_allclose(
    load("deployment/metadata.json").predict([0.7]), prediction[0], atol=1e-7
)
