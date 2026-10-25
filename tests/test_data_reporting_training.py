"""Dataset persistence, console contracts and final projected-state reconstruction."""

from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from daehn.api.data import DAEData
from daehn.api.logging import DAETrainingLogger
from daehn.dae import DAETrainer, DAETrainingConfig, TaylorNeighborhood, compile_dae
from daehn import MLPBackbone
from test_extended_differential import system


def test_dataset_roundtrip_with_optional_fields(tmp_path):
    x = np.ones((2, 1))
    p = np.empty((2, 0))
    for derivative in [None, x]:
        original = DAEData(
            x, p, x, derivatives=derivative, initial_coordinates=x, initial_targets=x
        )
        path = tmp_path / "data.npz"
        original.save(path)
        loaded = DAEData.load(path)
        np.testing.assert_array_equal(loaded.states, x)
        assert (loaded.derivatives is None) == (derivative is None)
        assert loaded.boundary_targets is None


def test_console_report_contract(capsys):
    logger = DAETrainingLogger(epochs=5, print_every=3)
    logger.start(
        system="decay",
        discretization="Taylor",
        projection="linear",
        samples=4,
        validation_samples=2,
        batch_size=2,
        network="small",
        pretrain_epochs=1,
        eta=None,
    )
    metrics = {
        key: 0
        for key in [
            "data_loss",
            "derivative_loss",
            "differential_violation",
            "algebraic_violation",
        ]
    }
    for epoch in range(1, 6):
        logger.epoch(epoch, metrics, metrics)
    logger.phase_switch(2, "pretrain_epochs=1")
    logger.finish(0.1)
    text = capsys.readouterr().out
    assert "001/005" in text and "003/005" in text and "005/005" in text
    assert "002/005" not in text and "004/005" not in text
    assert "Switching" in text and "eta: n/a" in text
    import daehn.api as api

    assert api.__getattr__("DAEHardNet")
    assert api.__getattr__("__version__")
    with pytest.raises(AttributeError):
        api.__getattr__("unknown")


@pytest.mark.coverage
def test_training_conditions_prediction_and_evaluation():
    compiled = compile_dae(
        system(), discretization=TaylorNeighborhood((0.01,), order=1)
    )
    config = DAETrainingConfig(
        epochs=1,
        pretrain_epochs=0,
        eta=None,
        batch_size=2,
        hidden_sizes=(3,),
        print_every=1,
        shuffle=False,
        full_model_derivative_steps=(0.01,),
        derivative_consistency_weight=0.1,
    )
    backbone = MLPBackbone((3,))
    trainer = DAETrainer(
        compiled=compiled,
        prepared_projection=compiled.prepare_projection(batch_size=2),
        backbone=backbone,
        config=config,
        initial_coordinates=np.array([[0.0]]),
        initial_targets=np.array([[1.0]]),
        boundary_coordinates=np.array([[1.0]]),
        boundary_targets=np.array([[np.exp(-1)]]),
    )
    x = np.linspace(0, 1, 6)[:, None]
    p = np.empty((6, 0))
    y = np.exp(-x)
    initial = backbone.init(jax.random.PRNGKey(3), 1, 1, jnp.float64)
    result = trainer.fit(
        x[:4],
        p[:4],
        y[:4],
        -y[:4],
        x[4:],
        p[4:],
        y[4:],
        -y[4:],
        initial_parameters=initial,
    )
    raw, projected, derivatives, independent, info = trainer.predict(
        result.parameters, x, p
    )
    assert (
        raw.shape == projected.shape == derivatives.shape == independent.shape == (6, 1)
    )
    assert len(info["converged"]) == 6
    np.testing.assert_allclose(projected + independent, 0, atol=1e-7)
    assert np.isfinite(trainer.evaluate(result.parameters, x, p, y, -y)["loss"])
    for which in ["training", "validation"]:
        args = [x[:4], p[:4], y[:4], -y[:4], x[4:], p[4:], y[4:], -y[4:]]
        args[3 if which == "training" else 7] = np.ones((1, 2))
        with pytest.raises(ValueError, match="derivative targets"):
            trainer.fit(*args)
    with pytest.raises(ValueError):
        DAETrainer(
            compiled=compiled,
            prepared_projection=compiled.prepare_projection(batch_size=2),
            backbone=backbone,
            config=replace(config, full_model_derivative_steps=(0.1, 0.2)),
        )
