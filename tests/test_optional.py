"""Orbax resume must reproduce uninterrupted training for both backbone adapters."""

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.optional


def same(a, b):
    aa, sa = jax.tree_util.tree_flatten(a)
    bb, sb = jax.tree_util.tree_flatten(b)
    assert sa == sb
    for one, two in zip(aa, bb):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(two))


@pytest.mark.parametrize("backend", ["jax", "flax"])
def test_exact_checkpoint_resume(tmp_path, monkeypatch, backend):
    monkeypatch.chdir(tmp_path)
    x = np.linspace(0.3, 1.3, 8)[:, None]
    y = np.column_stack((0.6 * x[:, 0], 0.4 * x[:, 0]))
    from daehn import (
        ProjectedNeuralModel,
        MLPBackbone,
        FlaxMLPBackbone,
        SymbolicProblem,
    )
    from purin.solvers.api import prepare_projection
    from purin.solvers.config import ProjectionConfig

    def make():
        backbone = (
            MLPBackbone(hidden_sizes=(4,))
            if backend == "jax"
            else FlaxMLPBackbone(hidden_sizes=(4,))
        )
        p = SymbolicProblem()
        t = p.parameters(["p"])
        v = p.variables(["u", "v"])
        p.subject_to(v.u + v.v == t.p, v.u >= 0, v.v >= 0)
        projection = prepare_projection(
            p.as_projection_problem(),
            method="primal-dual-interior-point",
            config=ProjectionConfig(max_iterations=12),
            batch_size=4,
        )
        m = ProjectedNeuralModel(
            projection=projection._prepared_solver,
            backbone_init=backbone.init,
            backbone_apply=backbone.apply,
            input_dim=1,
        )
        m.build(batch_size=4, seed=7, example_x=x[:4], example_y=y[:4])
        return m

    full = make()
    full.fit(x, y, epochs=2, shuffle=False, print_every=0, log_style="none")
    first = make()
    first.fit(x, y, epochs=1, shuffle=False, print_every=0, log_style="none")
    checkpoint = first.save_checkpoint(tmp_path / "checkpoint")
    resumed = make()
    resumed.restore_checkpoint(checkpoint)
    resumed.fit(x, y, epochs=1, shuffle=False, print_every=0, log_style="none")
    same(full._checkpoint_payload()[0], resumed._checkpoint_payload()[0])

    if backend == "jax":
        from daehn import DAEHardNet
        from ohagi.inference import load

        path = tmp_path / "inference/metadata.json"
        resumed.export_inference(path)
        expected = resumed.predict(np.array([0.7]))[1]
        np.testing.assert_allclose(load(path).predict([0.7]), expected, atol=1e-7)
        np.testing.assert_allclose(
            DAEHardNet().load(path).predict([0.7]), expected, atol=1e-7
        )
        resumed.save_predictor(tmp_path / "full-model")
