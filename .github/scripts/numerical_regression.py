import argparse


import json

import os

from pathlib import Path

import subprocess

import sys

import tempfile

import jax

import jax.numpy as jnp

import numpy as np


def reload_check(path, inputs, expected, framework, full_metadata=None):
    code = """
import importlib.abc, json, sys
import numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'kkthn','daehn','nlpoptnet','flax','orbax'}:
   raise AssertionError('Deployment imported '+fullname)
sys.meta_path.insert(0,Block())
from ohagi.inference import load
predictor=load(sys.argv[1])
x=json.loads(sys.argv[2]); expected=json.loads(sys.argv[3])
for _ in range(25):
 np.testing.assert_allclose(predictor.predict(x),expected,atol=1e-8,rtol=1e-7)
 np.testing.assert_allclose(predictor.predict_jax(x),expected,atol=1e-8,rtol=1e-7)
"""
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(path),
            json.dumps(np.asarray(inputs).tolist()),
            json.dumps(np.asarray(expected).tolist()),
        ],
        check=True,
        timeout=180,
    )
    if framework == "kkthn":
        from kkthn import KKTHardNet

        facade = KKTHardNet().load(full_metadata or path)
    elif framework == "daehn":
        from daehn import DAEHardNet

        facade = DAEHardNet().load(path)
    else:
        from nlpoptnet import NLPOptNet

        facade = NLPOptNet().load(full_metadata or path)
    np.testing.assert_allclose(
        np.asarray(facade.predict(np.asarray(inputs)[None])).reshape(-1),
        np.asarray(expected).reshape(-1),
        atol=1e-8,
        rtol=1e-7,
    )


def daehn(root):
    from daehn.dae import (
        Coordinate,
        DifferentialState,
        DerivativeSpec,
        EquationBlock,
        DAESystem,
        TaylorNeighborhood,
        ForwardDifference,
        CentralDifference,
        compile_dae,
        DAETrainer,
        DAETrainingConfig,
    )
    from daehn import MLPBackbone
    from ohagi.inference import export, load

    records = {}
    for name, strategy in [
        ("taylor", TaylorNeighborhood(deltas=(0.01,), order=1)),
        ("forward", ForwardDifference(deltas=(0.01,))),
        ("central", CentralDifference(deltas=(0.01,))),
    ]:
        state = DifferentialState("y")
        system = DAESystem(
            coordinates=(Coordinate("t"),),
            differential_states=(state,),
            derivatives=(DerivativeSpec(state, (1,)),),
            differential_equations=EquationBlock(
                lambda c, s, d, p: jnp.array([d[0] + s[0]]),
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
        np.testing.assert_allclose(
            decoded["states"] + decoded["derivatives"], 0, atol=1e-7
        )

        def graph(inputs):
            decoded, *_ = trainer.model.forward(
                result.parameters, jnp.repeat(inputs, 2, axis=0), jnp.empty((2, 0))
            )
            return decoded["states"][:1]

        path = root / name / "metadata.json"
        export(path, graph, input_size=1, output_size=1, framework="daehn")
        expected = load(path).predict([0.7])
        reload_check(path, [0.7], expected, "daehn")
        records[name] = {"epochs": len(result.history), "prediction": expected.tolist()}
    return records


def run_case(directory):
    """Exercise the installed frontend through short training and fresh-process reload."""
    jax.config.update("jax_enable_x64", True)
    original = Path.cwd()
    os.chdir(directory)
    try:
        return daehn(Path(directory))
    finally:
        os.chdir(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as directory:
        result = run_case(directory)
    baseline = root / ".github/validation/numerical.json"
    if args.write_baseline:
        baseline.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        expected = json.loads(baseline.read_text())

        def check(actual, reference):
            if isinstance(reference, dict):
                assert actual.keys() == reference.keys()
                for key in reference:
                    check(actual[key], reference[key])
            else:
                np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)

        check(result, expected)
    (root / "numerical-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
