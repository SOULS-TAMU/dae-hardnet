# Versions and migration

This guide describes **DAE-HardNet 2.0.0** on the current main branch. The previous
source is preserved in the [`vunversioned` branch](https://github.com/SOULS-TAMU/dae-hardnet/tree/vunversioned).

Problem suppliers (KKTHN/DAEHN), learning execution (OHAGI) and projection
mathematics (PURIN) have separate ownership. Benchmark signatures are retained;
the packages now use a `src/` layout. Use these public API pages rather than
copying removed internal module paths.

## Saved-model compatibility

New model saves and inference exports use the shared OHAGI JAX schemas. Legacy
readers and separate native/standalone-StableHLO deployment interfaces are not
part of the current workflow. JAX export uses StableHLO internally.

Historical benchmark artifacts can use other formats. KKTHN's archived
`kkt-hardnet-metadata-v3` models require their matching original runtime. DAEHN's
benchmark parameter-tree snapshots use the included case loaders. The filename
`metadata.json` alone does not establish compatibility. Preserve the original
artifacts and follow the benchmark repository's setup instructions.

See [inference](INFERENCE.md) for the current deployment interface and referenced
files. Orbax training checkpoints preserve a different state from inference exports.
