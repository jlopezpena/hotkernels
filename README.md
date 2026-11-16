# hotkernels

Hand-written assembly kernels for the NumPy operations where assembly can actually win, delivered as a zero-friction drop-in.

Kernels are written by coding agents iterating against a judge. The judge checks every kernel against NumPy and times it against NumPy and naive `clang -O3` C.

## What is here

- **The judge** (`judge/`): checks a kernel for correctness and memory safety, and reports how fast it is.
- **Fixture kernels** (`judge/fixtures/`): `argmax_f32` and `compress_f32`, used to test the judge itself.
- **Kernels** (`kernels/`): none yet.

Supported platform: macOS on Apple Silicon (ARM64).

## Using the judge

Requires the Xcode command line tools and [uv](https://docs.astral.sh/uv/).

```bash
uv run judge selftest            # prove the judge catches known-bad kernels
uv run judge check argmax_f32    # lint and correctness for one kernel
uv run judge run argmax_f32      # correctness, then a timing report
```

## Documentation

- [docs/core-design.md](docs/core-design.md): what the project is building and why
- [docs/judge-design.md](docs/judge-design.md): how kernels are checked and timed
- [AGENTS.md](AGENTS.md): rules for coding agents working in this repository
- [kernels/README.md](kernels/README.md): how to add a kernel

## Licence

MIT
