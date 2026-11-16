# hotkernels

Fast kernels for the NumPy operations where NumPy is slow, delivered as a zero-friction drop-in: hand-written assembly where it can actually win, plain C where the compiler is already enough.

Kernels are written by coding agents iterating against a judge. The judge checks every kernel against NumPy and times it against NumPy and naive `clang -O3` C.

## What is here

- **The judge** (`judge/`): checks a kernel for correctness and memory safety, and reports how fast it is.
- **Fixture kernels** (`judge/fixtures/`): `argmax_f32` and `compress_f32`, used to test the judge itself.
- **The survey** (`survey/`): measures NumPy against naive C per operation, to decide which kernels are worth writing. Recorded results are in `survey/results/`.
- **Kernels** (`kernels/`): none yet.

Supported platform: macOS on Apple Silicon (ARM64).

## Using the judge

Requires the Xcode command line tools and [uv](https://docs.astral.sh/uv/).

```bash
uv run judge selftest            # prove the judge catches known-bad kernels
uv run judge check argmax_f32    # lint and correctness for one kernel
uv run judge run argmax_f32      # correctness, then a timing report
```

## Using the survey

```bash
uv run survey run                 # measure every candidate; takes about 15 minutes
uv run survey run --only argmax   # measure the candidates whose name matches
uv run survey report survey/results/apple-m1-pro.json   # regenerate a report
```

## Documentation

- [docs/core-design.md](docs/core-design.md): what the project is building and why
- [docs/judge-design.md](docs/judge-design.md): how kernels are checked and timed
- [docs/survey-design.md](docs/survey-design.md): where NumPy is slow, and which kernels to build in what order
- [AGENTS.md](AGENTS.md): rules for coding agents working in this repository
- [kernels/README.md](kernels/README.md): how to add a kernel

## Licence

MIT
