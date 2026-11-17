# Distribution: a source distribution, other platforms, and `python -m hotkernels`

This is a proposal. It settles what [issue 19](https://github.com/sofeikov/hotkernels/issues/19) left open after release 0.1.0, which shipped macOS ARM64 wheels for CPython 3.11 to 3.14 and nothing else. How a release is made is [release-design.md](release-design.md); this document is about what a release contains.

## Summary

- **A source distribution that builds anywhere.** When the source distribution is made, the build copies in everything it needs from the checkout: the kernel sources, the generated header and kernel table, and the judge's compiler flags. Building it then needs no checkout and no judge. Releases publish it next to the wheels, and the wheels are built from it, which proves it builds.
- **Other platforms get a package that installs and does nothing.** Built from the source distribution on Linux, Windows or an Intel Mac, the package has no extension; `install()` does nothing and `report()` says why. That build needs no compiler. No separate pure-Python wheel is published.
- **`python -m hotkernels script.py`** runs a script with hotkernels installed, with no change to the script. It reads its few options by hand, an exception to the rule that command lines use `typer`, so the package gains no dependency.
- **Jinja stays a dependency of the package.** Decided in review.

## Where things stand

Release 0.1.0 is wheels only. `pip install hotkernels` works on an Apple Silicon Mac with CPython 3.11 to 3.14, and fails everywhere else with "no matching distribution". It fails the same way on an Apple Silicon Mac with a Python that has no wheel yet, such as 3.15 when it comes out.

The build cannot run from anything but a checkout, because:

- `hotkernels/tools/generate.py` imports the judge (`judge.build`, `judge.spec`) to read `kernels/*/spec.py` and write the C prototypes and the kernel table;
- the extension is compiled from `kernels/*/arm64/kernel.s` and `kernels/*/naive.c`, outside the package directory;
- `meson.build` declares C as the project's language, so even a build that compiles nothing needs a C compiler.

## The source distribution

`meson dist`, which meson-python runs to make a source distribution, packs the files git tracks under `hotkernels/` and then runs the project's dist scripts on the result. One dist script, `generate.py dist`, writes the **kernel set** into the source distribution's `kernel-set/` directory:

| File | What it holds |
|---|---|
| `kernels.h` | The C prototype of every kernel, from its spec |
| `kernels.json` | Per kernel: assembly or C, exact or within tolerance. Installed with the package for `report()` |
| `sources/<name>.s`, `sources/<name>.c` | Each kernel's shipped source: its ARM64 assembly where it has one, its naive C otherwise |
| `assembly-sources.txt`, `c-sources.txt` | The file names in `sources/`, one per line |
| `c-flags.txt` | The judge's C flags, one per line, with `-mcpu=native` as the judge has it |

A build from a checkout writes the same kernel set into its build directory with `generate.py write`, so `meson.build` has one way to build: it reads a kernel set, from `kernel-set/` when the source distribution carries one, from the build directory otherwise. The `kernel_cpu` option replaces `-mcpu=native` in `c-flags.txt` as it does today.

What this keeps true:

- **`kernels/` stays the only source.** The copies exist only inside a source distribution, made from the checkout at that moment.
- **A source distribution builds what was judged.** The same assembly, the same C with the same flags, and on the machine that builds it, `-mcpu=native` as the judge used.
- **The generator is the one place that knows the layout of `kernels/`.** `meson.build` knows only the kernel set.

**The wheels are built from the source distribution.** The release workflow makes the source distribution first, then points cibuildwheel at it. A source distribution that cannot build fails the release before anything is uploaded, so it is tested by every release and every pull request that runs the workflow.

## Other platforms

The extension is built only for macOS on ARM64, as today. Elsewhere `meson.build` adds the C language only inside that branch, so a build on Linux, Windows or an Intel Mac compiles nothing, needs no compiler, and installs the Python package alone. `install()` then finds no extension and does nothing, and `report()` says `needs macOS on ARM64, and this is linux on x86_64`. That is what the core design asked for: hotkernels never breaks a program, it only fails to speed it up.

A release workflow job on Linux builds the source distribution into a wheel, installs it and checks exactly that.

**No pure-Python wheel.** Issue 19 suggested one, to make the do-nothing install explicit. With a source distribution that builds without a compiler, it adds nothing on other platforms. It would also do harm on an Apple Silicon Mac with a Python that has no wheel yet: pip prefers any wheel to building from source, so that user would get the do-nothing package instead of building the real one.

## `python -m hotkernels`

```
python -m hotkernels [--inexact] [--report] script.py [args ...]
python -m hotkernels [--inexact] [--report] -m module [args ...]
```

It calls `install()`, then runs the script or module as `python` would: as `__main__`, with `sys.argv` set to the script and its arguments, and with the script's directory first on `sys.path`. Its exit status is the script's.

| Option | Effect |
|---|---|
| `--inexact` | `install(inexact=True)` |
| `--report` | Print `report()` to standard error when the script ends, however it ends |

**Read by hand, not with `typer`.** The repository's rule is that command lines use `typer`. That would make `typer` and its dependencies (`click`, `rich` and others) a dependency of every installation of a package whose point is one line and no friction. The command has two flags and then passes everything else through untouched, which `typer` is not shaped for in any case. AGENTS.md gets the exception: code that ships in the package reads `sys.argv` itself and stays free of command-line dependencies; development tools still use `typer`.

**Output goes through `logging`**, on a logger of the package's own with its own handler on standard error. The root logger is the user's program's, which is running in the same process, so the command does not configure it.

## Jinja

`report()` renders through a Jinja template, by the repository's rule on generated text, so `jinja2` and `markupsafe` are installed with the package. They are imported only when a report is printed. Decided in review: they stay.

## Order of work

| Step | What |
|---|---|
| 1 | The kernel set: `generate.py write` and `dist`, `meson.build` reading it, the C language only where it is used |
| 2 | `python -m hotkernels`, and the exception in AGENTS.md |
| 3 | The release workflow: make the source distribution, build the wheels from it, publish it, and the Linux job |
| 4 | READMEs and the documentation site: how to install, and the command |

Released in the next version. Issue 19 closes with it.

## Out of scope

- Kernels for other platforms. Linux on ARM64 is the obvious next one: the assembly is the same instruction set, with a different object format and calling details.
- conda-forge.

## Decisions made in review

| Question | Decision |
|---|---|
| `python -m hotkernels` with `typer`, a runtime dependency, or by hand as an exception to the rule? | By hand, with the exception written into AGENTS.md |
| Keep `jinja2` as a runtime dependency? | Yes |
