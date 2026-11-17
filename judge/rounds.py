"""Times contenders in several rounds, each in a process of its own.

Within one process, where a library is loaded and where its buffers lie is
fixed, and so is whatever that placement costs or saves each contender. To
see how much of a difference is placement, the measurement is repeated in
fresh processes: the operating system loads everything at new addresses each
time. `judge compare` uses this; see COMPARE_ROUNDS in judge/bench.py.

Run as `python -m judge.rounds` this module is one round: it times the
libraries it is given and writes the samples to a file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import typer

from . import bench as B
from . import cores
from .build import BUILD
from .spec import Kernel, load_spec

# How a contender is passed to a round: its name, this, and its library.
_NAMED = "="
# Sizes are passed to a round as one argument, separated by this.
_SIZE_SEPARATOR = ","

app = typer.Typer(add_completion=False)


def _to_json(data: B.Measurements) -> str:
    """Write measurements out as JSON.

    Args:
        data: Samples by table name, size and contender.

    Returns:
        The same structure with sizes as strings and samples as lists.
    """
    return json.dumps({table: {str(n): {name: times.tolist() for name, times in by_name.items()}
                               for n, by_name in by_size.items()}
                       for table, by_size in data.items()})


def _from_json(text: str) -> B.Measurements:
    """Read measurements written by `_to_json`.

    Args:
        text: The JSON text.

    Returns:
        Samples by table name, size and contender, sizes as integers and
        samples as arrays again.
    """
    return {table: {int(n): {name: np.array(times) for name, times in by_name.items()}
                    for n, by_name in by_size.items()}
            for table, by_size in json.loads(text).items()}


@app.command()
def run(
    kernel_dir: Path = typer.Argument(..., help="Directory holding the kernel's spec.py."),
    harness: Path = typer.Argument(..., help="Compiled harness library for this kernel."),
    out: Path = typer.Argument(..., help="File that receives the samples as JSON."),
    sizes: str = typer.Argument(..., help="Element counts to time, separated by commas."),
    samples: int = typer.Argument(..., help="The most samples per contender per size."),
    contender: list[str] = typer.Option(..., help="A contender as name=library; give one per contender."),
) -> None:
    """Time the given libraries against each other and write the samples to a file.

    Args:
        kernel_dir: Directory holding the kernel's spec.py.
        harness: Compiled harness library for this kernel.
        out: File that receives the samples.
        sizes: Element counts to time, separated by commas.
        samples: The most samples per contender per size.
        contender: Each contender as `name=library`.
    """
    k = load_spec(kernel_dir)
    loaded = B.Harness(k, harness)
    # Each entry keeps its library object, which has to stay loaded while it is timed.
    libraries = [(name, *B.function_address(Path(library), k.name))
                 for name, library in (entry.split(_NAMED, 1) for entry in contender)]
    addresses = {name: address for name, address, _ in libraries}
    counts = [int(n) for n in sizes.split(_SIZE_SEPARATOR)]
    out.write_text(_to_json(B.measure(k, loaded, addresses, counts, samples)))


def measure(k: Kernel, kernel_dir: Path, harness: Path, libraries: dict[str, Path],
            sizes: list[int], samples: int, rounds: int,
            core: cores.Claim | None) -> list[B.Measurements]:
    """Time compiled contenders in several rounds, each in a new process.

    Args:
        k: The kernel's spec.
        kernel_dir: Directory holding its spec.py.
        harness: The compiled harness library.
        libraries: Contender name -> compiled library holding it.
        sizes: Element counts to time.
        samples: The most samples per contender per size, in each round.
        rounds: How many rounds to run.
        core: The core this process is pinned to. Every round is pinned to
            the same one; None where threads are not pinned.

    Returns:
        The measurements of each round, in the shape `judge.bench.measure`
        returns.

    Raises:
        RuntimeError: A round ended with an error.
    """
    environment = dict(os.environ)
    if core is not None:
        environment[cores.INHERITED_CPU] = str(core.cpu)
    out = BUILD / k.name / f"round-{os.getpid()}.json"
    command = [sys.executable, "-m", "judge.rounds", str(kernel_dir), str(harness), str(out),
               _SIZE_SEPARATOR.join(str(n) for n in sizes), str(samples)]
    for name, library in libraries.items():
        command += ["--contender", f"{name}{_NAMED}{library}"]
    measured = []
    for _ in range(rounds):
        done = subprocess.run(command, env=environment, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"a timing round failed: {done.stderr.strip()}")
        measured.append(_from_json(out.read_text()))
    out.unlink(missing_ok=True)
    return measured


if __name__ == "__main__":
    app()
