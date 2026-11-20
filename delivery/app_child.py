"""The process `delivery.app_profile` samples: one application, run in a loop.

Run as `python -m delivery.app_child NAME SIZE MODE SECONDS`. SIZE is one of
`apps.SIZE_NAMES`; MODE is "plain" or "installed". It reads its arguments by
hand because it is not a command for people.
"""
from __future__ import annotations

import sys
import time
import warnings

import hotkernels

from . import apps
from .app_profile import READY

# The mode in which hotkernels is installed before the loop.
INSTALLED = "installed"


def main(name: str, size: str, mode: str, seconds: float) -> None:
    """Run an application until the time is up.

    Args:
        name: One of `apps.NAMES`.
        size: One of `apps.SIZE_NAMES`.
        mode: "plain" or "installed".
        seconds: How long to keep running it after the first run.
    """
    warnings.simplefilter("ignore")
    app = apps.load(name)
    data = app.make_data(size)
    if mode == INSTALLED:
        hotkernels.install()
    app.run(*data)
    # The parent waits for this line before it starts the sampler, so the
    # samples hold no imports and no building of data. It is a signal between
    # two processes, not output for a reader.
    sys.stdout.write(READY + "\n")
    sys.stdout.flush()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.run(*data)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
