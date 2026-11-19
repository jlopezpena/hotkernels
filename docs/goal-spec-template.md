# Goal-spec template

The prompt for a `/goal` run that writes or tunes one kernel. Replace `<kernel>` and `<arch>`, then paste everything below the line.

Before starting a run, the kernel's `spec.py` and `naive.c` must already be on the `main` branch. A person writes or reviews those; the agent does not.

---

Make `kernels/<kernel>/<arch>/kernel.s` as fast as you can while keeping it correct. Read `AGENTS.md` first and follow it exactly.

**Target.** A statistically significant overall speedup of at least 10% over naive C, as reported in the "naive C" column of `uv run judge run <kernel>`.

**Do not stop at the target.** Once you reach 10%, keep going until you have exhausted the optimisations you can think of.

**How to iterate.**

1. Read `kernels/<kernel>/<arch>/attempts.md` so you do not repeat earlier attempts on this architecture. If the kernel has assembly for another architecture, read that `kernel.s` and the `attempts.md` beside it for the idea and the proofs. Do not start from that code, and do not treat an attempt rejected there as rejected here.
2. Make one change to `kernel.s`.
3. Run `uv run judge check <kernel>`. If it fails, fix or revert.
4. Run `uv run judge compare <kernel> --against HEAD`. Keep the change only if the result is `significantly faster`; otherwise revert it.
5. Log the attempt in `attempts.md`, kept or not.
6. If kept, commit `kernel.s` and `attempts.md` together.

If there is no `kernel.s` yet, start with the simplest correct version, get `check` passing, commit it, and iterate from there.

**On x86-64.** Try each candidate listed for the kernel's family in `docs/x86-64-design.md` under "Candidates per family", or say why it could not be tried. A candidate that would remove a caveat of the ARM64 kernel, change what the binding must guarantee, or need a judge rule changed is reported, not kept on your own authority.

**When to stop.** Stop when eight consecutive attempts have produced no significant improvement and you have no untried idea that you expect to help.

**If the target cannot be reached.** Report your best result and explain why 10% was not possible. Typical reasons: the compiler already vectorises the naive loop well, or the operation is limited by memory speed and not by computation.

**Final report.** When you stop, run `uv run judge run <kernel>` and report:

- the overall speedup over naive C and over NumPy, with their 95% intervals
- the per-size table
- the optimisations that worked, in order of how much they gave
- the optimisations that did not work, and why
- anything you left untried, and why
- on x86-64, each of the family's candidates: kept, rejected with the judge's result, or not tried and why

**Limits.** Edit only `kernels/<kernel>/<arch>/kernel.s` and `kernels/<kernel>/<arch>/attempts.md`. Never pass `--unlocked`. If you think the spec, the naive reference or the judge is wrong, stop and report it.
