# Workloads over time: a history of recorded runs

This is a proposal for review. It extends [workloads-design.md](workloads-design.md): every recorded run of `uv run delivery workloads` is kept, and a new command shows how each workload's speedup changed from commit to commit. It builds on the runner in [pull request 90](https://github.com/sofeikov/hotkernels/pull/90) and needs it merged first.

## Summary

- **Today a recorded run replaces the one before it.** `delivery workloads --record` writes `delivery/results/<machine>-workloads.json`, and the previous run on that machine is gone. Nothing can say whether a workload became faster or slower, or when.
- **Proposed: one file per recorded run**, under `delivery/results/history/<machine>/`, named by date and commit, never changed after it is written. One file per run means that two pull requests that each record a run never conflict.
- **Each run records what can move the figures**, not only the commit: the NumPy version, the library versions, the load average and whether the tree was clean. The M4 Max figures ([README.md](../README.md#results)) show why: `np.searchsorted` went from 6.40x to 1.48x faster than NumPy between two machines, and the kernel had not changed. A change of NumPy version within one machine can do the same.
- **`uv run delivery history`** prints, per workload and size, the speedup at each recorded run, in commit order, and marks the runs where a version changed or the machine was in use.
- **Who records is the open part.** By hand on a quiet machine, as now. Recording from CI waits for the dedicated machine of [issue 51](https://github.com/sofeikov/hotkernels/issues/51).
- **Not proposed now:** asv, and running old commits to fill in history. See [Considered and not proposed](#considered-and-not-proposed).

## What is wrong today

Three tools record timings, and each keeps only the latest run per machine:

| Tool | Writes | A new run |
|---|---|---|
| `judge run --record` | `results/<kernel>/<arch>-<machine>.json` | Replaces it |
| `delivery time --record` | `delivery/results/<machine>.json` and `.md` | Replaces them |
| `delivery workloads --record` | `delivery/results/<machine>-workloads.json` and `.md` | Replaces them |

Git keeps the old versions, so history exists in principle. In practice a recorded file changes only when someone chooses to record, its old versions are spread across unrelated commits, and nothing reads them back.

This proposal covers the workloads. They are what a user gets, and they are few. The judge and `delivery time` can take the same format later; see [Decisions still open](#decisions-still-open).

## Design

### One file per recorded run

`--record` keeps writing the latest report as now, and also writes:

```
delivery/results/history/<machine>/<UTC date and time>-<commit>.json
```

for example `delivery/results/history/apple-m4-max/2026-11-18T1115Z-6a4fe1c.json`. The machine part is the slug `--record` already uses. A file is never edited after it is written. A run that turns out to be wrong is deleted, in a commit that says why.

Why not one file per machine with a line appended per run: two pull requests that each append a line conflict at the end of the file, and the second has to be merged by hand. Files with distinct names merge without conflict.

### What a run records

The same JSON `--record` writes today (`meta` and `rows`), with these fields added to `meta`:

| Field | Is |
|---|---|
| `commit` | The full commit hash. `git` already holds the short one |
| `commit_date` | The commit's date, so that runs can be put in commit order without the repository |
| `recorded_at` | When the run started, in UTC |
| `dirty` | Whether tracked files differed from the commit. Untracked files do not count, unlike the judge's `git_dirty` |
| `sizes` | The sizes run |
| `hotkernels` | The package version |

`libraries`, `numpy`, `python`, `os` and `load_average` are already there.

`--record` refuses to write history from a dirty tree, since the commit would not say what was run. The latest report is still written, as now.

### Series and breaks

A **series** is the runs of one workload at one size on one machine. Within a series, consecutive runs are comparable only if NumPy, Python and the workload's own libraries have the same versions. Where one of them changes, `delivery history` marks a break, and does not compare the two runs on either side of it. A speedup that moves at a break is NumPy's or the library's doing until shown otherwise.

Runs are ordered by commit date, and runs of the same commit by `recorded_at`. A commit that is not on `main` is shown, marked as such: a pull request may record a run to show its own effect.

### The command

```
uv run delivery history [--machine SLUG] [--workload NAME] [--size S|M|L]
```

Defaults: the machine it runs on, every workload, size `M`. For each series, one line per run: date, commit, NumPy version, the time without and with hotkernels, the speedup, and marks for a break, a busy machine (load average over `BUSY_LOAD`) and a commit not on `main`. Rendered from a Jinja template, as the other reports are.

It reads only the files under `delivery/results/history/`, so it runs on any machine and needs none of the workload libraries.

### Noise

How far a workload's figure moves between two runs of the same commit is not measured yet. In [pull request 90](https://github.com/sofeikov/hotkernels/pull/90) two workloads were run twice at the same size, at a load average of about 2, and agreed within 1%; two pairs say little. `delivery history` does not call a change a regression or a gain. It shows the figures and the marks. Deciding what is significant needs repeated runs of one commit, which is what `judge compare` does for a kernel; doing the same for workloads is left until the dedicated machine exists.

### The site

Not in the first step. Once a series has a few runs, the docsite can draw one chart per workload from the same files: speedup against commit date, with breaks shown as gaps.

## Considered and not proposed

| Option | Why not now |
|---|---|
| [asv](https://asv.readthedocs.io/) | It is made for this: it builds the package at each commit, stores results per commit and machine, and publishes charts. But it builds each commit in its own environment, and hotkernels is a `uv` workspace member built by meson. Teaching asv that build costs more than the history format here. The decision against asv as the runner, in [workloads-design.md](workloads-design.md#considered-and-not-proposed), stands; this would be asv as the store, which the files above do with less |
| Running old commits to fill in history | Possible without asv: build `hotkernels/` at an old commit in a worktree and run today's workloads against it. Worth doing once to see how the binding's changes since phase 3 show in the workloads. Not part of the regular record, since it times old code on today's machine and libraries |
| Recording on every merge to `main` from CI | The self-hosted Mac runs four jobs at once and hands work to GitHub's virtual machines when it is offline. Neither gives figures worth keeping. Waits for [issue 51](https://github.com/sofeikov/hotkernels/issues/51) |
| A database, or results kept outside the repository | The files are small: a run of four workloads at two sizes is about 7 KB. A thousand runs fit in the repository |

## Documents that change

| Document | Change |
|---|---|
| `docs/workloads-design.md` | A link here from "What the results are used for" |
| `README.md` | `delivery history` in "Using the delivery tool" |

No release note: nothing under `hotkernels/` changes.

## Order of work

1. This document, reviewed. Pull request 90 merged.
2. The history files written by `--record`, the new `meta` fields, the refusal on a dirty tree, and `delivery history`. One run recorded per machine at hand.
3. Once a series has several runs: the chart on the site.

## Exit criterion

- `delivery workloads --record` writes a history file, and refuses to from a dirty tree.
- `delivery history` shows two recorded runs of one series in order, and a break where the NumPy version changes. Checked with a run under each of two NumPy versions.
- Two branches that each record a run merge without conflict.

## Out of scope

- History for `judge run` and `delivery time`.
- Deciding automatically whether a change is significant.
- Recording from CI.

## Decisions still open

1. **The judge and `delivery time`.** Their recorded runs have the same problem. The same layout, `history/<machine>/<run>.json` beside each results directory, would serve them. Proposed: yes, in a later pull request, once this one has been used.
2. **Which runs are kept.** Every recorded run, or only runs on `main` and a quiet machine. Proposed: every run, marked; a reader can filter, and a run that is not kept cannot be looked at again.
3. **One backfill run.** Whether to time the workloads once against the binding at a few past commits (the first `install()`, issue 17's function hooks, the two-sizes change), to give the history a start. Proposed: yes, as a one-off, recorded with its own mark.
