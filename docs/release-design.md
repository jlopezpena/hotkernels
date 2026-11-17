# Releases: how `hotkernels` gets to PyPI

This is a proposal. It answers the core design's note under [Later](core-design.md#later) that "the release pipeline and its GitHub setup get their own design document", and [issue 24](https://github.com/sofeikov/hotkernels/issues/24).

It covers when a release happens, how its notes are written, how the version is set, and what runs on GitHub to build and publish it. What goes into a wheel and how it is built for every supported machine is [issue 19](https://github.com/sofeikov/hotkernels/issues/19); this document needs its answer before the first upload to PyPI, and says where.

## Summary

- **Merging to `main` does not publish.** `main` is always releasable; a release is a deliberate step taken every few weeks, or at once for a fix to a wrong result.
- **Each pull request leaves a note.** A pull request that changes what a user of the package gets adds one short file to `release-notes/upcoming/`. CI fails such a pull request without one. The notes pile up between releases.
- **A release is a pull request.** It gathers the notes into one document, adds a summary written by hand that spans every change since the last release, re-measures the headline numbers on native hardware, and bumps the version. Reviewing it is reviewing the release.
- **Merging that pull request publishes.** A workflow sees a version on `main` with no tag, builds the wheels, tests them, waits for the maintainer's approval, uploads to PyPI, and creates the tag and the GitHub release with the notes as its text.
- **No passwords or tokens are stored.** PyPI accepts the upload from that one workflow through trusted publishing.

## Why not publish on every merge

Publishing on every merge would release the judge's own changes, documentation fixes and half-finished kernel work as new versions, each with a one-line note. A user reading the history of the package would learn little from it. The project also has a rule that a release cannot be automated around: timings are only meaningful on native hardware, so the speedups a release claims have to be taken on a real machine by a person, not on a CI runner.

Releasing by hand from a laptop has the opposite problem: nothing checks that what was uploaded is what `main` holds, and the notes are written from memory.

The proposal sits between the two. Changes accumulate on `main`, each with its note written while its author still knows what it does. A release is cut when there is something worth telling users, and everything after the decision to cut it runs in CI.

## The note each pull request leaves

A note is a Markdown file of one to five sentences, written for someone who uses `hotkernels`, not for someone who works on it:

```
release-notes/upcoming/31.kernel.md
```

```markdown
`np.cumsum` on `int32` is served by a kernel: about 11x faster than NumPy at a million elements.
```

The file name is the pull request number and a type. The types become the sections of the release notes:

| Type | Section | For |
|---|---|---|
| `kernel` | New kernels | A NumPy call that a kernel now serves, or a new element type |
| `speed` | Faster | A kernel or the binding got faster |
| `fix` | Fixes | A wrong result, a crash, or a call that went to the wrong place |
| `compat` | Compatibility | A Python or NumPy version gained or lost, a new platform, a change to what `install()` installs by default |
| `change` | Other changes | Anything else a user would notice, such as `report()` output |

**Which pull requests need one.** Those that change `hotkernels/`, or a kernel the package installs (`kernels/<name>/arm64/kernel.s` or `naive.c`). A CI job checks this. A pull request in those paths that a user would not notice, such as a comment or a refactor, gets the `no release note` label and the job passes. Pull requests that touch only the judge, the survey, the documentation site or `docs/` need nothing.

**One note per pull request, not per commit.** A kernel run keeps many commits; its pull request leaves one `speed` note saying what the run achieved.

**Numbers in a note are allowed but provisional.** A note may quote the judge's figure at the time. The release pull request re-measures, and the release notes quote the re-measured figures.

This is how NumPy writes its own release notes (`doc/release/upcoming_changes/`), with the same tool, [towncrier](https://towncrier.readthedocs.io/). Towncrier renders the notes through a Jinja template, which is the repository's rule for generated text, and its `check` command is the CI job.

## The release pull request

Anyone can open it; the maintainer decides to merge it. One command prepares it:

```
uv run release prepare 0.2.0
```

It:

1. Sets the version to `0.2.0` in the one place the package keeps it ([Versions](#versions)).
2. Runs towncrier, which writes `release-notes/0.2.0.md` from the notes in `upcoming/` and deletes them.
3. Leaves a **Highlights** section at the top of that file to be written by hand, with the list of pull requests merged since the last tag as material.

Then, on the maintainer's machine:

4. `uv run judge selftest`, `uv run delivery check` and `uv run delivery numpy-tests`.
5. The timings: `uv run delivery time`, and `uv run judge run` for any kernel whose note claims a speed. The results files under `delivery/results/` and `kernels/` are updated in the same pull request, and the README's tables if they changed.
6. The highlights: a few paragraphs on what this release means for a user, with the re-measured numbers. This is where the notes become longer than any one pull request. An agent can draft it from the notes and the pull request descriptions; the maintainer edits it.

The finished notes look like this:

```markdown
# hotkernels 0.2.0

## Highlights

Three more NumPy calls are served by kernels, ... (hand-written)

Measured on an Apple M1 Pro, NumPy 2.5.5, Python 3.13.

## New kernels

- `np.cumsum` on `int32` is served by a kernel: ... (#31)

## Fixes

- `np.clip(x, lo, hi, out=x)` gave wrong results on some data and now goes to NumPy. (#34)
```

Fixes to wrong results come first within Fixes, and are named in the highlights. A user who stays on the previous version needs to know what they are exposed to.

### When to release

There is no fixed calendar. A release is cut when `release-notes/upcoming/` holds something a user would want, which is expected every few weeks while kernels are being added. Two exceptions:

- **A fix to a wrong result is released at once**, as a patch release, even if it is the only note.
- **Notes that wait a long time are a prompt.** A scheduled workflow opens an issue when notes have been waiting in `upcoming/` for more than four weeks. It only reminds; it does not release.

## Versions

Semantic versioning, read for this package:

| Part | Bumped when | Example |
|---|---|---|
| Minor (`0.2.0`) | New kernels or operations, a change to what `install()` does by default, a supported Python or NumPy version dropped | Most releases |
| Patch (`0.2.1`) | Only fixes and speed | A wrong-result fix released at once |
| Major (`1.0.0`) | When the maintainer decides the public API (`install`, `uninstall`, `enabled`, `report`) is stable | Not planned |

**One place holds the version.** Today it is written twice, in `hotkernels/pyproject.toml` and in `hotkernels/meson.build`. `meson-python` can take it from `meson.build` when `pyproject.toml` declares it dynamic, so that becomes the one place. The package gains `hotkernels.__version__`, and `report()` prints it, so a bug report says which release it is about.

**Pre-releases** use the same path: a release pull request for `0.3.0rc1` publishes a pre-release, which `pip` installs only when asked. Worth it for a release that changes default behaviour; otherwise optional.

The root package, `hotkernels-judge`, is not published and keeps no version of its own beyond what `uv` needs.

## What runs on GitHub

A new workflow, `.github/workflows/release.yml`.

**Trigger.** Every push to `main`. The first job reads the version from `meson.build` and checks whether the tag `v<version>` exists. If it does, which is every push except a release pull request's merge, the workflow stops there in a few seconds. It can also be started by hand, to retry a release that failed part-way, or to upload to TestPyPI as a rehearsal.

**Jobs, in order:**

| Job | Runs on | What it does |
|---|---|---|
| `version` | Ubuntu | Reads the version; stops the workflow if it is already tagged. Fails if `release-notes/<version>.md` is missing or `upcoming/` is not empty |
| `build` | macOS 15, Apple Silicon | Builds the source distribution and one wheel per supported Python with [cibuildwheel](https://cibuildwheel.pypa.io/). What the wheel contains and the compiler flags it uses are [issue 19](https://github.com/sofeikov/hotkernels/issues/19) |
| `test` | macOS 15, Apple Silicon | Installs each wheel into a clean environment, outside the checkout, and runs `delivery check` against it. This is the test that the wheel works without the repository next to it |
| `publish` | Ubuntu, environment `pypi` | Waits for the maintainer's approval, then uploads with [trusted publishing](https://docs.pypi.org/trusted-publishers/) |
| `release` | Ubuntu | Creates the tag `v<version>` on the merged commit and a GitHub release with `release-notes/<version>.md` as its text and the wheels attached |

**The order protects what cannot be undone.** PyPI never accepts the same version twice, even after a file is deleted. Nothing is uploaded until every wheel is built and tested, and the tag is created only after the upload succeeds, so a failed run leaves the version untagged and the workflow can be run again.

**The approval** is the last point to stop a release. Merging the release pull request has already said "release this"; the approval says "these built files". It costs one click, and can be dropped once the pipeline has run a few times without surprises.

**Timings are not taken in CI.** The runners are too noisy, as `ci.yml` already notes. The numbers in the notes come from step 5 of the release pull request.

## Setup outside the repository

Done once by the maintainer:

| Where | What |
|---|---|
| PyPI | The name `hotkernels` is held already, by a placeholder `0.0.1` whose repository link still uses the old name `kernel-arena`; the first release replaces that metadata. Add a trusted publisher to the project: repository `sofeikov/hotkernels`, workflow `release.yml`, environment `pypi` |
| TestPyPI | The same, for rehearsals |
| GitHub, environments | Create `pypi` with the maintainer as required reviewer, limited to `main` |
| GitHub, labels | Create `no release note` |
| GitHub, branch protection | Require `ci` and the release-note check on `main` |

## Order of work

| Step | What | Depends on |
|---|---|---|
| 1 | The version in one place, `__version__`, `report()` prints it | Nothing |
| 2 | towncrier, `release-notes/`, the template, `uv run release prepare`, the CI check for notes, the label | Nothing. From here on pull requests leave notes |
| 3 | `release.yml` up to `release`, publishing to TestPyPI only | Step 1; a wheel that builds, even if it is not yet the final one |
| 4 | Answers to issue 19: a source distribution that builds without the checkout, the flags for C kernels, `python -m hotkernels` | Its own decisions |
| 5 | The first release, `0.1.0` or `0.2.0`, to PyPI | Steps 1 to 4, and [issue 18](https://github.com/sofeikov/hotkernels/issues/18) if the release is to claim a NumPy range |

Steps 1 and 2 can land now; they are useful before anything is published, because notes start piling up from the next pull request.

## Out of scope

- What a wheel contains, its compiler flags, and the source distribution's contents: [issue 19](https://github.com/sofeikov/hotkernels/issues/19).
- The NumPy versions the package claims: [issue 18](https://github.com/sofeikov/hotkernels/issues/18).
- conda-forge. Worth it after PyPI works.
- Publishing the release notes on the documentation site. A small change to `docsite/` once there are notes to show.

## Decisions still open

Each has a recommendation, which the implementation follows until it is decided otherwise.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Note per pull request, or write the notes at release time** from the list of merged pull requests? | A note per pull request. It is written while the change is fresh, by the agent or person who made it, and costs a minute. Writing from the list at release time means reconstructing what each change meant to a user |
| 2 | **towncrier, or a few lines of our own?** | towncrier. NumPy uses it for the same job, it renders with Jinja, and its `check` command is the CI job. Our own would be a short `typer` command; worth it only if towncrier's file naming gets in the way |
| 3 | **What starts the release: merging the release pull request, or pushing a tag?** | Merging. It is one act after a review, and the tag is created by CI on a commit that passed CI. A pushed tag can point at any commit, and the notes would need a second place to live |
| 4 | **The approval before upload** | Keep it for the first few releases |
| 5 | **First version on PyPI**: `0.1.0`, what the package says today, or `0.2.0`? | `0.1.0`. Only the `0.0.1` placeholder is published, and the first release notes describe the package as it stands rather than a change from an earlier one |
| 6 | **Release-note check on pull requests made by agents during a kernel run** | The same rule. A kernel run's pull request leaves one `speed` note; it is the agent's summary of the run, which it writes anyway |
