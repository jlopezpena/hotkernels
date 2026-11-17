"""Release tooling for the hotkernels package: `uv run release <command>`.

Release notes are gathered from one short note per pull request in
`release-notes/upcoming/`, compiled by towncrier into `release-notes/<version>.md`
by `prepare`, and published by pushing the tag that `tag` creates. The release
workflow checks the tag with `verify`, and pull requests are checked for their
note with `check-note`. See docs/release-design.md.
"""
