# Docs development

The documentation is built with [Zensical](https://zensical.org/), the successor to
MkDocs Material. The site configuration lives in `zensical.toml` at the repository root.

## Building the docs locally

Install the documentation dependencies from the repository root:

```bash
uv sync --only-group docs
```

Build the documentation with Zensical:

```bash
uv run zensical build --strict
```

The generated site is written to `public/`. The `--strict` flag fails the build on broken
links or references, matching what the CI pipeline runs. To preview the documentation while
editing, run:

```bash
uv run zensical serve
```

## API reference

The API reference pages under `docs/api/` are generated from the source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). Each page contains `:::` blocks pointing at
the relevant `salt` module; the shared handler options are configured in `zensical.toml`.

## Merge request previews

The docs are deployed for commits on the `main` branch to
[ftag-salt.docs.cern.ch](https://ftag-salt.docs.cern.ch/). Merge requests opened in the main
repository also get a public preview at:

```text
https://ftag-salt.docs.cern.ch/mr-<MR-number>/
```

The preview is published by the `pages_review` job and expires automatically one week after
the pipeline runs.
