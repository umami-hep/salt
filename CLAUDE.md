# CLAUDE.md — salt

Conventions for agents (and humans) working in this repository.

## Docstring policy

Docstrings document **contracts, once, at the contract's home**. The volume rule:
module docstrings ≤3 lines; most functions ≤1 line; full documentation only where a
reader genuinely needs it. Specifically:

1. **Base classes and protocols carry the docs.** Methods that define an interface
   (`forward`, `declare_io`, `bind`, `setup`, `process`, `get_output`, `onnx_outputs`,
   `read`, `columns`, ...) get a full docstring ONCE, on the base/protocol.
2. **Concrete implementations skip the docstring**, or give a single line stating only
   what differs from the base contract. Never restate the inherited contract.
3. **Public API gets full numpy-style docstrings**: classes referenced from YAML configs
   (`class_path` targets), names exported via `__all__`, and CLI entry points — these
   render into the docs site. `Parameters`/`Returns` sections belong here and (sparingly)
   wherever behavior is genuinely non-obvious.
4. **Private helpers (`_name`): keep the contract, cut the ceremony.** One line when the
   body is obvious. When the helper encodes real semantics (pipeline order, edge-case
   rules like "NaN fails the cut", conventions), KEEP those lines — compressed, no
   `Parameters`/`Returns` blocks (the signature + type hints carry that), no prose
   restating the code, no provenance.
5. **Properties, getters, setters: one line maximum.**
6. **No history or provenance in docstrings** — no "ported from", "absorbed at wave X",
   "was previously" narration. Git history is the archive (see
   `docs/architecture.md` §Parity-closure).
7. Tests: one-line docstrings; the test name should carry most of the meaning.

## Comparisons against old code

Never vendor or freeze old code/outputs into the tree for comparison. v1 salt lives at
pin `29c67a1`; upstream MaskFormer equivalence closed at `6570e85`. Check out the pin.

## Running tests

Unit suite: `pytest salt/tests/unit` (CPU-safe). Integration suite is GPU/`--run-integration`
gated. Any invocation that builds a DataLoader from a shipped config must override
`num_workers` (shipped configs assume large training machines).
