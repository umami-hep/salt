"""Static contract of ``setup/Dockerfile``: the system packages the shipped image must carry."""

from __future__ import annotations

from pathlib import Path

# salt/tests/unit/test_dockerfile_contract.py -> parents[3] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "setup" / "Dockerfile"


def _apt_install_tokens(text: str) -> list[str]:
    """Whitespace tokens of every ``RUN apt-get ... install`` instruction, continuation lines joined."""
    logical_lines: list[str] = []
    buffer = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.endswith("\\"):
            buffer += line[:-1] + " "
        else:
            buffer += line
            logical_lines.append(buffer)
            buffer = ""
    if buffer:
        logical_lines.append(buffer)

    tokens: list[str] = []
    for logical_line in logical_lines:
        stripped = logical_line.strip()
        if stripped.startswith("RUN") and "apt-get" in stripped and "install" in stripped:
            tokens.extend(stripped.split())
    return tokens


def test_dockerfile_exists():
    assert DOCKERFILE.is_file()


def test_apt_line_installs_graphviz():
    tokens = _apt_install_tokens(DOCKERFILE.read_text())
    assert tokens, "no RUN apt-get install instruction found"
    assert "graphviz" in tokens, (
        "setup/Dockerfile's apt-get install line is missing `graphviz` — without the `dot` "
        "binary, salt fit's graph render degrades to .dot-only "
        "(salt/callbacks/graph_artifacts.py::_render)"
    )


def test_helper_joins_continuation_lines():
    dockerfile_text = "RUN apt-get update && apt-get install -y \\\n    foo bar \\\n    && rm -rf x"
    tokens = _apt_install_tokens(dockerfile_text)
    assert "bar" in tokens

    non_apt_text = "RUN echo graphviz"
    assert _apt_install_tokens(non_apt_text) == []
