"""``include:`` — a config names the configs it stacks on, so the user need not.

Before this, a config that was an overlay documented its bases in a header
comment and the user had to reproduce that chain on the command line
(``GN3_tracklabel`` needed five ``-c`` flags in the right order). ``include:``
moves that knowledge into the config, where it can't drift from the file it
describes.

Resolution, merge order and the null-marker rule are all checked here; the
shipped configs' own chains are exercised by the integration suite.
"""

from __future__ import annotations

import pytest
import yaml

from salt.utils.config_utils import IncludeError, expand_includes


def _write(path, data) -> str:
    path.write_text(yaml.dump(data, sort_keys=False))
    return str(path)


def _expanded(path, config_dir) -> dict:
    return yaml.safe_load(open(expand_includes(path, config_dir=config_dir)).read())


class TestMergeOrder:
    def test_including_config_wins_over_what_it_includes(self, tmp_path):
        """A file always overrides its bases — the documented stacking order."""
        _write(tmp_path / "base.yaml", {"trainer": {"max_epochs": 10, "precision": 32}})
        child = _write(
            tmp_path / "child.yaml",
            {"include": ["base.yaml"], "trainer": {"max_epochs": 50}},
        )
        cfg = _expanded(child, tmp_path)
        assert cfg["trainer"]["max_epochs"] == 50, "the including config must win"
        assert cfg["trainer"]["precision"] == 32, "unrelated base keys must survive"

    def test_later_includes_win_over_earlier(self, tmp_path):
        _write(tmp_path / "a.yaml", {"trainer": {"max_epochs": 1, "a": True}})
        _write(tmp_path / "b.yaml", {"trainer": {"max_epochs": 2}})
        cfg = _expanded(_write(tmp_path / "c.yaml", {"include": ["a.yaml", "b.yaml"]}), tmp_path)
        assert cfg["trainer"] == {"max_epochs": 2, "a": True}

    def test_dicts_merge_and_lists_replace(self, tmp_path):
        """The same rule stacked --config files already follow."""
        _write(tmp_path / "a.yaml", {"m": {"x": 1, "y": 2}, "streams": ["jets", "tracks"]})
        child = _write(
            tmp_path / "b.yaml",
            {"include": ["a.yaml"], "m": {"y": 99, "z": 3}, "streams": ["jets"]},
        )
        cfg = _expanded(child, tmp_path)
        assert cfg["m"] == {"x": 1, "y": 99, "z": 3}
        assert cfg["streams"] == ["jets"], "lists replace rather than concatenate"

    def test_null_marker_is_kept_not_resolved(self, tmp_path):
        """Deletion happens at assembly, so the null must survive expansion.

        Resolving it here would let a later config resurrect an entry an
        earlier one deleted.
        """
        _write(tmp_path / "a.yaml", {"modules": {"keep": 1, "drop": {"cls": "X"}}})
        child = _write(
            tmp_path / "b.yaml", {"include": ["a.yaml"], "modules": {"drop": None}}
        )
        cfg = _expanded(child, tmp_path)
        assert cfg["modules"]["drop"] is None
        assert "drop" in cfg["modules"], "the marker itself must reach assembly"


class TestResolution:
    def test_relative_to_the_including_config_first(self, tmp_path):
        (tmp_path / "family").mkdir()
        _write(tmp_path / "family" / "base.yaml", {"who": "sibling"})
        _write(tmp_path / "base.yaml", {"who": "config_dir"})
        child = _write(tmp_path / "family" / "child.yaml", {"include": ["base.yaml"]})
        assert _expanded(child, tmp_path)["who"] == "sibling"

    def test_falls_back_to_the_configs_root(self, tmp_path):
        (tmp_path / "family").mkdir()
        _write(tmp_path / "shared.yaml", {"who": "config_dir"})
        child = _write(tmp_path / "family" / "child.yaml", {"include": ["shared.yaml"]})
        assert _expanded(child, tmp_path)["who"] == "config_dir"

    def test_absolute_path_is_used_as_given(self, tmp_path):
        target = _write(tmp_path / "abs.yaml", {"who": "absolute"})
        child = _write(tmp_path / "child.yaml", {"include": [target]})
        assert _expanded(child, tmp_path)["who"] == "absolute"

    def test_missing_include_names_both_places_tried(self, tmp_path):
        child = _write(tmp_path / "child.yaml", {"include": ["nope.yaml"]})
        with pytest.raises(IncludeError, match="Tried:"):
            expand_includes(child, config_dir=tmp_path)


class TestRecursion:
    def test_includes_are_transitive(self, tmp_path):
        _write(tmp_path / "a.yaml", {"a": 1})
        _write(tmp_path / "b.yaml", {"include": ["a.yaml"], "b": 2})
        cfg = _expanded(_write(tmp_path / "c.yaml", {"include": ["b.yaml"], "c": 3}), tmp_path)
        assert cfg == {"a": 1, "b": 2, "c": 3}

    def test_cycles_are_reported_with_the_path(self, tmp_path):
        _write(tmp_path / "a.yaml", {"include": ["b.yaml"]})
        _write(tmp_path / "b.yaml", {"include": ["a.yaml"]})
        with pytest.raises(IncludeError, match="include cycle"):
            expand_includes(str(tmp_path / "a.yaml"), config_dir=tmp_path)

    def test_self_include_is_a_cycle(self, tmp_path):
        _write(tmp_path / "a.yaml", {"include": ["a.yaml"]})
        with pytest.raises(IncludeError, match="include cycle"):
            expand_includes(str(tmp_path / "a.yaml"), config_dir=tmp_path)


class TestPassThrough:
    def test_a_config_without_includes_is_returned_unchanged(self, tmp_path):
        """The common case costs nothing and keeps the original path in errors."""
        path = _write(tmp_path / "plain.yaml", {"trainer": {"max_epochs": 1}})
        assert expand_includes(path, config_dir=tmp_path) == path

    def test_include_key_never_survives_expansion(self, tmp_path):
        """jsonargparse would reject it as an unknown key."""
        _write(tmp_path / "a.yaml", {"a": 1})
        cfg = _expanded(_write(tmp_path / "b.yaml", {"include": ["a.yaml"]}), tmp_path)
        assert "include" not in cfg

    def test_editing_an_included_config_re_expands(self, tmp_path):
        """The cache is keyed on the whole chain, not the entry point."""
        base = tmp_path / "base.yaml"
        _write(base, {"trainer": {"max_epochs": 1}})
        child = _write(tmp_path / "child.yaml", {"include": ["base.yaml"]})
        assert _expanded(child, tmp_path)["trainer"]["max_epochs"] == 1
        _write(base, {"trainer": {"max_epochs": 999}})
        assert _expanded(child, tmp_path)["trainer"]["max_epochs"] == 999
