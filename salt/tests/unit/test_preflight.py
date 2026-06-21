"""Tests confirming the norm-dict preflight is gone (plan 01, self-normalising).

The v2 ``Normaliser`` now learns its statistics online over valid (non-padded)
objects during training, so there is NO norm-dict file to validate: the
``preflight()`` / ``materialise()`` hooks were removed, ``SaltModule._run_preflights``
is a clean no-op for it, and ``salt2 graph validate`` needs no norm dict.
"""

from __future__ import annotations

from salt.core.graph.spec import Mode, flatten_spec
from salt.core.nn.modules import Normaliser
from salt.core.saltmodule import SaltModule
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, gn2v2_sources

LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}


class TestNoNormDictPreflight:
    def test_normaliser_has_no_preflight_or_materialise(self):
        """The norm-dict lifecycle hooks are gone (stats are learned online)."""
        norm = Normaliser(streams=["jets", "tracks"], global_object="jets")
        assert not hasattr(norm, "preflight")
        assert not hasattr(norm, "materialise")
        assert not hasattr(norm, "norm_dict_path")

    def test_norm_dict_arg_is_ignored_no_file_read(self, tmp_path):
        """A nonexistent norm_dict path is accepted and never read."""
        norm = Normaliser(streams=["tracks"], norm_dict=tmp_path / "absent.yaml")
        # declare_io is file-free regardless of the bogus path
        io = norm.declare_io(Mode.FIT)
        assert "inputs.tracks" in flatten_spec(io.requires)

    def test_run_preflights_is_a_clean_noop(self):
        """SaltModule._run_preflights no longer trips on the Normaliser."""
        modules = build_gn2v2_modules("ignored.yaml")
        model = SaltModule(modules, lrs=LRS)
        model.compile_mode(Mode.FIT, flatten_spec(gn2v2_sources()))
        model._ensure_bound()  # noqa: SLF001 - exercising the setup path piecewise
        # no module implements preflight() anymore -> no error, even with no file
        model._run_preflights()  # noqa: SLF001
