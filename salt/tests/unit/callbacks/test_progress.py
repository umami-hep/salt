"""Tests for `salt.callbacks.ProgressBar` (split from test_callbacks.py)."""

from __future__ import annotations

from salt.callbacks import Checkpoint, ProgressBar
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict


class TestProgressBar:
    def test_is_stock_tqdm_under_salt_name(self):
        from lightning.pytorch.callbacks import TQDMProgressBar

        bar = ProgressBar(refresh_rate=50)
        assert isinstance(bar, TQDMProgressBar)  # the stock bar (v1 base.yaml:38-39)
        assert bar.refresh_rate == 50  # init args pass straight through

    def test_registered_by_default_in_base(self, tmp_path):
        # base.yaml ships checkpoint (Checkpoint) + progress (ProgressBar); the
        # CLI assembles both into trainer.callbacks (run=False, no fit)
        from salt.schema import dump_schema, save_schema
        from salt.tests.unit.test_main import make_cli
        from salt.testing.inputs import write_dummy_file

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        cli = make_cli({"dir": tmp_path, "h5": h5, "nd": nd, "schema": schema})
        assert any(isinstance(cb, Checkpoint) for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ProgressBar) for cb in cli.trainer.callbacks)
