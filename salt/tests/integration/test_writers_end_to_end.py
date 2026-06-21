"""End-to-end writer test: ``salt2 fit`` -> ``salt2 test`` on the dummy config.

Runs the REAL ``salt2 fit`` -> ``salt2 test`` surface on a tmp dummy file and
checks the v1 output-file shape (one H5 next to the checkpoint, input copies +
prob columns + VertexIndex + mask, padded-position encodings); negative tests
cover the TEST dead-preds hard error, the writer-less eval refusal, the
wrong-dtype-at-compile rejection, and the no-ckpt single-config contract
(design §9.5 M3).

This is an INTEGRATION test (it builds + fits + evaluates a real model on CPU).
Split out of the former monolithic ``test_writers.py``; the per-writer unit
tests live under ``tests/unit/writers/``. Shared fixtures / toy-writers /
constants come from ``salt.tests._fixtures.writers_common``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.core.main import main
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES
from salt.tests._fixtures.writers_common import (  # noqa: F401  (data is a fixture)
    DUMMY_CFG,
    L_FILE,
    RUN_NAME,
    NamedFeatureWriter,
    data,
)

# ---------------------------------------------------------------------------
# end to end: salt2 fit -> salt2 test on the dummy config (design §9.5 M3)
# ---------------------------------------------------------------------------


def overrides(data) -> list[str]:
    return [
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        "--trainer.accelerator=cpu",
        # base2 ships a default-ON CometLogger (plan-24 Wave 0); turn it off on the
        # fit fixture so the end-to-end run emits no offline Comet archive (and
        # lr_monitor drops). On the salt2 test path this is already forced off.
        "--trainer.logger=false",
        # null-delete the base2 ProgressBar (D2 default-on) — can't combine the
        # stock enable_progress_bar=false with a configured ProgressBar callback
        "--callbacks.progress=null",
    ]


@pytest.fixture(scope="module")
def ckpt(data, tmp_path_factory) -> Path:
    fit_dir = tmp_path_factory.mktemp("m3_fit")
    rc = main([
        "fit",
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        *overrides(data),
        f"--trainer.default_root_dir={fit_dir}",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
    ])
    assert rc == 0
    ckpts = sorted(fit_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint under {fit_dir}"
    return ckpts[0]


def run_test_cli(data, ckpt: Path, extra: list[str] | None = None) -> int:
    return main([
        "test",
        "--config",
        str(DUMMY_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        "--data.num_test=300",
        # keep run artifacts (GraphArtifacts plan/graph files) out of the cwd
        f"--trainer.default_root_dir={data['dir']}",
        *overrides(data),
        *(extra or []),
    ])


class TestSalt2TestEndToEnd:
    def test_eval_file_v1_layout(self, data, ckpt):
        rc = run_test_cli(data, ckpt)
        assert rc == 0
        # exactly ONE h5 next to the checkpoint, v1 path contract (PW:164-171)
        outputs = list(ckpt.parent.glob("*.h5"))
        assert len(outputs) == 1
        expected = ckpt.parent / f"{ckpt.stem}__test_ttbar.h5"
        assert outputs == [expected]
        run_cols_jets = [f"{RUN_NAME}_p{c}" for c in "bcu"]
        run_cols_origin = [f"{RUN_NAME}_p{c}" for c in ORIGIN_CLASSES]
        with h5py.File(expected) as f:
            assert set(f.keys()) == {"jets", "tracks"}
            assert f.attrs["writer_version"]
            jets, tracks = f["jets"][:], f["tracks"][:]
        with h5py.File(data["h5"]) as f:
            src_jets = list(f["jets"].dtype.names)
            src_tracks = list(f["tracks"].dtype.names)
            valid = f["tracks"]["valid"][:300]
        assert len(jets) == 300
        assert tracks.shape == (300, L_FILE)
        # v1 per-group layout: input copies, task columns, mask last
        assert list(jets.dtype.names) == src_jets + run_cols_jets
        assert list(tracks.dtype.names) == src_tracks + run_cols_origin + ["VertexIndex", "mask"]
        # conversions ran: probabilities, not logits
        prob_sum = sum(jets[c].astype("f8") for c in run_cols_jets)
        assert np.allclose(prob_sum, 1.0, atol=1e-3)
        # padded-position encodings (design §8): probs 0.0, vertexing -inf-cast-int
        padded = tracks["mask"]
        assert (padded == ~valid).all()
        assert (tracks[run_cols_origin[0]][padded] == 0.0).all()
        assert (tracks["VertexIndex"][padded] == np.int64(-2147483648)).all()
        # float input copies downcast to f4, ints untouched (v1 'full' policy)
        assert jets.dtype["pt"] == np.dtype("f4")

    def test_half_precision_v1_flag(self, data, ckpt, tmp_path):
        # M3-review fix: the advertised v1 half_precision compat surface has
        # coverage — float columns (probs AND input copies) land as f2,
        # integer/bool columns untouched (the shared ftag H5Writer policy)
        out = tmp_path / "half.h5"
        rc = run_test_cli(
            data,
            ckpt,
            extra=["--writers.half_precision=true", f"--writers.output={out}"],
        )
        assert rc == 0
        with h5py.File(out) as f:
            jets_dtype, tracks_dtype = f["jets"].dtype, f["tracks"].dtype
        assert jets_dtype["pt"] == np.dtype("f2")  # input copy downcast
        assert jets_dtype[f"{RUN_NAME}_pb"] == np.dtype("f2")  # prob column
        assert tracks_dtype["VertexIndex"] == np.dtype("i8")  # ints untouched
        assert tracks_dtype["mask"] == np.dtype("?")  # bools untouched

    def test_artifacts_land_next_to_checkpoint(self, data, ckpt):
        # M3-review HIGH fix: on the test path the GraphArtifacts default is
        # the checkpoint dir (with the eval H5), not the cwd-derived log dir
        rc = run_test_cli(data, ckpt)
        assert rc == 0
        plan = ckpt.parent / "plan_test.txt"
        assert plan.exists()
        text = plan.read_text()
        # the writer-sinks table answers "which writer consumes preds.X"
        assert "# writer sinks (design §8)" in text
        assert "tasks (TaskWriter):" in text
        assert "preds.tracks.track_origin" in text
        assert (ckpt.parent / "resolved_io.yaml").exists()

    def test_no_ckpt_fallback_globs_v2_checkpoints(self, data, ckpt, tmp_path):
        # D2: the salt.core.callbacks.Checkpoint port names checkpoints
        # 'epoch=NNN-loss=...' under ckpts/ (the v1 run-dir layout) and the
        # fallback globs {ckpts,checkpoints}/ next to the saved config — salt2
        # test works without --ckpt_path on a v2-trained run dir
        assert "loss=" in ckpt.name  # the v1 Checkpoint filename contract
        assert ckpt.parent.name == "ckpts"  # the D2 run-dir layout
        config = ckpt.parent.parent / "config.yaml"
        assert config.is_file()  # the saved run config next to ckpts/
        out = tmp_path / "fallback.h5"
        rc = main([
            "test",
            "--config",
            str(config),
            f"--data.test_file={data['h5']}",
            "--data.num_test=300",
            f"--writers.output={out}",
            "--callbacks.progress=null",
        ])
        assert rc == 0
        assert out.exists()

    def test_custom_writer_journey(self, data, ckpt, tmp_path):
        # W4 shape: subclass + a few config lines -> a new column end to end
        out = tmp_path / "custom.h5"
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                '--writers.modules.n_tracks={"class_path": '
                '"salt.tests._fixtures.writers_common.JetCountWriter"}',
                f"--writers.output={out}",
            ],
        )
        assert rc == 0
        with h5py.File(out) as f:
            jets = f["jets"][:]
        assert "n_valid_tracks" in jets.dtype.names
        with h5py.File(data["h5"]) as f:
            counts = f["tracks"]["valid"][:300].sum(-1).astype("i4")
        assert (jets["n_valid_tracks"] == counts).all()

    def test_custom_writer_resolves_features_by_name(self, data, ckpt, tmp_path):
        # M3-review ergonomics fix: WriteCtx.feature_fields lets a custom
        # writer resolve a feature column by NAME (no index arithmetic)
        out = tmp_path / "by_name.h5"
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                '--writers.modules.eta={"class_path": '
                '"salt.tests._fixtures.writers_common.NamedFeatureWriter"}',
                f"--writers.output={out}",
            ],
        )
        assert rc == 0
        with h5py.File(out) as f:
            got = f["jets"][NamedFeatureWriter.COLUMN][:]
        with h5py.File(data["h5"]) as f:
            expected = f["jets"]["eta_btagJes"][:300].astype("f4")
        assert got.tobytes() == expected.tobytes()

    def test_dead_preds_hard_error(self, data, ckpt, capsys):
        # narrowing TaskWriter to the jets task leaves the two track preds unconsumed
        rc = run_test_cli(
            data, ckpt, extra=['--writers.modules.tasks.init_args.tasks=["jets_classification"]']
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "consumed by NO writer" in err
        assert "track_origin" in err
        assert "track_vertexing" in err
        # §4.2-exemplar attribution (M3-review fix): culprit writer address
        # with the excluded tasks, per-task config addresses, and the
        # null-deletion workaround for train-only aux tasks
        assert "writers.modules.tasks.init_args.tasks" in err
        assert "excludes" in err
        assert "model.modules.track_origin" in err
        assert "--model.modules.track_origin=null" in err

    def test_wrong_dtype_writer_rejected_at_compile(self, data, ckpt, capsys):
        # the FD §2.7 ask end to end: a deliberately wrong-dtype writer input
        # is caught at salt2 test SETUP (validate_specs on the TEST path), loud,
        # before the first batch — not a confusing in-write() failure
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                '--writers.modules.wrong={"class_path": '
                '"salt.tests._fixtures.writers_common.WrongDtypeMaskWriter"}',
            ],
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "dtype mismatch on 'masks.tracks'" in err
        assert "writer 'wrong'" in err

    def test_writerless_eval_refused(self, data, ckpt, capsys):
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                "--writers.modules.inputs_copy=null",
                "--writers.modules.tasks=null",
                "--writers.modules.pad_mask=null",
            ],
        )
        assert rc == 1
        assert "at least one writer" in capsys.readouterr().err

    def test_no_ckpt_needs_single_config_with_losses(self, data, capsys):
        # no --ckpt_path: the v1 best-epoch glob needs loss= checkpoint names
        rc = main([
            "test",
            "--config",
            str(DUMMY_CFG),
            f"--data.test_file={data['h5']}",
            *overrides(data),
        ])
        assert rc == 1
        assert "ckpt" in capsys.readouterr().err
