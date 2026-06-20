"""Wave-1 norm_dict / class_dict CLI fan-out (plan-24 §6 Wave 1 + R1.6, §5.4).

The two convenience flags ``--norm_dict`` / ``--class_dict`` restore v1's
one-flag ergonomics for the loss-weight + normalisation artifacts, fanning out
(`Salt2CLI` → `DeepMergeParser.parse_args` → `_fan_out_artifacts`) onto the
model-side consumers WITHOUT any data module or setup graph (R1.1/R1.2):

- ``norm_dict`` → the `Normaliser`'s ``norm_dict`` init_arg (its only consumer);
- ``class_dict`` → ``weight_source: {from_class_dict: <path>}`` on EACH
  `ClassificationTaskModule` whose ``weight_source`` is unset/null. Tasks that
  set ``weight_source`` explicitly (e.g. the saved run-dir config on resume) are
  LEFT ALONE.

The gates here mirror plan-24 Wave 1:

- (a) ``--print_config`` byte-equality: the two-flag form resolves to the SAME
  assembled config as today's verbose per-module override block (byte-equal on
  the norm_dict + weight_source fields — in fact on the whole model block).
- (c) a unit assertion that the fan-out lands norm_dict on the norm module +
  class_dict on each unset-weight_source classification task and leaves
  explicitly-set ones alone (incl. the resume / saved-run-dir-config case).

(Gate (b) — `parity_gn2` bitwise — is the existing `test_parity_gn2.py` control:
the CE weight is a loss buffer, so it is bitwise-invariant regardless of how the
path arrived; it never touches this CLI surface.)

Everything runs through the REAL `Salt2CLI` surface (``run=False``) with the
shipped ``gn2v2-dummy.yaml`` worked config + its documented required overrides.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import yaml

from salt.core.main import CONFIG_DIR, Salt2CLI
from salt.core.nn.tasks import ClassificationTaskModule
from salt.core.saltmodule import SaltModule
from salt.core.schema import dump_schema, save_schema
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
# the two classification tasks in gn2v2-dummy.yaml, both with weight_source unset
CLS_TASKS = ("jets_classification", "track_origin")
NORM_MODULE = "norm"


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("wave1_fanout")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # a SECOND class dict, used to prove an explicit per-task weight_source is
    # left untouched even when --class_dict is also supplied
    cd_explicit = base / "class_dict_explicit.yaml"
    write_parity_norm_dict(base / "_throwaway_nd.yaml", cd_explicit)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {
        "dir": base,
        "h5": h5_path,
        "nd": nd_path,
        "cd": cd_path,
        "cd_explicit": cd_explicit,
        "schema": schema_path,
    }


def base_overrides(data) -> list[str]:
    """The gn2v2-dummy.yaml required path overrides MINUS the norm_dict block.

    The norm_dict + per-task weight_source are deliberately omitted so the two
    flags (or the verbose block) are the ONLY source of those values.

    Returns
    -------
    list[str]
        The CLI args common to every case here.
    """
    return [
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        "--trainer.logger=false",  # opt out of the default-ON CometLogger (plan-24 W0)
    ]


def verbose_block(data) -> list[str]:
    """Today's verbose per-module override block (the form being retired).

    Returns
    -------
    list[str]
        The three ``--model.modules.*`` overrides.
    """
    cd = data["cd"]

    def ws_override(task: str) -> str:
        return f'--model.modules.{task}.init_args.weight_source={{"from_class_dict": "{cd}"}}'

    return [
        f"--model.modules.{NORM_MODULE}.init_args.norm_dict={data['nd']}",
        ws_override("jets_classification"),
        ws_override("track_origin"),
    ]


def two_flag_block(data) -> list[str]:
    """The Wave-1 two-flag form.

    Returns
    -------
    list[str]
        The ``--norm_dict`` / ``--class_dict`` flags.
    """
    return [f"--norm_dict={data['nd']}", f"--class_dict={data['cd']}"]


def make_cli(data, extra: list[str]) -> Salt2CLI:
    return Salt2CLI(args=[*base_overrides(data), *extra], run=False)


def print_config(data, extra: list[str]) -> str:
    """Capture the ``--print_config`` dump for the given override block.

    Returns
    -------
    str
        The dumped YAML config.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        make_cli(data, [*extra, "--print_config"])
    assert excinfo.value.code == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gate (a) — --print_config byte-equality (two-flag == verbose three-line)
# ---------------------------------------------------------------------------


class TestPrintConfigByteEquality:
    def test_model_block_byte_equal(self, data):
        verbose = yaml.safe_load(print_config(data, verbose_block(data)))
        two_flag = yaml.safe_load(print_config(data, two_flag_block(data)))
        # the whole assembled model block is byte-identical between the two forms
        assert two_flag["model"] == verbose["model"]

    def test_norm_dict_field_equal(self, data):
        verbose = yaml.safe_load(print_config(data, verbose_block(data)))
        two_flag = yaml.safe_load(print_config(data, two_flag_block(data)))

        def norm_path(cfg):
            return cfg["model"]["init_args"]["modules"][NORM_MODULE]["init_args"]["norm_dict"]

        assert norm_path(two_flag) == norm_path(verbose) == str(data["nd"])

    def test_weight_source_fields_equal(self, data):
        verbose = yaml.safe_load(print_config(data, verbose_block(data)))
        two_flag = yaml.safe_load(print_config(data, two_flag_block(data)))

        def ws(cfg, task):
            return cfg["model"]["init_args"]["modules"][task]["init_args"]["weight_source"]

        for task in CLS_TASKS:
            assert ws(two_flag, task) == ws(verbose, task)
            assert ws(two_flag, task) == {"from_class_dict": str(data["cd"])}


# ---------------------------------------------------------------------------
# Gate (c) — the fan-out lands on the right modules / leaves explicit ones alone
# ---------------------------------------------------------------------------


class TestFanOutInstantiated:
    def test_norm_dict_lands_on_normaliser(self, data):
        cli = make_cli(data, two_flag_block(data))
        assert isinstance(cli.model, SaltModule)
        assert str(cli.model.net[NORM_MODULE].norm_dict_path) == str(data["nd"])

    def test_class_dict_lands_on_every_unset_task(self, data):
        cli = make_cli(data, two_flag_block(data))
        for task in CLS_TASKS:
            mod = cli.model.net[task]
            assert isinstance(mod, ClassificationTaskModule)
            assert mod.weight_source == {"from_class_dict": str(data["cd"])}

    def test_class_dict_skips_non_classification_tasks(self, data):
        # track_vertexing is a VertexingTaskModule, NOT a ClassificationTaskModule,
        # so --class_dict must NOT fan a weight_source onto it (the .endswith
        # ClassificationTaskModule suffix-match in _fan_out_artifacts). Pins the
        # negative case so a future broadening of the match (e.g. to "TaskModule")
        # would be caught here.
        cli = make_cli(data, two_flag_block(data))
        vtx = cli.model.net["track_vertexing"]
        assert not isinstance(vtx, ClassificationTaskModule)
        assert getattr(vtx, "weight_source", None) is None

    def test_explicit_weight_source_left_alone(self, data):
        # a task that ALREADY sets weight_source (here jets_classification, the
        # resume / saved-run-dir-config case) must NOT be overwritten by the
        # --class_dict fan-out; the other (unset) task still gets it.
        cd_explicit = data["cd_explicit"]
        explicit_ws = (
            "--model.modules.jets_classification.init_args.weight_source="
            f'{{"from_class_dict": "{cd_explicit}"}}'
        )
        cli = make_cli(data, [*two_flag_block(data), explicit_ws])
        jets = cli.model.net["jets_classification"]
        track = cli.model.net["track_origin"]
        assert jets.weight_source == {"from_class_dict": str(cd_explicit)}  # untouched
        assert track.weight_source == {"from_class_dict": str(data["cd"])}  # fanned out

    def test_no_flags_is_a_noop(self, data):
        # with neither flag, an unset task stays unset (no accidental fan-out);
        # norm_dict supplied the verbose way still works (control that the
        # fan-out is purely additive on the new flags)
        cli = make_cli(
            data,
            [f"--model.modules.{NORM_MODULE}.init_args.norm_dict={data['nd']}"],
        )
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source is None

    def test_norm_dict_only_does_not_touch_tasks(self, data):
        # --norm_dict alone fans out to the normaliser but leaves tasks unset
        cli = make_cli(data, [f"--norm_dict={data['nd']}"])
        assert str(cli.model.net[NORM_MODULE].norm_dict_path) == str(data["nd"])
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source is None

    def test_class_dict_only_requires_norm_dict_elsewhere(self, data):
        # --class_dict alone fans out to the tasks; norm_dict must still be
        # supplied (it is REQUIRED on the Normaliser) — here via the flag too,
        # proving the two flags are independent knobs
        cli = make_cli(data, [f"--norm_dict={data['nd']}", f"--class_dict={data['cd']}"])
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source == {"from_class_dict": str(data["cd"])}
