# Spike: jsonargparse `dict[str, BaseClass]` config mechanism (design §5.3, risk §11.4)

- **Date**: 2026-06-11 (M1 stage E)
- **jsonargparse version**: 4.46.0 (salt container, `/opt/conda/lib/python3.11/site-packages/jsonargparse`)
- **Test evidence**: `salt/tests/core/test_jsonargparse_spike.py` — 11 passed, 2 xfailed (strict)
  in the salt container. Plain `jsonargparse.ArgumentParser` (no LightningCLI), toy classes only.

## Overall verdict

**The design's mechanism is viable. The ModuleSpec-dataclass fallback (risk §11.4) is NOT needed.**
The instantiation layer — `dict[str, Base]` from `{class_path, init_args}` blocks, subclass
validation, `--print_config`, dotted CLI overrides, env vars, callbacks-dict assembly — all work
natively. Two **config-file merge** behaviors fail natively, both with the same root cause
(`merge_config` treats dict leaves atomically), and both are restored by a ~10-line
`ArgumentParser.merge_config` override (`DeepMergeParser` in the test file, spike-validated)
plus `| None` value typing. Crucially, the ModuleSpec fallback would NOT have fixed these two
failures anyway: `dict[str, ModuleSpec]` is still a dict leaf and would be replaced wholesale by a
later config file in exactly the same way.

## Per-capability results

| # | Capability | Verdict | Test |
|---|------------|---------|------|
| 1 | `dict[str, Base]` instantiation from YAML class_path/init_args | **PASS** | `test_dict_of_base_instantiation_from_yaml` |
| 2 | `--print_config` round-trip | **PASS** | `test_print_config_round_trip` |
| 3 | Null-deletion from a second config file | **FAIL natively / PASS with shim** | `test_null_deletion_via_second_config_file` (xfail) + 2 workaround tests |
| 4 | Dotted CLI overrides into dict values | **PASS** | `test_dotted_cli_override_into_dict_value` |
| 5 | Env-var overrides (`default_env=True`) | **PASS** | `test_env_var_override` |
| 6 | Deep-merge of the dict across two config files | **FAIL natively / PASS with shim** | `test_deep_merge_across_config_files` (xfail) + 3 workaround tests |
| 7 | callbacks-dict → `trainer.callbacks` assembly | **PASS** | `test_callbacks_dict_assembly` |

### 1. Basic instantiation — PASS

`parser.add_class_arguments(Model, "model")` with `modules: dict[str, ToyModule] | None` gives
per-entry `class_path`/`init_args` parsing, subclass-specific init args (`heads` on `Encoder`),
base-class defaults filled in, subclass validation (a non-`ToyModule` class path is a parse error),
and YAML insertion order preserved through `instantiate_classes`.

### 2. `--print_config` — PASS

Exits 0; output contains the full `class_path`/`init_args` blocks with defaults made explicit;
feeding the printed YAML back through `parse_string` yields a namespace equal to the original
(`cfg_again.model == cfg_orig.model` and identical `parser.dump`).

### 3. Null-deletion via a second config file — FAIL natively

Minimal repro (second file `{model: {modules: {decoder: null}}}` stacked on a base file):

- With `dict[str, ToyModule]`: parse error — `None` fails value validation
  ("Not a valid subclass of ToyModule. Got value: None").
- With `dict[str, ToyModule | None]`: parses, but the result is `{'decoder': None}` only —
  the earlier `encoder` key is **gone**, because the second file replaced the dict (see 6).

**Workarounds (both spike-validated):**

- **Config-file null** — `DeepMergeParser.merge_config` override (union dict leaves key-by-key,
  drop `None` values) + `| None` value typing: `decoder` is deleted, siblings survive
  (`test_null_deletion_workaround_merge_config_override`).
- **CLI null** — `--model.modules.decoder=null` with `| None` value typing natively keeps
  siblings and sets the entry to `None`; the framework filters `None` entries at assembly
  (`test_null_deletion_workaround_optional_values_cli`). CLI args do not pass through
  `merge_config`, so the assembly-time filter is needed regardless.

### 4. Dotted CLI overrides — PASS

All three forms work and preserve sibling keys and unmentioned init_args:

- `--model.modules.encoder.init_args.dim=128` (the design §5.3 form),
- the short form `--model.modules.encoder.dim=96`,
- adding a brand-new key with a JSON block: `--model.modules.head='{"class_path": ..., "init_args": {...}}'`.

Mechanism: dotted CLI args become `NestedArg`s, which the dict typehint adapter merges into the
previous value (`_typehints.py:925-933`: `val = {**prev_val, key: ...}`).

### 5. Env-var overrides — PASS

With `ArgumentParser(default_env=True, env_prefix="SALT2")`, `SALT2_MODEL__MODULES` (dots in the
dest become `__`, uppercased) set to a JSON/YAML dict parses and instantiates correctly.
Caveats: the env var sets the **whole dict** (no per-key env merge), and precedence is
CLI/config-file > env > defaults — a `--config` file on the command line wins over the env var.

### 6. Deep-merge across two config files — FAIL natively

Minimal repro: base file defines `{encoder, decoder}`, second file adds `{head}` → parsed dict is
`{head}` only. Root cause (jsonargparse 4.46.0 source): `ActionConfigFile` merges files via
`ArgumentParser.merge_config` (`_core.py:1406`), which ends in `Namespace.update`
(`_namespace.py:261`) — dict-typed leaves are set wholesale, never unioned. Dict types also do not
support the `+` append syntax (`ActionTypeHint.supports_append` is sequence-only,
`_typehints.py:482`).

Two important *native* nuances that narrow the gap:

- **Per-entry merge IS native**: a second file restating a key with only `init_args` inherits the
  earlier `class_path` and merges init_args (`heads: 4` survives a `dim: 999` update) — only the
  *sibling keys* are lost (`test_per_entry_init_args_merge_is_native`). The documented
  "restate class_path to merge init_args" rule is actually laxer in 4.46.0: class_path need not be
  restated; a *changed* class_path discards previous init_args.
- **Dotted CLI overrides already merge** (capability 4), so the gap is config-files only.

**Workaround (spike-validated)**: `DeepMergeParser` — override `merge_config`, union dict-typed
leaves key-by-key (`{**old, **new}`) and drop `None` values before delegating to the standard
merge. With it: later file adds keys with earlier keys surviving, per-entry init_args updates keep
siblings, and file-level `null` deletes (`test_deep_merge_workaround_*`). The override applies to
every dict-typed leaf merged from config files, which is exactly the §5.3 semantics for
`modules:` / `writers:` / `callbacks:`; if some future dict field needs replace semantics, the
override can be restricted by action/destination.

### 7. callbacks-dict assembly — PASS

`--callbacks` as `dict[str, Callback]` plus `--trainer.callbacks` as `list[Callback] | None`
parse side by side from one YAML; the §5.3 one-key override
(`--callbacks.checkpoint.init_args.monitor=...`) works; framework-side assembly
`[*callbacks.values(), *trainer.callbacks]` preserves YAML insertion order and types.

## Recommendation for the salt2 CLI (M1+)

1. Use plain `dict[str, Base | None]` annotations for module/writer/callback dicts
   (the `| None` admits the null-deletion sentinel).
2. Subclass the parser with the `DeepMergeParser.merge_config` override (10 lines) to restore
   §5.3 cross-file dict semantics (add/update/delete-by-null).
3. Filter `None` entries at assembly time (covers CLI-set nulls, which bypass `merge_config`).
4. Keep the two strict xfail markers in the spike test: if a container jsonargparse upgrade makes
   native behavior match the design, the xpass will fail loudly and the shim can be retired.

No change to the YAML surface of design §5.3 is required.
