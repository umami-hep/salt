# MNIST with salt

Salt is built for jet tagging, but nothing in its core knows about jets.
In this tutorial you train an MNIST digit classifier end-to-end using only:

- **one custom data reader** (~80 lines, lives in *your* workspace, not in salt), and
- **existing salt modules** for everything else — normalisation, embedding,
  classification head, loss, and H5 output writing — wired together in a single
  YAML config.

You will run `salt graph validate`, `salt fit`, and `salt test`, then compute the
test accuracy (expect **~0.95 or better**) from the evaluation H5 with five lines
of `h5py`. Everything runs on CPU in a few minutes.

## Prerequisites

You need salt installed ([Setup](../setup.md)). The short version:

```bash
git clone https://gitlab.cern.ch/aft/algorithms/salt.git
cd salt
pip install -e .
cd ..
```

!!! info "Running in a container?"

    If you use salt from an Apptainer/Singularity image, install it inside the
    image's environment — `apptainer exec <image> pip install --user --no-deps -e .`
    from the salt clone — and note the `salt` CLI then lands in `~/.local/bin`,
    which must be on `PATH` inside the container. Pass the import path of
    step 4 as `apptainer exec --env PYTHONPATH=$PWD <image> salt ...` rather
    than relying on an `export` in your host shell. Shell globs (like the
    checkpoint path in step 7) expand on the host, not in the container — use
    the literal filename, or run the glob from a shell inside the container.

Now create an empty working directory. All commands below run from it, in order:

```bash
mkdir mnist-tutorial && cd mnist-tutorial
```

## 1. Get the data

MNIST ships as four files in the IDX format — a binary header (magic number,
dimension sizes) followed by raw `uint8` data. Salt has no IDX reader; that is
the point of this tutorial.

```bash
mkdir data && cd data
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz \
     https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz \
     https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz \
     https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

That is 60,000 training and 10,000 test images (28×28 pixels) with digit labels.

## 2. Write the reader

A reader is the only piece of code you write today. It subclasses
`salt.data.base.Reader` and turns "files on disk" into named **streams** of
numpy arrays. Everything downstream — normalisation, the model, label handling,
output writing — is configured, not coded.

Save the following as `my_mnist/reader.py`, and create an empty
`my_mnist/__init__.py` next to it:

```bash
mkdir my_mnist && touch my_mnist/__init__.py
```

```python
"""A salt Reader for the MNIST IDX format."""

from pathlib import Path

import numpy as np

from salt.data.base import Reader, WorkerCtx
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec

N_PIXELS = 28 * 28


def _read_idx(path: Path) -> np.ndarray:
    """Parse an IDX file: magic (2 bytes zero, 1 dtype, 1 ndim), dims, raw data."""
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        dims = [int.from_bytes(f.read(4), "big") for _ in range(magic & 0xFF)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def _labels_path(images_path: Path) -> Path:
    """train-images-idx3-ubyte -> train-labels-idx1-ubyte."""
    return images_path.with_name(images_path.name.replace("images-idx3", "labels-idx1"))


class IdxReader(Reader):
    """Serve one IDX images/labels pair as a single fixed-length 'mnist' stream."""

    def __init__(self, filename: str | Path | None = None, num: int = -1) -> None:
        super().__init__()
        self.filename = Path(filename) if filename is not None else None
        self.num = num
        self._images: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    @property
    def streams(self) -> tuple[str, ...]:
        return ("mnist",)

    def with_source(self, filename, num: int = -1, vds_path=None, stage=None) -> "IdxReader":
        del vds_path, stage  # single-file reader: nothing to build, nothing stage-specific
        clone = IdxReader(filename=filename, num=num)
        clone.name = self.name
        return clone

    def prepare(self) -> None:
        if self._images is not None:
            return
        images = _read_idx(self.filename).reshape(-1, N_PIXELS).astype(np.float32)
        labels = _read_idx(_labels_path(self.filename)).astype(np.int64)
        n = images.shape[0] if self.num < 0 else min(images.shape[0], self.num)
        self._images, self._labels = images[:n], labels[:n]

    def __len__(self) -> int:
        self.prepare()
        return len(self._images)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat = {
            "raw.mnist": TensorSpec(shape=("B",), kind="data"),
            "inputs.mnist": TensorSpec(shape=("B", N_PIXELS), dtype="float32", kind="data"),
            "meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST),
        }
        return IO(produces=unflatten_spec(flat))

    def bind(self, ctx: WorkerCtx) -> None:
        del ctx
        self.prepare()

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        batch = np.empty(rows.stop - rows.start, dtype=np.dtype([("digit", "i8")]))
        batch["digit"] = self._labels[rows]
        out = {
            "raw.mnist": batch,
            "inputs.mnist": self._images[rows].copy(),  # never hand out the reader's buffer
        }
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([rows.start, rows.stop], dtype=np.int64)
        return out
```

### The reader contract, method by method

- **`streams`** — the stream names this reader serves. One here (`mnist`);
  a jet reader might serve `jets` and `tracks`. Downstream modules refer to
  streams by these names.
- **`with_source(filename, num, ...)`** — clone the configured reader prototype
  onto a concrete file. You configure *one* reader; the datamodule calls
  `with_source` three times to derive the train/val/test readers from
  `data.train_file` / `val_file` / `test_file`. Config-only: no file I/O here.
- **`prepare()` / `__len__()`** — main-process file probing. `prepare` is
  idempotent and lazy; `__len__` is the row count the sampler slices over.
  MNIST fits in memory, so this reader just loads both arrays up front.
- **`bind(ctx)`** — per-worker setup (open handles, allocate buffers), called
  once per dataloader worker. The only place a dataset module may touch files
  at serving time.
- **`declare_io(mode)`** — the static half of the contract. Before any data is
  read, salt compiles a plan from every module's declared `requires`/`produces`
  and validates the whole pipeline connects. A reader requires nothing and
  produces three keys here:
    - `raw.mnist` — the label record (a structured array with a `digit` field).
      The `raw.*` namespace is what label/feature processors consume.
    - `inputs.mnist` — the pixel block, with the **concrete** shape
      `("B", 784)`. Producing model-ready `inputs.*` directly (instead of
      routing through the `Features` processor) is deliberate: `Features`
      derives a stream's width from its list of named variables, which is right
      for named physics variables and wrong for 784 anonymous pixels.
    - `meta.rows` — the `[start, stop)` row window of each batch, TEST mode
      only. The H5 output sink requires it to anchor each batch of predictions
      at the correct rows of the output file.
- **`read(rows, mode)`** — the runtime half: return exactly what you declared,
  as numpy arrays for one contiguous row slice. Copy anything that aliases a
  reusable internal buffer before handing it out.

## 3. Write the config

Save as `config.yaml`. This is the whole model. Each section is explained below.

```yaml
name: MNIST_MLP

data:
  batch_size: 32
  num_workers: 0
  train_file: data/train-images-idx3-ubyte
  val_file: data/t10k-images-idx3-ubyte
  test_file: data/t10k-images-idx3-ubyte
  modules:
    reader:
      class_path: my_mnist.reader.IdxReader
    labels:
      class_path: salt.data.Labels
      init_args: {dtype_policy: int64-for-int}

model:
  class_path: salt.model.SaltModule
  init_args:
    lrs: {initial: 1.0e-4, max: 1.0e-3, end: 1.0e-5, pct_start: 0.1}
    optimizer: AdamW
    modules:
      norm:
        class_path: salt.model.modules.MaskedInputNormaliser
        init_args:
          streams: [mnist]
          global_object: mnist
      mnist_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          stream: mnist
          out_dim: 128
          dense: {hidden_layers: [256]}
      mnist_classification:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
        init_args:
          stream: mnist
          input: embed.mnist
          sequence: false
          label: digit
          class_names: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
          loss: torch.nn.CrossEntropyLoss
          dense: {hidden_layers: [64]}
      loss:
        class_path: salt.model.modules.LossSum

outputs:
  run_tasks:
    class_path: salt.outputs.RunTaskOutput
    init_args: {tasks: [mnist_classification]}

trainer:
  max_epochs: 2
  precision: 32-true
  logger: false
  default_root_dir: run
```

### `data:`

The reader is wired in by `class_path: my_mnist.reader.IdxReader` — a plain
Python import path pointing at *your* module. Any importable class satisfying
the `Reader` contract can go here; nothing needs to be added to salt itself.

`Labels` is salt's demand-driven label producer: it watches which labels the
model's tasks ask for (here `digit`, declared by the classification task below)
and extracts exactly those fields from `raw.mnist`. No label is configured
twice, and unused labels are never read.

The per-stage files reuse the 10k test set for validation — fine for a
tutorial. `num_workers: 0` and `batch_size: 32` keep the run laptop-safe.

### `model:`

Four modules, matched to salt's shipped
[`DL1.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/legacy/DL1.yaml)
— the simplest production config (global feature vector → MLP → classifier).
MNIST is structurally identical: a fixed-length global vector per sample.

- **`norm`** (`MaskedInputNormaliser`) — self-normalising input layer: it
  accumulates running mean/std during training and standardises `inputs.mnist`
  (raw 0–255 pixels) on the fly. No precomputed normalisation dictionary
  needed — which is why this tutorial uses it instead of `Normaliser`, the
  variant that reads a `norm_dict` file produced by the preprocessing pipeline.
- **`mnist_embed`** (`StreamEmbed`) — a dense embedding block:
  `normed.mnist [B, 784] → embed.mnist [B, 128]` through a 256-wide hidden
  layer. Widths are inferred from the bound input shape at plan time — note the
  config nowhere repeats "784".
- **`mnist_classification`** (`ClassificationTaskModule`) — a 10-class head on
  `embed.mnist` (`sequence: false`: one prediction per sample, not per token).
  `label: digit` is the demand that makes `Labels` serve
  `labels.mnist.digit` from the reader's `raw.mnist` record.
- **`loss`** (`LossSum`) — sums all task losses; with one task, a pass-through.

### `outputs:`

`RunTaskOutput` runs the named task heads at evaluation time and hands their
per-class probabilities to the H5 output sink, which salt attaches implicitly
for `salt test`. Column names derive from the run `name` and the task's
`class_names`: `MNIST_MLP_p0` … `MNIST_MLP_p9`, plus a
`target_mnist_classification` truth column.

Physics configs usually also carry an `InputCopyWriter` here to copy input
variables from the source H5 into the eval file; it needs source *groups* to
copy from, which an IDX file does not have, so it is omitted.

### `trainer:`

Standard Lightning settings. Your config is merged on top of salt's built-in
base config, which already sets `seed_everything: 42`, checkpointing (every
epoch, into `<default_root_dir>/ckpts/`), and a progress bar. `logger: false`
overrides the default Comet logger, which would otherwise require an API key.

## 4. Make your reader importable

```bash
export PYTHONPATH=$PWD
```

This is required, not optional. `class_path` resolution is a normal Python
import: `my_mnist` must be on `sys.path`, and salt does not implicitly add your
working directory (imports can happen after internal directory changes, so
"just run from the right directory" is not reliable). Every `salt` command
below assumes this variable is set.

## 5. Validate the graph before training

Salt can compile and check the full pipeline — reader, processors, model
modules, output sinks — without touching the data:

```bash
salt graph validate -c config.yaml
```

Expected output (plus a couple of warnings, explained below):

```text
OK [mode=FIT] 6 steps, 7 edges, plan_hash=47c3b6a9ad3c
OK [mode=VAL] 6 steps, 7 edges, plan_hash=47c3b6a9ad3c
OK [mode=TEST] 7 steps, 18 edges, plan_hash=0ec503f3ff5d
OK [mode=ONNX] 7 steps, 15 edges, plan_hash=5acc794d083a
```

Each mode (training, validation, evaluation, export) gets its own plan: note
TEST has one more step than FIT — the output sink only participates in
evaluation. If you miswire anything (say, point the task at a stream that no
module produces), this command tells you now, with the producer/consumer names,
instead of a shape error mid-training.

Two warnings are expected and harmless here: no `schema:` artifact (field
spellings can't be checked statically for a custom reader) and no export
contract declared (this tutorial does not export to ONNX).

!!! warning "Stale validate results after editing your config"

    `salt graph validate` currently caches a preprocessed copy of your config
    under `/tmp/salt_config_no_logger_*.yaml`, keyed by the config's *path* —
    so after editing `config.yaml`, a re-run can silently report the old
    result. If validate seems to ignore an edit, clear the cache:
    `rm -f /tmp/salt_config_no_logger_*.yaml`. Only the `graph` tooling is
    affected; `salt fit` and `salt test` always read your file.

## 6. Train

```bash
salt fit --config config.yaml
```

Two epochs over 60k images takes a few minutes on CPU at most. The run ends
with:

```text
`Trainer.fit` stopped: `max_epochs=2` reached.
salt fit artifacts: config.yaml in run
salt fit artifacts: checkpoints in run/ckpts
```

`run/` now contains the fully-resolved `config.yaml`, one checkpoint per epoch
in `ckpts/`, and rendered graph plots (`graph_fit.svg` — your reader is the
root node).

!!! note "Large validation loss values are expected"

    The checkpoint filenames embed the validation cross-entropy (e.g.
    `epoch=001-loss=17.51267.ckpt`), which looks alarming. A small unregularised
    MLP becomes extremely overconfident, so its few mistakes dominate the CE
    sum. Accuracy — measured next — is the meaningful metric here.

## 7. Evaluate

Point `salt test` at the resolved config and the last checkpoint (the exact
filename varies with the loss value, so use the `epoch=001*` glob):

```bash
salt test --config run/config.yaml --ckpt_path run/ckpts/epoch=001*.ckpt
```

```text
Wrote eval file run/ckpts/epoch=001-loss=17.51267__test_t10k-images-idx3-ubyte.h5
```

If `ckpts/` also contains `-v1` variants (e.g. `epoch=001-loss=17.51267-v1.ckpt`),
you ran `salt fit` more than once into the same `run/` directory — the seeded
re-run reproduces identical filenames and Lightning appends a version suffix
instead of overwriting, so the `-v1` file is simply the newer (equivalent) save:
pass `salt test` one explicit filename, or `rm -rf run` and retrain.

The eval H5 has one structured dataset per stream. Read it back and compute the
accuracy (the `hdf5plugin` import registers the compression filter salt writes
with):

```python
import glob
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

with h5py.File(glob.glob("run/ckpts/*__test_*.h5")[0]) as f:
    table = f["mnist"][:]
probs = np.stack([table[f"MNIST_MLP_p{d}"] for d in range(10)], axis=1)
accuracy = (probs.argmax(axis=1) == table["target_mnist_classification"]).mean()
print(f"test accuracy: {accuracy:.4f}")
```

```text
test accuracy: 0.9640
```

Your number will differ slightly, but with this exact recipe it should land
around **0.95–0.97**.

## What you just proved

The only code you wrote was a reader for a binary format salt has never heard
of. Everything else was **configuration of existing modules**:

- **The data side is pluggable.** Any class satisfying the `Reader` contract —
  `streams`, `with_source`, `prepare`/`__len__`, `bind`, `declare_io`, `read` —
  can feed salt, wired in with a `class_path` from your own workspace. The same
  seam serves H5 jets, ROOT ntuples, and MNIST alike.
- **Demand drives the pipeline.** You never listed which labels to load: the
  task declared `label: digit`, `Labels` narrowed to it, and the reader was
  asked for exactly that field.
- **The graph is checked statically.** `salt graph validate` compiled the full
  reader-to-sink pipeline per mode before a single byte of data was read.
- **The model side has the same seam.** `norm`, `mnist_embed`, and
  `mnist_classification` are stock modules, but each is just a `class_path`
  entry — a custom model module in your workspace plugs in the same way the
  reader did.
