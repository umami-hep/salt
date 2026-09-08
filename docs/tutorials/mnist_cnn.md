# MNIST part 2 — your own CNN module

In [part 1](mnist.md) the only code you wrote was a data reader; the model was
assembled entirely from existing salt modules. In this tutorial you write your
first **model module**: a small ConvNet subclassing `SaltModelModule`, the same
base class every shipped embed, encoder, and task head is built on.

The composition lesson is the point: the CNN **replaces exactly one module** in
the part 1 config — `StreamEmbed`. The reader, normaliser, classification head,
loss, and output writing are all untouched. You then verify the swap with
`salt graph validate`, train, beat part 1's MLP accuracy (expect **~0.98** vs
**~0.96**), and render the model graph as an image. Training takes about a
minute on CPU.

## Prerequisites

Complete [part 1](mnist.md) first. Everything below runs from the same
`mnist-tutorial` directory and assumes its state: `data/` with the four IDX
files, `my_mnist/` with the reader, `config.yaml`, and the part 1 training run
in `run/`. As in part 1, every command assumes:

```bash
export PYTHONPATH=$PWD
```

The container notes from part 1 apply unchanged.

## 1. The model-module lifecycle

A model module's life has strictly separated phases, and salt calls each hook
for you at the right time:

1. **`__init__`** — capture configuration only. No layers, no tensors, no
   data. At this point the module does not know its input width.
2. **`declare_io(mode)`** — declare what the module reads and writes as
   `TensorSpec`s. Pure function of config; this is what
   `salt graph validate` compiles.
3. **`bind(schema)`** — build the actual `torch.nn` layers. The `schema`
   carries the **resolved width** of every input key, so layer sizes are
   derived, never configured twice.
4. **`forward(b, mode)`** — the per-batch math: read declared inputs from the
   bundle `b`, return the declared outputs as a dict.

You already know this shape — the reader in part 1 had the same
declare-then-serve split. The model side just adds `bind`, the one place
where resolved widths become `nn.Linear`/`nn.Conv2d` sizes.

## 2. Write the CNN

Save as `my_mnist/cnn.py`, next to the part 1 reader:

```python
"""A convolutional embedding module for the flat MNIST pixel stream."""

import math

from torch import Tensor, nn

from salt.graph.bundle import Bundle
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.modules import ResolvedSchema, SaltModelModule


class MnistCNN(SaltModelModule):
    """Embed `normed.mnist [B, F]` as `embed.mnist [B, out_dim]` through a small ConvNet."""

    def __init__(
        self,
        stream: str,
        out_dim: int,
        channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.stream = stream
        self.out_dim = out_dim
        self.channels = tuple(channels)
        self.kernel_size = kernel_size
        self.side = 0  # image side length, resolved at bind
        self.conv: nn.Module | None = None
        self.head: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({
                f"normed.{self.stream}": TensorSpec(shape=("B", "F:mnist"), dtype="float32"),
            }),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(shape=("B", self.out_dim), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        n_pixels = schema.width(f"normed.{self.stream}")
        self.side = math.isqrt(n_pixels)
        if self.side * self.side != n_pixels:
            raise ValueError(f"input width {n_pixels} is not a square image")
        layers: list[nn.Module] = []
        in_ch, side = 1, self.side
        for ch in self.channels:
            layers += [
                nn.Conv2d(in_ch, ch, self.kernel_size, padding=self.kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch, side = ch, side // 2
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch * side * side, self.out_dim)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        del mode
        x = b.get(f"normed.{self.stream}")
        x = x.reshape(x.shape[0], 1, self.side, self.side)
        assert self.conv is not None and self.head is not None, "forward before bind()"
        out = self.head(self.conv(x).flatten(1))
        return {f"embed.{self.stream}": out}
```

`SaltModelModule` and `ResolvedSchema` come straight from `salt.model.modules` —
the same public API the shipped modules use.

### What a model module has to implement

`MnistCNN` requires `normed.mnist` with a **symbolic** feature width and
produces `embed.mnist` with a **concrete** one, so `bind` derives its conv
stack from whatever width the planner resolves rather than a hardcoded
`784`, and `forward` reshapes that same flat tensor into a `1×28×28` image
without changing what the normaliser upstream or the classification head
downstream see it as. The full method-by-method reference, including when to
declare a width as `None` and implement `derived_widths()` instead, lives in
[Model modules](../modules/model.md).

## 3. The config: swap one module

Copy the part 1 config and edit:

```bash
cp config.yaml config_cnn.yaml
```

Three bookkeeping edits keep the two runs separate — rename
`name: MNIST_MLP` to `name: MNIST_CNN` (this also renames the output H5
columns) and `default_root_dir: run` to `default_root_dir: run_cnn`. Then the
one real change, the `mnist_embed` block:

```yaml
      mnist_embed:
        class_path: my_mnist.cnn.MnistCNN
        init_args:
          stream: mnist
          out_dim: 128
          channels: [16, 32]
```

The full diff against part 1:

```diff
-name: MNIST_MLP
+name: MNIST_CNN
...
-        class_path: salt.model.modules.StreamEmbed
+        class_path: my_mnist.cnn.MnistCNN
...
-          dense: {hidden_layers: [256]}
+          channels: [16, 32]
...
-  default_root_dir: run
+  default_root_dir: run_cnn
```

That is the whole model change: one `class_path` now points into your
workspace instead of salt, exactly like the reader's did in part 1. Every
other module block is untouched — `mnist_classification` still consumes
`embed.mnist` and cannot tell (and does not care) that an MLP became a CNN.

## 4. Validate and inspect the plan

```bash
salt graph validate -c config_cnn.yaml
```

```text
OK [mode=FIT] 6 steps, 7 edges, plan_hash=f95f876fa411
OK [mode=VAL] 6 steps, 7 edges, plan_hash=f95f876fa411
OK [mode=TEST] 7 steps, 18 edges, plan_hash=2546ebe2aec9
OK [mode=ONNX] 7 steps, 15 edges, plan_hash=bc623c36df6c
```

Same step and edge counts as part 1 — the graph *topology* is identical, only
the plan hashes changed because one node's implementation did. You can see
your module seated in the compiled plan:

```bash
salt graph plan -c config_cnn.yaml
```

```text
plan [mode=FIT] 6 steps  plan_hash=f95f876fa411...
   1. reader               after (no inputs)          (needs nothing)
   2. labels               after reader               (needs raw.mnist)
   3. norm                 after reader               (needs inputs.mnist)
   4. mnist_embed          after norm                 (needs normed.mnist)
   5. mnist_classification after mnist_embed          (needs embed.mnist, labels.mnist.digit)
   6. loss                 after mnist_classification (needs losses.mnist_classification)
```

Step 4 is now your `MnistCNN`. The same part 1 caveat applies if you edit the
config and re-validate: clear `/tmp/salt_config_no_logger_*.yaml` first.

## 5. Train

```bash
salt fit --config config_cnn.yaml
```

```text
`Trainer.fit` stopped: `max_epochs=2` reached.
salt fit artifacts: config.yaml in run_cnn
salt fit artifacts: checkpoints in run_cnn/ckpts
```

Two epochs take about a minute on CPU — roughly 1.5× the MLP's time.
The checkpoint filenames embed a much smaller validation loss than part 1's
(single digits instead of ~17): the CNN generalises better, so its
cross-entropy is not dominated by a few confidently-wrong outliers.

## 6. Evaluate and compare

As in part 1, evaluate the last checkpoint (glob because the loss value in the
filename varies):

```bash
salt test --config run_cnn/config.yaml --ckpt_path run_cnn/ckpts/epoch=001*.ckpt
```

Then the same five-line accuracy check, pointed at the new run and the
`MNIST_CNN` columns:

```python
import glob
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

with h5py.File(glob.glob("run_cnn/ckpts/*__test_*.h5")[0]) as f:
    table = f["mnist"][:]
probs = np.stack([table[f"MNIST_CNN_p{d}"] for d in range(10)], axis=1)
accuracy = (probs.argmax(axis=1) == table["target_mnist_classification"]).mean()
print(f"test accuracy: {accuracy:.4f}")
```

```text
test accuracy: 0.9816
```

With this exact recipe (seed 42, 2 epochs, batch 32) expect **~0.975–0.985**
— clearly above the ~0.96 MLP from part 1, with the same data, normalisation,
head, and training schedule. The only variable is the module you wrote.

## 7. Finale: render the graph

Salt can draw the compiled plan as an image. This needs the Graphviz `dot`
binary: the salt containers ship it, but on a native install you'll need to
install Graphviz first — see [Setup: Install Graphviz](../setup.md#install-graphviz).

```bash
salt graph plot -c config_cnn.yaml -o graph_cnn.png
```

```text
wrote DOT to graph_cnn.dot
wrote PNG to graph_cnn.png (graphviz/dot)
wrote PDF to graph_cnn.pdf (graphviz/dot)
```

Open `graph_cnn.png`: one card per module, each listing its in/out keys with
resolved shapes. The `mnist_embed` card is labelled with *your* class name,
`MnistCNN`, reading `normed.mnist (B, 784)` and producing
`embed.mnist (B, 128)` — the static render works from `declare_io` alone,
which is why every shape is known before any data is read.

## What you just proved

- **The model side is pluggable, exactly like the data side.** Part 1 wired a
  custom reader in with a `class_path`; part 2 did the same for a model
  module. Both sit in *your* workspace; salt gained no MNIST code.
- **The lifecycle keeps config and construction apart.** `__init__` captured
  knobs, `declare_io` published the interface, `bind` built layers from
  resolved widths, `forward` did the math. No width was configured twice.
- **Modules own their view of the data.** The reader kept serving flat
  vectors; the CNN reshaped them internally. Nothing upstream had to change.
- **Swapping an implementation is a one-block config edit.** Topology,
  neighbours, and outputs all held still — validated statically, then
  confirmed by a ~2 point accuracy gain over part 1's MLP.
