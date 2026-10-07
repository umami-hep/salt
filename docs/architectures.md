# Architectures

Every tagger in Salt is built from the same three stages, so switching architecture is mostly
a matter of swapping pieces in a config (see [Configuration](configuration.md)):

1. **Embed** each input object (tracks, flows, …) with a small per-object network.
2. **Encode** the set of objects with a transformer, letting them share information.
3. **Pool** the encoded objects into a single per-jet vector and feed it to one or more
   **task heads** (flavour classification plus auxiliary tasks).

The models below differ mainly in the encoder, the pooling head, and which inputs and tasks
they use.

## At a glance

| Model | Config | Reference |
|---|---|---|
| **GN2** | [`GN2/GN2.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/GN2/GN2.yaml) | [Transforming jet flavour tagging at ATLAS](https://doi.org/10.1038/s41467-025-65059-6) (Nat. Commun., 2025) |
| **GN3V00** | [`GN3/GN3V00.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/GN3/GN3V00.yaml) | [GN3: Multi-task, Multi-modal Transformers for Jet Flavour Tagging in ATLAS](https://cds.cern.ch/record/2953652) |
| **GN3PflowMuonsV00** | [`GN3/GN3PflowMuonsV00.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/GN3/GN3PflowMuonsV00.yaml) | [GN3: Multi-task, Multi-modal Transformers for Jet Flavour Tagging in ATLAS](https://cds.cern.ch/record/2953652) |
| **GN3EPCLV01** | [`GN3/GN3EPCLV01.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/GN3/GN3EPCLV01.yaml) | [Identification of the charge of heavy-flavour jets using transformers with the ATLAS experiment](https://cds.cern.ch/record/2961896) |
| **ParT** | [`ParticleTransformer/ParT.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/ParticleTransformer/ParT.yaml) | [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772) (CMS) |
| **DeParT** | [`ParticleTransformer/DeParT.yaml`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/configs/ParticleTransformer/DeParT.yaml) | [Jet tagging using Dynamically Enhanced Particle Transformer](https://cds.cern.ch/record/2878932) (ATLAS) |

## The GN family

The GN\* models (GN = "General Network") are ATLAS' baseline flavour taggers. They all share the
same transformer-plus-attention-pooling backbone and differ mainly in which inputs they read and
which auxiliary tasks they train.

### GN2

The established ATLAS baseline. GN2 takes **tracks**, runs them through the transformer, and
pools with a learned attention weight. Alongside flavour classification it trains two auxiliary
tasks — predicting each track's origin and grouping tracks into vertices — which teach the model
the substructure that flavour tagging relies on.

### GN3V00

The **tracks-only** GN3 baseline. Relative to GN2 it already contains the full GN3 recipe:
ghost association, a looser track selection, a six-class flavour output (splitting light jets
into ud/s/gluon), two new auxiliary tasks — track-type classification and jet
transverse-momentum regression — on top of track origin and vertexing, and an updated training
setup (gated feed-forward block, register tokens, GLS loss balancing, Lion optimiser). It is
the reference point the other GN3 variants build on.

### GN3PflowMuonsV00

The full **multimodal** GN3: the same tasks as GN3V00, with two additional sources of input —
**charged and neutral particle-flow objects** as a second input collection, and **soft-muon
information** as extra variables on tracks matched to muons. The reference above also studies
the intermediate variants with only one of the two additions (GN3MuonsV00 and GN3PflowV00).

### GN3EPCLV01

Extends the multimodal GN3 further by adding **electrons** as an input and a **jet-charge** task,
on top of the tracks and particle-flow objects.

## ParT and DeParT

[**ParT**](https://arxiv.org/abs/2202.03772) (Particle Transformer, from CMS) adds
**pairwise information** between tracks — features of each *pair* of tracks are fed into the
attention step, so the model reasons about relationships directly rather than only about
individual tracks. It also replaces the pooling stage with a **class-attention head**: a single
learned "summary" token that attends over the tracks to build the per-jet vector.

[**DeParT**](https://cds.cern.ch/record/2878932) (Dynamically Enhanced Particle Transformer,
developed for ATLAS quark/gluon tagging) is ParT with a handful of well-established tricks that
make deeper models train more reliably: talking-heads attention, LayerScale, stochastic depth,
and a gated feed-forward block.

In Salt these two are the **same config with a few switches flipped** — DeParT simply turns the
extra tricks on. Both share the class-attention pooling head
([`ClassAttentionPooling`][salt.models.pooling.ClassAttentionPooling]):

| | ParT | DeParT |
|---|---|---|
| Pairwise track features | ✅ | ✅ |
| Talking-heads attention | – | ✅ |
| LayerScale | – | ✅ |
| Stochastic depth | – | ✅ |
| Gated feed-forward | – | ✅ |
