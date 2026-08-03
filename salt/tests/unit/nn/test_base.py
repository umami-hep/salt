"""Unit tests for `SaltModelModule` (mirror of salt/model/base.py)."""

from __future__ import annotations

import pytest
from torch import nn

from salt.graph import IO, Mode
from salt.model.modules import SaltModelModule


class _Bare(SaltModelModule):
    """A minimal concrete subclass overriding nothing but `declare_io`."""

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(requires={}, produces={})


class _NoDeclareIo(SaltModelModule):
    """A manifest-only-style subclass overriding neither `declare_io` nor `forward`
    (a manifest-only-writer pattern — never entered into the plan).
    """


def test_init_assigns_unnamed_placeholder():
    m = _Bare()
    assert m.name == "unnamed"


def test_is_nn_module():
    assert issubclass(SaltModelModule, nn.Module)
    assert isinstance(_Bare(), nn.Module)


def test_bind_default_is_a_no_op():
    m = _Bare()
    assert m.bind(schema=None) is None  # type: ignore[arg-type]


def test_materialise_default_is_a_no_op():
    m = _Bare()
    assert m.materialise() is None


def test_derived_widths_default_returns_empty_dict():
    m = _Bare()
    assert m.derived_widths({"x": 3}) == {}


def test_is_sink_default_is_false():
    m = _Bare()
    assert m.is_sink() is False


def test_forward_default_raises_not_implemented():
    m = _Bare()
    with pytest.raises(NotImplementedError, match="no forward"):
        m.forward(b=None, mode=Mode.FIT)  # type: ignore[arg-type]


def test_declare_io_default_raises_not_implemented():
    # the manifest-only carve-out: a subclass that never overrides declare_io
    # (a manifest-only-writer pattern) instantiates fine (not a hard abstractmethod)...
    m = _NoDeclareIo()
    assert isinstance(m, SaltModelModule)
    # ...but calling it unimplemented is a loud NotImplementedError, not silent.
    with pytest.raises(NotImplementedError, match="no declare_io"):
        m.declare_io(Mode.FIT)


def test_manifest_only_subclass_never_needs_forward_or_declare_io():
    """Documents why neither hook is a Python abstractmethod (see base.py class
    docstring): a manifest-only outputs: section writer legitimately implements
    neither, since it is never folded into the plan / executor forward loop.
    """
    m = _NoDeclareIo()
    m.name = "manifest_only"
    assert m.name == "manifest_only"
    # never calling forward()/declare_io() on it is exactly the real usage pattern
