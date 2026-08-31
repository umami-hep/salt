"""Tests for `MultiSampleReader`'s DEBUG-only per-worker rows-served counter."""

from __future__ import annotations

import logging
import math
import random
import re

import numpy as np
import pytest

from salt.data.base import WorkerCtx
from salt.data.readers import multisample_reader as msr
from salt.data.readers.multisample_reader import MultiSampleReader, SampleConfig
from salt.graph.spec import Mode
from salt.tests.unit.data.readers.test_multisample_reader import StubReader

_LOGGER_NAME = msr.__name__  # "salt.data.readers.multisample_reader"
_RECORD_RE = re.compile(r"(\w+)=(\d+)/(\d+)")


@pytest.fixture(autouse=True)
def _restore_salt_log_level():
    """Save/restore only the `salt` logger's level so this file never leaks state."""
    logger = logging.getLogger("salt")
    original_level = logger.level
    yield
    logger.setLevel(original_level)


def _two_sample_reader() -> MultiSampleReader:
    sig = StubReader(n=300, seed=1, offset=0.0)
    bkg = StubReader(n=700, seed=2, offset=1000.0)
    reader = MultiSampleReader(
        samples=[
            SampleConfig(name="signal", label=1, reader=sig),
            SampleConfig(name="background", label=0, reader=bkg),
        ]
    )
    reader.prepare()
    return reader


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER_NAME]


def test_salt_logger_propagates_so_caplog_needs_no_workaround() -> None:
    assert logging.getLogger("salt").propagate is True


def test_counter_counts_rows_actually_served(caplog: pytest.LogCaptureFixture) -> None:
    reader = _two_sample_reader()
    window = slice(0, 250)

    # independently derive the expectation from the index: a COUNT of window
    # positions belonging to each sample, not a max of local rows.
    sample_of = reader._sample_of[window]
    expected_served: dict[int, int] = {
        int(sid): int(np.count_nonzero(sample_of == sid)) for sid in np.unique(sample_of)
    }

    with caplog.at_level(logging.DEBUG, logger="salt"):
        caplog.clear()
        reader.read(window, Mode.FIT)

    records = _records(caplog)
    assert len(records) == 1
    text = records[0].getMessage()

    for sid, served in expected_served.items():
        name = reader.samples[sid].name
        total = len(reader.samples[sid].reader)
        assert f"{name}={served}/{total} ({served / total:.1%})" in text


def test_counter_is_monotonic_across_reads_presented_in_shuffled_order(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(msr, "_EXHAUSTION_LOG_EVERY", 1)
    reader = _two_sample_reader()
    n = len(reader)  # 1000
    window_size = 100
    windows = [slice(i, i + window_size) for i in range(0, n, window_size)]
    assert len(windows) == 10

    shuffled = list(windows)
    random.Random(7).shuffle(shuffled)
    assert shuffled != windows  # guard the test's own premise

    with caplog.at_level(logging.DEBUG, logger="salt"):
        caplog.clear()
        for window in shuffled:
            reader.read(window, Mode.FIT)
        records = _records(caplog)

    assert len(records) == len(shuffled)

    series: dict[str, list[int]] = {}
    for record in records:
        text = record.getMessage()
        for name, served_str, _total_str in _RECORD_RE.findall(text):
            series.setdefault(name, []).append(int(served_str))

    for name, values in series.items():
        assert values == sorted(values), f"{name} series not non-decreasing: {values}"

    totals = {s.name: len(s.reader) for s in reader.samples}
    for name, values in series.items():
        assert values[-1] == totals[name]


def test_worker_id_is_in_the_record(caplog: pytest.LogCaptureFixture) -> None:
    reader = _two_sample_reader()
    window = slice(0, 250)

    with caplog.at_level(logging.DEBUG, logger="salt"):
        caplog.clear()
        reader.read(window, Mode.FIT)

    records = _records(caplog)
    assert len(records) == 1
    assert "worker=main" in records[0].getMessage()


def test_cadence_emits_on_read_one_and_every_fiftieth(caplog: pytest.LogCaptureFixture) -> None:
    reader = _two_sample_reader()
    window = slice(0, 10)
    n_reads = 51  # crosses the cadence boundary: emits on read 1 and read 51

    with caplog.at_level(logging.DEBUG, logger="salt"):
        caplog.clear()
        for _ in range(n_reads):
            reader.read(window, Mode.FIT)
        records = _records(caplog)

    assert len(records) == math.ceil(n_reads / msr._EXHAUSTION_LOG_EVERY)


def test_bind_resets_the_counter_for_a_new_epoch(caplog: pytest.LogCaptureFixture) -> None:
    reader = _two_sample_reader()

    with caplog.at_level(logging.DEBUG, logger="salt"):
        caplog.clear()
        reader.read(slice(0, 250), Mode.FIT)
        reader.read(slice(250, 500), Mode.FIT)

        reader.bind(WorkerCtx(mode=Mode.FIT, read_fields={}, seed=0))

        caplog.clear()
        window = slice(0, 100)
        reader.read(window, Mode.FIT)
        records = _records(caplog)

    assert len(records) == 1
    text = records[0].getMessage()

    sample_of = reader._sample_of[window]
    expected_served: dict[int, int] = {
        int(sid): int(np.count_nonzero(sample_of == sid)) for sid in np.unique(sample_of)
    }
    for sid, served in expected_served.items():
        name = reader.samples[sid].name
        total = len(reader.samples[sid].reader)
        assert f"{name}={served}/{total} ({served / total:.1%})" in text


def test_behaviour_unchanged_between_debug_and_info_and_silent_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reader = _two_sample_reader()
    window = slice(0, 250)

    with caplog.at_level(logging.DEBUG, logger="salt"):
        caplog.clear()
        out_debug = reader.read(window, Mode.FIT)
        debug_records = _records(caplog)

    with caplog.at_level(logging.INFO, logger="salt"):
        caplog.clear()
        out_info = reader.read(window, Mode.FIT)
        info_records = _records(caplog)

    assert len(debug_records) == 1
    assert len(info_records) == 0

    assert set(out_debug) == set(out_info)
    for key in out_debug:
        a, b = out_debug[key], out_info[key]
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        names = a.dtype.names
        if names:
            for field in names:
                assert np.array_equal(a[field], b[field]), f"{key}.{field} diverged"
        else:
            assert np.array_equal(a, b), f"{key} diverged"
