"""`ProgressBar` — the thin salt-owned progress bar."""

from __future__ import annotations

from lightning.pytorch.callbacks import TQDMProgressBar


class ProgressBar(TQDMProgressBar):
    """Progress bar callback — a thin, salt-owned subclass of `TQDMProgressBar`."""
