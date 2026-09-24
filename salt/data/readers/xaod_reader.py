"""`xAODReader` — the `UprootReader` for xAOD POOL / DAOD_PHYSLITE files: dereferences
ElementLinks uproot can only read (1:many `link_branch`/`target_prefix` constituent streams,
1:1 `join_*` field joins).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from salt.data.readers.uproot_reader import (
    GROUP_KEYS,
    XAOD_GROUP_KEYS,
    UprootGroupConfig,
    UprootReader,
)
from salt.graph.errors import ConfigError, SchemaError

__all__ = ["xAODGroupConfig", "xAODReader"]


@dataclass(frozen=True)
class xAODGroupConfig(UprootGroupConfig):  # noqa: N801
    """Per-stream reader configuration for an `xAODReader` — adds ElementLink dereference.

    Inherits ``branches``/``prefix``/``jagged``/``pad_max`` from `UprootGroupConfig`.

    Parameters
    ----------
    link_branch : str | None, optional
        Turns a jagged stream into an ElementLink-dereferenced constituent stream
        (PHYSLITE ``GhostTrack``): the per-row link vector
        ``<unroll_prefix><link_branch>`` carries ``m_persIndex`` into the target
        container. Requires the reader's ``unroll`` to name a group.
    target_prefix : str | None, optional
        The aux-store prefix of the ElementLink target container
        (``InDetTrackParticlesAuxDyn.``); this group's ``branches`` map onto it.
        Set together with ``link_branch``.
    join_branch : str | None, optional
        A **1:1** ElementLink carried by this group's own elements
        (``btaggingLink``, resolved with this group's ``prefix``), used to JOIN
        extra fields onto the elements this group already serves — as opposed to
        ``link_branch``, which is 1:many and *creates* a constituent stream. One
        link per element, so the joined fields line up with ``branches``
        one-for-one and the group's shape is unchanged.
    join_prefix : str | None, optional
        The aux-store prefix of the join target container
        (``BTagging_AntiKt4EMPFlowAuxDyn.``). Set together with ``join_branch``.
    join_branches : dict[str, str], optional
        Field name -> *bare* branch in the join target, appended to ``branches``
        in the served field order. Required (and only valid) with ``join_branch``.
    """

    link_branch: str | None = None
    target_prefix: str | None = None
    join_branch: str | None = None
    join_prefix: str | None = None
    join_branches: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.link_branch is None) != (self.target_prefix is None):
            raise ConfigError(
                "group config: 'link_branch' and 'target_prefix' must be set together "
                "(ElementLink dereference needs both the row link vector and the target prefix)"
            )
        if self.link_branch is not None and not self.jagged:
            raise ConfigError(
                "group config: 'link_branch'/'target_prefix' are only valid for jagged "
                "(constituent) streams"
            )
        self._validate_join()

    def _validate_join(self) -> None:
        """Enforce the 1:1-join invariants (all three keys together, no overlap, not on a link)."""
        given = {
            "join_branch": self.join_branch is not None,
            "join_prefix": self.join_prefix is not None,
            "join_branches": bool(self.join_branches),
        }
        if any(given.values()) and not all(given.values()):
            missing = sorted(k for k, v in given.items() if not v)
            raise ConfigError(
                f"group config: a 1:1 join needs 'join_branch', 'join_prefix' and a non-empty "
                f"'join_branches' together — missing {missing}"
            )
        if not self.is_joined:
            return
        if self.is_linked:
            raise ConfigError(
                "group config: 'join_branch' on a group that already sets 'link_branch' — "
                "the 1:many link BUILDS this stream out of a target container, so there is "
                "no element of this group left to join a second container onto"
            )
        clash = sorted(set(self.branches) & set(self.join_branches))
        if clash:
            raise ConfigError(
                f"group config: field(s) {clash} are declared in both 'branches' and "
                "'join_branches' — a served field has exactly one source"
            )

    @property
    def is_linked(self) -> bool:
        """Whether this stream reads constituents via an ElementLink dereference."""
        return self.link_branch is not None

    @property
    def is_joined(self) -> bool:
        """Whether this stream joins extra fields on via a 1:1 ElementLink."""
        return self.join_branch is not None

    @property
    def served_branches(self) -> dict[str, str]:
        """Every field this group serves -> its bare branch: ``branches`` then ``join_branches``."""
        return {**self.branches, **self.join_branches}


class xAODReader(UprootReader):  # noqa: N801
    """`UprootReader` for xAOD POOL / DAOD_PHYSLITE files — dereferences ElementLinks.

    Same constructor as `UprootReader`; groups additionally accept the five
    `xAODGroupConfig` keys (``link_branch``/``target_prefix``/``join_branch``/
    ``join_prefix``/``join_branches``). ``unroll`` must name a group when any
    configured group is linked — ElementLink dereference is per-row-object, no
    meaning on the entry axis. A joined jagged group cannot sit under an
    ``unroll`` (the 1:1 by-index gather does not implement a third nesting
    level). Null links (``m_persKey == 0``, thinned targets) are silently
    dropped from 1:many constituent streams and refused (raised) on 1:1 joins,
    where a null link would otherwise silently desynchronise the joined fields
    from the group's own.

    Raises
    ------
    ConfigError
        On an empty/malformed group config; an ``unroll`` that names no group
        or names a ``jagged=True`` group; a linked group while ``unroll is
        None``; a joined jagged group while ``unroll`` names a group.
    SchemaError
        When a configured branch is missing, or a 1:1 join hits a null link /
        an out-of-range index / a misaligned target container.
    """

    _group_keys: ClassVar[frozenset[str]] = GROUP_KEYS | XAOD_GROUP_KEYS
    _group_config_cls: ClassVar[type[UprootGroupConfig]] = xAODGroupConfig
    groups: Mapping[str, xAODGroupConfig]

    # -- group config ----------------------------------------------------------

    @classmethod
    def _group_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        """Add the five link/join keys to the base branches/prefix/jagged/pad_max kwargs."""
        kw = super()._group_kwargs(cfg)
        kw["link_branch"] = cfg.get("link_branch")
        kw["target_prefix"] = cfg.get("target_prefix")
        kw["join_branch"] = cfg.get("join_branch")
        kw["join_prefix"] = cfg.get("join_prefix")
        kw["join_branches"] = {
            str(k): str(v) for k, v in dict(cfg.get("join_branches") or {}).items()
        }
        return kw

    def _validate_links(self) -> None:
        """Add the link/join invariants: linked needs unroll; no joined-jagged under unroll."""
        linked = [s for s, c in self.groups.items() if c.is_linked]
        if linked and self.unroll is None:
            raise ConfigError(
                f"linked groups {linked} (link_branch/target_prefix) require unroll to name a "
                "group — ElementLink dereference is per-row-object, no meaning on the entry axis"
            )
        # A join reads one link per element of the group's OWN axis, so a jagged
        # group under an unroll nests it a third level down — one the by-index
        # gather has no per-entry offset for.
        deep = [
            s
            for s, c in self.groups.items()
            if c.is_joined and c.jagged and self.unroll is not None
        ]
        if deep:
            raise ConfigError(
                f"joined groups {deep} (join_branch/join_prefix) are jagged while unroll="
                f"{self.unroll!r} — that nests the join two levels below the entry axis, which "
                "the 1:1 by-index dereference does not implement. Join on the unroll group "
                "itself, or read this group on the entry axis (unroll: null)"
            )

    # -- branch-name resolution --------------------------------------------------

    def _link_branch(self, cfg: xAODGroupConfig) -> str:
        """The on-disk link vector of a linked group (a decoration on the unroll group)."""
        assert self.unroll is not None
        return f"{self.groups[self.unroll].prefix}{cfg.link_branch}"

    @staticmethod
    def _target_branch(cfg: xAODGroupConfig, bare: str) -> str:
        """The on-disk target-container branch for an ElementLink constituent field."""
        return f"{cfg.target_prefix}{bare}"

    @staticmethod
    def _join_branch(cfg: xAODGroupConfig, bare: str) -> str:
        """The on-disk join-target branch for a 1:1-joined field."""
        return f"{cfg.join_prefix}{bare}"

    # -- schema probing ----------------------------------------------------------

    def _stream_dtypes(
        self,
        t: Any,
        avail: set[str],
        path: Path,
        stream: str,
        cfg: xAODGroupConfig,
        n_events: int,
    ) -> dict[str, str]:
        """Linked arm (target-branch dtypes) verbatim; else direct + joined + valid, in order."""
        if cfg.is_linked:
            link = self._link_branch(cfg)
            if link not in avail:
                raise SchemaError(
                    f"group {stream!r}: link branch {link!r} not in {path.name!r} "
                    "(ElementLink dereference needs the row link vector)"
                )
            fdtypes: dict[str, str] = {}
            for fieldname, bare in cfg.branches.items():
                branch = self._target_branch(cfg, bare)
                if branch not in avail:
                    raise SchemaError(
                        f"group {stream!r}: target branch {branch!r} (field {fieldname!r}) not "
                        f"in {path.name!r}; ElementLink target fields live under "
                        f"{cfg.target_prefix!r}"
                    )
                fdtypes[fieldname] = self._branch_dtype(t, branch, n_events)  # [entry][obj]
            fdtypes["valid"] = "bool"
            return fdtypes
        fd = super()._stream_dtypes(t, avail, path, stream, cfg, n_events)
        if cfg.is_joined:
            # preserve field order branches -> join_branches -> valid: pull
            # `valid` (added by the base for a jagged stream) out, join, re-add
            valid = fd.pop("valid", None)
            fd.update(self._join_dtypes(t, avail, path, stream, cfg, n_events))
            if valid is not None:
                fd["valid"] = valid
        return fd

    def _join_dtypes(
        self,
        t: Any,
        avail: set[str],
        path: Path,
        stream: str,
        cfg: xAODGroupConfig,
        n_events: int,
    ) -> dict[str, str]:
        """Validate a 1:1 join's link + target branches and capture the joined field dtypes.

        No jaggedness check on the target columns: they live in a different
        container with its own multiplicity, and the join's whole job is to
        re-shape them onto this group's axis.
        """
        link = self._on_disk(cfg, cfg.join_branch or "")
        if link not in avail:
            raise SchemaError(
                f"group {stream!r}: join link branch {link!r} not in {path.name!r} "
                f"(the 1:1 join needs one ElementLink per {stream!r} element)"
            )
        fdtypes: dict[str, str] = {}
        for fieldname, bare in cfg.join_branches.items():
            branch = self._join_branch(cfg, bare)
            if branch not in avail:
                raise SchemaError(
                    f"group {stream!r}: join target branch {branch!r} (field {fieldname!r}) "
                    f"not in {path.name!r}; joined fields live under {cfg.join_prefix!r}"
                )
            fdtypes[fieldname] = self._branch_dtype(t, branch, n_events)
        return fdtypes

    def _stream_max_mult(self, t: Any, stream: str, kept: np.ndarray) -> int:
        """Linked arm: per-row NON-NULL link count off the link vector; else defer to base."""
        cfg = self.groups[stream]
        if not cfg.is_linked:
            return super()._stream_max_mult(t, stream, kept)

        import awkward as ak

        branch = self._link_branch(cfg)
        arr = t[branch].array(library="ak")
        if self.unroll is not None:
            arr = ak.flatten(arr, axis=1)  # [entry][obj][...] -> [row][...]
        pkey = _pers_key(arr)
        arr = arr if pkey is None else _pers_index(arr)[pkey != 0]
        counts = ak.to_numpy(ak.num(arr, axis=1))
        if kept.size == 0 or counts.size == 0:
            return 0
        return int(counts[kept].max())

    # -- read ----------------------------------------------------------------

    def _direct_branches(self, cfg: xAODGroupConfig, fields: list[str]) -> dict[str, str]:
        """Empty for a linked group (no direct branches); else on-disk names for its fields."""
        if cfg.is_linked:
            return {}
        return {f: self._on_disk(cfg, cfg.branches[f]) for f in fields if f in cfg.branches}

    def _group_block(
        self,
        t: Any,
        cfg: xAODGroupConfig,
        fields: list[str],
        e0: int,
        e1: int,
        shared: dict[str, Any],
    ) -> dict[str, Any]:
        """Linked arm dereferences via `_read_linked_block`; else base direct + joined fields."""
        if cfg.is_linked:
            return self._read_linked_block(t, cfg, fields, e0, e1)
        block = super()._group_block(
            t, cfg, [f for f in fields if f in cfg.branches], e0, e1, shared
        )
        joined = [f for f in fields if f in cfg.join_branches]
        if joined:
            block.update(self._read_joined_cols(t, cfg, joined, e0, e1))
        return block

    def _read_joined_cols(
        self, t: Any, cfg: xAODGroupConfig, fields: list[str], e0: int, e1: int
    ) -> dict[str, Any]:
        """Gather 1:1-joined fields over entry block ``[e0, e1)`` onto this group's axis.

        One ElementLink per element, so ``m_persIndex`` is a straight per-element
        gather from the join target's per-entry arrays — no ragged re-grouping,
        and the result has exactly the shape the group's own branches have. Every
        failure mode is loud: a null link (``m_persKey == 0``) would silently
        break the 1:1 pairing, an index outside the target is a wrong-container
        read, and a target with a different entry count is not this file's.
        """
        import awkward as ak

        link_branch = self._on_disk(cfg, cfg.join_branch or "")
        # link + every join target in ONE grouped request (see `_read_branches`)
        tgt_names = {f: self._join_branch(cfg, cfg.join_branches[f]) for f in fields}
        raw = self._read_branches(t, [link_branch, *tgt_names.values()], e0, e1)
        links = raw[link_branch]  # [entry][elem]
        pidx = _pers_index(links)
        pkey = _pers_key(links)
        _check_single_pers_key(pkey, link_branch)
        if pkey is not None:
            n_null = int(ak.sum(ak.flatten(pkey, axis=None) == 0))
            if n_null:
                raise SchemaError(
                    f"1:1 join {link_branch!r}: {n_null:,} null ElementLink(s) (m_persKey == 0) "
                    f"in entries [{e0}, {e1}) — the joined fields have no value for those "
                    "elements. A 1:1 join cannot drop or pad them without silently "
                    "desynchronising the joined fields from the group's own"
                )
        per_elem = ak.to_numpy(ak.num(pidx, axis=1)).astype(np.int64)  # elements per entry
        idx_flat = ak.to_numpy(ak.flatten(pidx, axis=1)).astype(np.int64)

        block: dict[str, Any] = {}
        for f in fields:
            branch = tgt_names[f]
            tgt = raw[branch]  # [entry][target]
            counts = ak.to_numpy(ak.num(tgt, axis=1)).astype(np.int64)
            if counts.size != per_elem.size:
                raise SchemaError(
                    f"1:1 join {link_branch!r} -> {branch!r}: the join target has "
                    f"{counts.size} entries but the link has {per_elem.size} over the same "
                    "entry range — the two branches are not aligned"
                )
            limits = np.repeat(counts, per_elem)
            bad = (idx_flat < 0) | (idx_flat >= limits)
            if bad.any():
                first = int(np.flatnonzero(bad)[0])
                raise SchemaError(
                    f"1:1 join {link_branch!r} -> {branch!r}: m_persIndex out of range for "
                    f"{int(bad.sum()):,} element(s) (first: index {idx_flat[first]} into a "
                    f"container of {limits[first]}) — the link does not address this container"
                )
            starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
            gathered = np.asarray(ak.flatten(tgt, axis=1))[idx_flat + np.repeat(starts, per_elem)]
            arr = ak.unflatten(gathered, per_elem)  # [entry][elem]
            block[f] = ak.flatten(arr, axis=1) if self.unroll is not None else arr
        return block

    def _read_linked_block(
        self, t: Any, cfg: xAODGroupConfig, fields: list[str], e0: int, e1: int
    ) -> dict[str, Any]:
        """Dereference an ElementLink constituent stream over entry block ``[e0, e1)``.

        Reads the per-row link vector (``m_persIndex`` into the target container),
        gathers each demanded target column by index, and returns ``[row][const]``
        awkward arrays (entries flattened away) ready for the kept-row ``sel``.
        """
        import awkward as ak

        # link vector + every demanded target column in ONE grouped request
        link_branch = self._link_branch(cfg)
        tgt_names = {f: self._target_branch(cfg, cfg.branches[f]) for f in fields}
        raw = self._read_branches(t, [link_branch, *tgt_names.values()], e0, e1)
        links = raw[
            link_branch
        ]  # [event][obj][link] (struct m_persKey/m_persIndex, or plain int for synthetic)
        pidx = _pers_index(links)  # [event][obj][link] int (m_persIndex into the target)
        pkey = _pers_key(links)  # [event][obj][link] uint | None
        _check_single_pers_key(pkey, cfg.link_branch)  # type: ignore[arg-type]
        # drop null ElementLinks (m_persKey==0): PHYSLITE thins the target container, so
        # ghost links into thinned tracks are null and are not real constituents.
        if pkey is not None:
            pidx = pidx[pkey != 0]  # [event][obj][valid_link]
        # shift each event's local indices onto the flattened target axis (a per-event
        # scalar offset broadcast over [obj][link]), flatten events away, then
        # numpy-gather + re-impose the [obj][link] grouping (avoids nested ak fancy-index).
        block: dict[str, Any] = {}
        for f in fields:
            tgt = raw[tgt_names[f]]  # [event][track]
            counts = ak.to_numpy(ak.num(tgt, axis=1))
            offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]  # (n_events,) event starts
            global_idx = pidx + ak.Array(offsets)  # broadcast [event] over [event][obj][link]
            gidx_rows = ak.flatten(global_idx, axis=1)  # [row][link] in flat row order
            links_per_row = ak.to_numpy(ak.num(gidx_rows, axis=1))  # [row] -> #links
            gidx_1d = ak.to_numpy(ak.flatten(gidx_rows, axis=None)).astype(np.int64)
            vals_1d = np.asarray(ak.flatten(tgt, axis=1))[gidx_1d]  # 1-D gathered
            block[f] = ak.unflatten(vals_1d, links_per_row)  # [row][link]
        return block


def _link_member(links: Any, member: str) -> Any | None:
    """One member of an ElementLink array, or None when the array carries no records.

    uproot names an ElementLink's members after the WHOLE branch path
    (``AnalysisJetsAuxDyn.btaggingLink.m_persIndex``), not bare — real POOL files
    and hand-zipped fixtures therefore spell the same member differently, so match
    the exact name first and fall back to the dotted suffix.
    """
    fields = getattr(links, "fields", []) or []
    if member in fields:
        return links[member]
    matches = [f for f in fields if f.rsplit(".", 1)[-1] == member]
    if len(matches) > 1:
        raise SchemaError(
            f"ElementLink array carries {len(matches)} members ending in {member!r} "
            f"({matches}) — cannot tell which one addresses the target container"
        )
    return links[matches[0]] if matches else None


def _pers_index(links: Any) -> Any:
    """The ``m_persIndex`` (target-container index) of an ElementLink array, or the array
    itself when it is already a plain integer index (synthetic fixtures).
    """
    member = _link_member(links, "m_persIndex")
    return links if member is None else member


def _pers_key(links: Any) -> Any | None:
    """The ``m_persKey`` (target-container key) of an ElementLink array, or None when the
    array is a plain integer index (synthetic fixtures — no key to validate).
    """
    return _link_member(links, "m_persKey")


def _check_single_pers_key(pers_key: Any | None, link_branch: str) -> None:
    """Guard: all non-null ElementLinks must point at a single target container key.

    ``m_persKey==0`` is the null-link sentinel (a thinned target) and is ignored;
    more than one distinct NON-ZERO key means the link vector spans several
    containers, which the by-index gather cannot honor — raise rather than read the
    wrong tracks.
    """
    if pers_key is None:
        return
    import awkward as ak

    flat = ak.to_numpy(ak.flatten(pers_key, axis=None))
    nonzero = np.unique(flat[flat != 0]) if flat.size else np.empty(0)
    if nonzero.size > 1:
        raise SchemaError(
            f"ElementLink branch {link_branch!r} spans multiple target containers "
            f"(m_persKey values {nonzero.tolist()[:8]}); the by-index dereference requires a "
            "single target container per file"
        )
