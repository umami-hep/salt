#!/usr/bin/env python3
"""
Automatic MR labeler for GitLab CI pipelines.

- Applies labels based on `.gitlab/workflow/label_mapping.yaml`.
- Adds a one-time "first-run" label (`Needs Review::Level 1`) the first time it
  ever runs on a given MR. If a user later removes that label, it will NOT be
  re-added on subsequent runs.

Environment Variables
---------------------
GITLAB_API_TOKEN : str
    API token with `api` scope (Project Access Token recommended).
CI_PROJECT_ID : str
    ID of the current GitLab project.
CI_MERGE_REQUEST_IID : str
    Internal ID (IID) of the current merge request.
CI_SERVER_URL : str, optional
    Base URL of the GitLab instance (defaults to https://gitlab.cern.ch).
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from pathlib import Path

import gitlab
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAPPING_PATH = Path(".gitlab/workflow/label_mapping.yaml")
FIRST_TIME_LABEL = "Needs Review::Level 1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _need(var: str) -> str:
    """Retrieve a required environment variable or raise an error.

    Parameters
    ----------
    var : str
        Name of the environment variable.

    Returns
    -------
    str
        Value of the variable.

    Raises
    ------
    RuntimeError
        If the environment variable is not defined.
    """
    v = os.getenv(var)
    if not v:
        raise RuntimeError(f"Missing environment variable: {var}")
    return v


def load_mapping() -> dict[str, list[str]]:
    """Load the label mapping from the YAML file.

    Returns
    -------
    dict of str to list of str
        Dictionary mapping label names to path/glob patterns.

    Raises
    ------
    FileNotFoundError
        If the mapping file does not exist.
    """
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    mapping: dict[str, list[str]] = {}
    for label, entries in data.items():
        if entries is None:
            mapping[label] = []
        elif isinstance(entries, list):
            mapping[label] = [str(e) for e in entries]
        else:
            mapping[label] = [str(entries)]
    return mapping


def path_matches(entry: str, changed_path: str) -> bool:
    """Check whether a changed path matches a mapping entry.

    Matching logic
    --------------
    - If ``entry`` ends with a slash, treat it as a directory prefix.
    - Otherwise, treat it as a shell-style glob pattern.

    Parameters
    ----------
    entry : str
        Mapping entry from the YAML file.
    changed_path : str
        Path of the changed file in the MR.

    Returns
    -------
    bool
        True if the entry matches the changed path.
    """
    entry = entry.strip()
    if entry.endswith("/"):
        return changed_path.startswith(entry)
    return fnmatch.fnmatch(changed_path, entry)


def labels_for_changes(mapping: dict[str, list[str]], changed_paths: Iterable[str]) -> set[str]:
    """Compute all labels that should be applied to the MR from file changes.

    Parameters
    ----------
    mapping : dict of str to list of str
        Mapping of labels to path or glob patterns.
    changed_paths : iterable of str
        List of file paths changed in the merge request.

    Returns
    -------
    set of str
        Labels that should be added based on the changed files.
    """
    labels: set[str] = set()
    for path in changed_paths:
        for label, patterns in mapping.items():
            for pat in patterns:
                if path_matches(pat, path):
                    labels.add(label)
                    break
    return labels


def label_was_ever_added(mr, label_name: str) -> bool:
    """Check MR label event history to see if a label was ever added.

    Parameters
    ----------
    mr : gitlab.v4.objects.ProjectMergeRequest
        Merge request object.
    label_name : str
        Label to check.

    Returns
    -------
    bool
        True if the label has an 'add' event at any point in this MR's history.
    """
    events = mr.resourcelabelevents.list(all=True)
    for ev in events:
        # python-gitlab exposes both 'action' and 'label'
        action = getattr(ev, "action", ev.attributes.get("action"))
        label = getattr(ev, "label", ev.attributes.get("label", {}))
        ev_name = (label.get("name") if isinstance(label, dict) else None) or ""
        if ev_name == label_name and action == "add":
            return True
    return False


def add_first_time_label_if_needed(mr, current_labels: set[str], label_name: str) -> set[str]:
    """Add a special label only on the very first run for this MR.

    Logic
    -----
    - If the label is already present: do nothing.
    - Else if the label has *ever* been added before (per MR label events): do nothing.
    - Else: add it (first run).

    Parameters
    ----------
    mr : gitlab.v4.objects.ProjectMergeRequest
        Merge request object.
    current_labels : set of str
        Labels currently on the MR (pre-update).
    label_name : str
        Special label to add once.

    Returns
    -------
    set of str
        Updated label set (may be unchanged).
    """
    if label_name in current_labels:
        return current_labels
    if label_was_ever_added(mr, label_name):
        print(f"Not re-adding '{label_name}' (was added previously and likely removed).")
        return current_labels
    print(f"Adding first-run label: '{label_name}'")
    return set(current_labels) | {label_name}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the label bot logic."""
    # --- GitLab authentication ---
    base_url = os.getenv("CI_SERVER_URL", "https://gitlab.cern.ch")
    token = _need("GITLAB_API_TOKEN")

    gl = gitlab.Gitlab(base_url, private_token=token)
    gl.auth()

    project_id = _need("CI_PROJECT_ID")
    mr_iid = _need("CI_MERGE_REQUEST_IID")

    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(mr_iid)

    # --- Load mapping and collect changed files ---
    mapping = load_mapping()
    changes = mr.changes()
    changed_files = [c["new_path"] for c in changes.get("changes", [])]
    print(f"Found {len(changed_files)} changed files.")

    # --- Determine labels to add from mapping ---
    to_add = labels_for_changes(mapping, changed_files)

    # --- Compute final labels (never remove existing) ---
    current = set(mr.labels or [])
    # First-run label policy
    current = add_first_time_label_if_needed(mr, current, FIRST_TIME_LABEL)
    # Add mapped labels
    new_labels = sorted(current | to_add)

    if set(new_labels) == set(mr.labels or []):
        print(f"Labels unchanged: {sorted(new_labels)}")
        return

    mr.labels = new_labels
    mr.save()
    print(f"✅ Updated MR labels: {new_labels}")


if __name__ == "__main__":
    main()
