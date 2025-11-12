#!/usr/bin/env python3
"""
Automatic MR labeler for GitLab CI pipelines (modern API).

- Applies labels based on .gitlab/workflow/label_mapping.yaml.
- Adds reviewers based on .gitlab/workflow/reviewers_mapping.yaml.
- Adds "Needs Review::Level 1" the first time it runs.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import gitlab
import yaml

MAPPING_PATH = Path(".gitlab/workflow/label_mapping.yaml")
REVIEWERS_MAPPING_PATH = Path(".gitlab/workflow/reviewers_mapping.yaml")
FIRST_TIME_LABELS = ["Needs Review::Level 1"]


def main() -> None:
    """Label MRs and add Reviewers based on the Labels added."""
    # Get all envrionment variables needed
    base_url = os.environ["CI_SERVER_URL"]
    token = os.environ["GITLAB_API_TOKEN"]
    project_id = os.environ["CI_PROJECT_ID"]

    # Using get because in scheduled mode there could be no open MRs
    mr_iid = os.environ.get("CI_MERGE_REQUEST_IID")

    # Authenticate to Gitlab
    gl = gitlab.Gitlab(base_url, private_token=token)
    gl.auth()

    # Get the project (and possibly MRs)
    project = gl.projects.get(project_id)

    # Load the label mapping
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    # Get the reviewer map
    with REVIEWERS_MAPPING_PATH.open("r", encoding="utf-8") as f:
        reviewers_map = yaml.safe_load(f)

    # Decide which MRs to process
    if mr_iid:
        mrs = [project.mergerequests.get(mr_iid)]
        print(f"Running in MR mode for !{mr_iid}")

    else:
        mrs = project.mergerequests.list(state="opened", all=True)
        print(f"Running in scheduled mode over {len(mrs)} open MRs")

    # Process each MR
    for mr in mrs:
        # Get the list of file changes
        changes = mr.changes()
        changed_files = [c["new_path"] for c in changes["changes"]]
        print(f"!{mr.iid}: Found {len(changed_files)} changed files.")

        # Define a set for the labels which should be added
        to_add: set[str] = set()

        # Check the changed files and add the labels accordingly
        for changed_path in changed_files:
            for label, patterns in mapping.items():
                for entry in patterns:
                    stripped_entry = str(entry).strip()
                    if (
                        stripped_entry.endswith("/") and changed_path.startswith(stripped_entry)
                    ) or fnmatch.fnmatch(changed_path, stripped_entry):
                        to_add.add(label)
                        break

        # Get the current labels and check if the FIRST_TIME_LABELS were ever added
        current = set(mr.labels)
        did_add_first_run_labels = False
        for iter_first_time_label in FIRST_TIME_LABELS:
            if iter_first_time_label not in current:
                was_ever_added = any(
                    ev.label["name"] == iter_first_time_label and ev.action == "add"
                    for ev in mr.resourcelabelevents.list(all=True)
                )
                if not was_ever_added:
                    print(f"!{mr.iid}: Adding first-run label: '{iter_first_time_label}'")
                    current.add(iter_first_time_label)
                    did_add_first_run_labels = True

        # Get one set of labels from the existing labels on the MR and the ones of the changed files
        effective_labels = set(to_add) | current

        # Collect reviewer usernames (skip labels not in mapping)
        reviewer_usernames = [
            u for lbl in sorted(effective_labels) for u in (reviewers_map.get(lbl, []))
        ]

        # Resolve usernames to GitLab user IDs
        reviewer_ids = []
        for uname in reviewer_usernames:
            stripped_uname = str(uname).lstrip("@")
            users = gl.users.list(username=stripped_uname)
            reviewer_ids.append(users[0].id)

        # Set reviewers if first-run label was added and reviewers are not already set
        if did_add_first_run_labels and reviewer_ids:
            existing_ids = [u["id"] for u in mr.reviewers]
            if not existing_ids:
                mr.reviewer_ids = sorted({*existing_ids, *reviewer_ids})
                mr.save()
                reloaded_mr = project.mergerequests.get(mr.iid)
                print(
                    f"!{reloaded_mr.iid}: Set MR reviewers: "
                    f"{[u['username'] for u in reloaded_mr.reviewers]}"
                )

        # Get the final labels that need to be added
        final_labels = sorted(current | to_add)
        if set(final_labels) != set(mr.labels):
            mr.labels = final_labels
            mr.save()
            print(f"!{mr.iid}: Updated MR labels: {final_labels}")
        else:
            print(f"!{mr.iid}: Labels unchanged: {final_labels}")


if __name__ == "__main__":
    main()
