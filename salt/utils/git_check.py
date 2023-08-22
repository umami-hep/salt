import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError


class GitError(Exception):
    pass


def check_is_git_repo():
    git_workdir = Path(__file__).resolve().parent
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree", "HEAD"], cwd=git_workdir
        )
    except CalledProcessError:
        git_workdir = None
    return git_workdir


def check_for_uncommitted_changes():
    if "pytest" in sys.modules:
        return
    git_workdir = check_is_git_repo()
    if git_workdir:
        try:
            subprocess.check_output(["git", "diff", "--quiet", "--exit-code"], cwd=git_workdir)
        except CalledProcessError:
            raise GitError(
                "Uncommitted changes detected. Please commit them before running, or use --force."
            ) from None


def check_for_fork(git_workdir):
    ssh_url = "ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/salt.git"
    https_url = "https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git"
    cmd = ["git", "remote", "get-url", "origin"]
    origin = subprocess.check_output(cmd, cwd=git_workdir).decode("utf-8").strip()
    if origin in (ssh_url, https_url):
        raise GitError(
            f"Your origin {origin} is not a fork of the upstream repo "
            "https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt"
        )


def create_and_push_tag(tagname):
    print(f"Pushing tag {tagname}")
    git_workdir = check_is_git_repo()
    if git_workdir:
        check_for_fork(git_workdir)
        subprocess.check_output(
            ["git", "tag", tagname, "-m", "automated salt training tag"], cwd=git_workdir
        )
        subprocess.check_output(["git", "push", "-q", "origin", "--tags"], cwd=git_workdir)


def get_git_hash():
    git_workdir = check_is_git_repo()
    if git_workdir:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=git_workdir)
        return git_hash.decode("ascii").strip()
    return None
