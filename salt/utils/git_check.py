import subprocess
from subprocess import CalledProcessError


class GitError(Exception):
    pass


def check_for_uncommitted_changes():
    try:
        subprocess.check_output(["git", "diff", "--quiet", "--exit-code"])
    except CalledProcessError:
        raise GitError("Uncommitted changes detected. Please commit them before running.") from None


def check_for_fork():
    ssh_url = "ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/salt.git"
    https_url = "https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git"
    cmd = ["git", "remote", "get-url", "origin"]
    origin = subprocess.check_output(cmd).decode("utf-8").strip()
    if origin in (ssh_url, https_url):
        raise GitError(
            f"Your origin {origin} is not a fork of the upstream repo "
            "https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt"
        )


def create_and_push_tag(tagname):
    print(f"Pushing tag {tagname}")
    check_for_fork()
    subprocess.check_output(["git", "tag", tagname, "-m", "automated salt training tag"])
    subprocess.check_output(["git", "push", "-q", "origin", "--tags"])
