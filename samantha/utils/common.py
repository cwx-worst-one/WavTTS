import subprocess


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
